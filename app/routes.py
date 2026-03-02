import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from starlette.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from openai import OpenAI

from app.config import settings
from app.database import get_db, sync_engine
from app.models import Chunk, Document
from app.services.embeddings import ask_llm, get_embedding, get_embeddings, rerank
from app.services.ingest import ingest_pdf
from app.services.pdf_processing import extract_chunks_from_pdf
from app.services.search import enhanced_search
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChunkResult,
    DocumentResponse,
    EnhancedSearchResponse,
    QueryRequest,
    QueryResponse,
    SearchOnlyRequest,
    UploadResponse,
)

# Templates for HTML responses
BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter()

# OpenAI client for chat
openai_client = OpenAI(api_key=settings.openai_api_key)

# Tool definitions matching chat.py
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search across ALL uploaded documents for chunks semantically similar "
                "to the query. Returns the most relevant text passages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural language search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_book",
            "description": (
                "Search within a SPECIFIC document (by filename) for chunks semantically "
                "similar to the query. Use this when the user asks about a specific PDF."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_name": {
                        "type": "string",
                        "description": "The exact filename of the PDF document to search in",
                    },
                    "query": {
                        "type": "string",
                        "description": "The natural language search query",
                    },
                },
                "required": ["book_name", "query"],
            },
        },
    },
]


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4()
    file_path = upload_dir / f"{file_id}.pdf"

    content = await file.read()
    file_path.write_bytes(content)

    try:
        chunks = extract_chunks_from_pdf(str(file_path), doc_id=None)
        if not chunks:
            raise HTTPException(
                status_code=422, detail="Could not extract text from PDF"
            )

        page_count = max(c["page_number"] for c in chunks)
        doc = Document(filename=file.filename, page_count=page_count)
        db.add(doc)
        await db.flush()

        texts = [c["content"] for c in chunks]
        embeddings = get_embeddings(texts)

        for chunk_data, embedding in zip(chunks, embeddings):
            chunk = Chunk(
                document_id=doc.id,
                content=chunk_data["content"],
                page_number=chunk_data["page_number"],
                chunk_index=chunk_data["chunk_index"],
                chunk_type=chunk_data.get("chunk_type"),
                bbox=chunk_data.get("bbox"),
                embedding=embedding,
            )
            db.add(chunk)

        await db.commit()
        await db.refresh(doc)

        return UploadResponse(
            document=DocumentResponse.model_validate(doc),
            chunks_created=len(chunks),
            message=f"Successfully processed '{file.filename}' into {len(chunks)} chunks",
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")
    finally:
        file_path.unlink(missing_ok=True)


@router.post("/documents/query", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    response: EnhancedSearchResponse = await enhanced_search(
        body.query, db, top_k=body.top_k
    )

    # Flatten matched chunks from all documents, sort by score, take top_k
    all_chunks: list[tuple[str, int | None, int, str | None, float]] = []
    for doc_result in response.results:
        for c in doc_result.matched_chunks:
            all_chunks.append(
                (c.content, c.page_number, c.chunk_index, c.chunk_type, c.score)
            )
    all_chunks.sort(key=lambda x: x[4], reverse=True)
    top_chunks = all_chunks[: body.top_k]

    if not top_chunks:
        raise HTTPException(
            status_code=404, detail="No documents found. Upload a PDF first."
        )

    sources = [
        ChunkResult(
            content=content,
            page_number=page_number,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            score=score,
        )
        for content, page_number, chunk_index, chunk_type, score in top_chunks
    ]
    context_texts = [c[0] for c in top_chunks]

    answer = ask_llm(body.query, context_texts)

    return QueryResponse(query=body.query, answer=answer, sources=sources)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return result.scalars().all()


@router.post("/documents/search")
async def search_documents(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Search documents and return matched chunks with document info (no LLM answer).
    Returns HTML if requested via HTMX, JSON otherwise.
    Accepts form data (HTMX) or JSON body (API).
    """
    # Handle form data (HTMX) vs JSON (API)
    # FastAPI parses body params before the handler runs; form data would fail JSON
    # validation, so we parse manually based on content type.
    content_type = (request.headers.get("content-type") or "").lower()
    if (
        request.headers.get("hx-request")
        or "application/x-www-form-urlencoded" in content_type
        or "multipart/form-data" in content_type
    ):
        # HTMX / form submission
        form_data = await request.form()
        raw_query = form_data.get("query")
        if not raw_query or not isinstance(raw_query, str):
            raise HTTPException(status_code=422, detail="query is required")
        search_query: str = raw_query.strip()
        if not search_query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        raw_top_k = form_data.get("top_k", "10")
        top_k_str = raw_top_k if isinstance(raw_top_k, str) else "10"
        try:
            search_top_k = int(top_k_str)
        except ValueError:
            search_top_k = 10
        search_top_k = max(1, min(50, search_top_k))
    else:
        # API sends JSON body
        try:
            raw = await request.json()
        except Exception:
            raise HTTPException(
                status_code=422, detail="Request body must be valid JSON"
            )
        body = SearchOnlyRequest.model_validate(raw)
        search_query = body.query
        search_top_k = body.top_k

    response: EnhancedSearchResponse = await enhanced_search(
        search_query, db, top_k=search_top_k
    )

    if not response.results:
        if request.headers.get("hx-request"):
            return templates.TemplateResponse(
                request=request,
                name="search_results.html",
                context={
                    "response": response,
                    "query": search_query,
                },
            )
        raise HTTPException(
            status_code=404, detail="No documents found. Upload a PDF first."
        )

    if request.headers.get("hx-request"):
        # Convert to dict for reliable template rendering (Pydantic models can be finicky in Jinja)
        response_dict = response.model_dump(mode="json")
        return templates.TemplateResponse(
            request=request,
            name="search_results.html",
            context={
                "response": response_dict,
                "query": search_query,
            },
        )

    # API: return full enhanced search response
    return response


async def build_system_prompt(db: AsyncSession) -> str:
    """Build system prompt with document list."""
    result = await db.execute(select(Document).order_by(Document.filename))
    documents = result.scalars().all()

    if not documents:
        return (
            "You are a helpful assistant with access to a document library. "
            "However, no documents have been uploaded yet. "
            "Format your answers using Markdown when appropriate."
        )

    doc_list = "\n".join(f"  - {d.filename} ({d.page_count} pages)" for d in documents)
    return (
        "You are a helpful assistant with access to a document library. "
        "You can search through the documents to answer user questions.\n\n"
        "Available documents:\n"
        f"{doc_list}\n\n"
        "Use the `search` tool to find relevant information across all documents, "
        "or `search_in_book` to search within a specific document by its filename. "
        "Always cite which document and page your information comes from. "
        "If you can't find relevant information, say so honestly.\n\n"
        "Format your answers using Markdown: use **bold** for emphasis, bullet points "
        "for lists, `inline code` for technical terms, and headings when structuring "
        "longer answers. The answer will be displayed with markdown rendering."
    )


async def execute_tool_call(
    name: str, arguments: dict, db: AsyncSession, top_k: int = 8
) -> str:
    """Execute a tool call and return JSON results."""
    query_embedding = get_embedding(arguments["query"])
    fetch_limit = top_k * settings.rerank_top_n_multiplier

    if name == "search":
        result = await db.execute(
            text("""
                SELECT c.content, c.page_number, c.chunk_index,
                       d.filename,
                       c.embedding <=> :embedding AS distance
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> :embedding
                LIMIT :limit
            """),
            {"embedding": str(query_embedding), "limit": fetch_limit},
        )
    elif name == "search_in_book":
        result = await db.execute(
            text("""
                SELECT c.content, c.page_number, c.chunk_index,
                       d.filename,
                       c.embedding <=> :embedding AS distance
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.filename = :book_name
                ORDER BY c.embedding <=> :embedding
                LIMIT :limit
            """),
            {
                "embedding": str(query_embedding),
                "book_name": arguments["book_name"],
                "limit": fetch_limit,
            },
        )
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})

    rows = result.fetchall()
    candidates = [
        {
            "document": r.filename,
            "page": r.page_number,
            "chunk_index": r.chunk_index,
            "score": round(1 - r.distance, 4),
            "content": r.content,
        }
        for r in rows
    ]

    results = rerank(arguments["query"], candidates, top_n=top_k)
    return json.dumps(results, ensure_ascii=False)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Chat endpoint with tool calling support (replicates CLI chatbot)."""
    # Build messages list for OpenAI
    messages = []

    # Add system prompt if not already present
    has_system = any(msg.role == "system" for msg in body.messages)
    if not has_system:
        system_prompt = await build_system_prompt(db)
        messages.append({"role": "system", "content": system_prompt})

    # Convert request messages to OpenAI format
    for msg in body.messages:
        if msg.role == "tool":
            # Tool messages need tool_call_id
            msg_dict = {
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id or "",
            }
        else:
            msg_dict = {"role": msg.role, "content": msg.content}
        messages.append(msg_dict)

    # Loop until we get a final answer (handle tool calls server-side)
    while True:
        # Call OpenAI with tool calling
        response = openai_client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            tools=TOOLS,  # type: ignore[arg-type]
        )

        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            # Add assistant message with tool calls to conversation
            messages.append(msg.model_dump())

            # Execute all tool calls and add results to messages
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name  # type: ignore[attr-defined]
                fn_args = json.loads(tool_call.function.arguments)  # type: ignore[attr-defined]

                result_json = await execute_tool_call(fn_name, fn_args, db)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_json,
                    }
                )
            # Continue loop to get final answer after tool execution
            continue

        # No tool calls - return final assistant message
        content = msg.content or ""
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            tool_calls=None,
        )


class ToolExecuteRequest(BaseModel):
    tool_name: str
    arguments: dict


@router.post("/chat/tool")
async def execute_chat_tool(
    body: ToolExecuteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Execute a tool call from chat and return results."""
    result_json = await execute_tool_call(body.tool_name, body.arguments, db)
    return {"result": result_json}


@router.post("/documents/{document_id}/reprocess")
async def reprocess_document(
    request: Request,
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Reprocess an errored document. Returns HTML if HTMX request, JSON otherwise."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    # Get document
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not document.path:
        raise HTTPException(
            status_code=400,
            detail="Document has no file path. Only documents ingested from file system can be reprocessed.",
        )

    pdf_path = Path(document.path)
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404, detail=f"PDF file not found at path: {document.path}"
        )

    # Run sync ingest_pdf in executor
    def _reprocess():
        with Session(sync_engine) as session:
            return ingest_pdf(session, pdf_path, reingest=True, path_str=document.path)

    try:
        chunk_count = await asyncio.to_thread(_reprocess)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reprocessing document: {e}")

    # Refresh document with enrichment
    await db.refresh(document)
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(Document)
        .options(selectinload(Document.enrichment))
        .where(Document.id == doc_uuid)
    )
    document = result.scalar_one()

    # Return HTML if HTMX request, JSON otherwise
    if request and request.headers.get("hx-request"):
        return templates.TemplateResponse(
            request=request,
            name="document_card_fragment.html",
            context={"document": document},
        )
    else:
        return {
            "message": f"Successfully reprocessed document. Created {chunk_count} chunks.",
            "chunks_created": chunk_count,
            "document": DocumentResponse.model_validate(document),
        }
