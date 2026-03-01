import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from openai import OpenAI

from app.config import settings
from app.database import get_db
from app.directory_service import (
    ingest_pdf_from_directory,
    scan_directory_for_pdfs,
    sync_directory,
)
from app.embeddings import ask_llm, get_embedding, get_embeddings
from app.models import Chunk, Directory, Document
from app.pdf_processing import chunk_text, extract_text_from_pdf
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChunkResult,
    ChunkResultWithDocument,
    DirectoryCreateRequest,
    DirectoryIngestRequest,
    DirectoryIngestResponse,
    DirectoryResponse,
    DirectorySyncResponse,
    DocumentResponse,
    QueryRequest,
    QueryResponse,
    SearchOnlyRequest,
    SearchOnlyResponse,
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
        pages = extract_text_from_pdf(str(file_path))
        if not pages:
            raise HTTPException(
                status_code=422, detail="Could not extract text from PDF"
            )

        chunks = chunk_text(pages)

        doc = Document(filename=file.filename, page_count=len(pages))
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
    query_embedding = get_embedding(body.query)

    result = await db.execute(
        text("""
            SELECT id, content, page_number, chunk_index,
                   embedding <=> :embedding AS distance
            FROM chunks
            ORDER BY embedding <=> :embedding
            LIMIT :limit
        """),
        {"embedding": str(query_embedding), "limit": body.top_k},
    )
    rows = result.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404, detail="No documents found. Upload a PDF first."
        )

    sources = []
    context_texts = []
    for row in rows:
        score = 1 - row.distance
        sources.append(
            ChunkResult(
                content=row.content,
                page_number=row.page_number,
                chunk_index=row.chunk_index,
                score=round(score, 4),
            )
        )
        context_texts.append(row.content)

    answer = ask_llm(body.query, context_texts)

    return QueryResponse(query=body.query, answer=answer, sources=sources)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return result.scalars().all()


@router.post("/documents/search")
async def search_documents(
    body: SearchOnlyRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Search documents and return matched chunks with document info (no LLM answer).
    Returns HTML if requested via HTMX, JSON otherwise.
    """
    query_embedding = get_embedding(body.query)

    result = await db.execute(
        text("""
            SELECT c.id, c.content, c.page_number, c.chunk_index,
                   c.document_id, d.filename,
                   c.embedding <=> :embedding AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> :embedding
            LIMIT :limit
        """),
        {"embedding": str(query_embedding), "limit": body.top_k},
    )
    rows = result.fetchall()

    if not rows:
        # Check if this is an HTMX request
        if request.headers.get("hx-request"):
            return templates.TemplateResponse(
                request=request,
                name="search_results.html",
                context={"results": []}
            )
        raise HTTPException(
            status_code=404, detail="No documents found. Upload a PDF first."
        )

    results = []
    for row in rows:
        score = 1 - row.distance
        results.append(
            ChunkResultWithDocument(
                content=row.content,
                page_number=row.page_number,
                chunk_index=row.chunk_index,
                score=round(score, 4),
                document_id=row.document_id,
                document_filename=row.filename,
            )
        )

    # Return HTML for HTMX requests, JSON for API requests
    if request.headers.get("hx-request"):
        return templates.TemplateResponse(
            request=request,
            name="search_results.html",
            context={"results": results}
        )
    
    return SearchOnlyResponse(query=body.query, results=results)


async def build_system_prompt(db: AsyncSession) -> str:
    """Build system prompt with document list."""
    result = await db.execute(
        select(Document).order_by(Document.filename)
    )
    documents = result.scalars().all()
    
    if not documents:
        return (
            "You are a helpful assistant with access to a document library. "
            "However, no documents have been uploaded yet."
        )
    
    doc_list = "\n".join(
        f"  - {d.filename} ({d.page_count} pages)" for d in documents
    )
    return (
        "You are a helpful assistant with access to a document library. "
        "You can search through the documents to answer user questions.\n\n"
        "Available documents:\n"
        f"{doc_list}\n\n"
        "Use the `search` tool to find relevant information across all documents, "
        "or `search_in_book` to search within a specific document by its filename. "
        "Always cite which document and page your information comes from. "
        "If you can't find relevant information, say so honestly."
    )


async def execute_tool_call(
    name: str, arguments: dict, db: AsyncSession, top_k: int = 8
) -> str:
    """Execute a tool call and return JSON results."""
    query_embedding = get_embedding(arguments["query"])
    
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
            {"embedding": str(query_embedding), "limit": top_k},
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
                "limit": top_k,
            },
        )
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})
    
    rows = result.fetchall()
    results = [
        {
            "document": r.filename,
            "page": r.page_number,
            "chunk_index": r.chunk_index,
            "score": round(1 - r.distance, 4),
            "content": r.content,
        }
        for r in rows
    ]
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
    
    # Call OpenAI with tool calling
    response = openai_client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        tools=TOOLS,
    )
    
    choice = response.choices[0]
    msg = choice.message
    
    if msg.tool_calls:
        # Return tool calls for the client to handle
        tool_calls = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
        return ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            tool_calls=tool_calls,
        )
    
    # Return assistant message
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


# Directory management routes


@router.post("/directories", response_model=DirectoryResponse)
async def create_directory(
    body: DirectoryCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Add a new directory to monitor."""
    directory_path = Path(body.path)
    
    if not directory_path.exists():
        raise HTTPException(status_code=400, detail="Directory path does not exist")
    
    if not directory_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    # Check if directory already exists
    existing = await db.execute(
        select(Directory).where(Directory.path == str(directory_path.absolute()))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Directory already registered")
    
    directory = Directory(
        path=str(directory_path.absolute()),
        name=body.name,
        status="active",
    )
    db.add(directory)
    await db.commit()
    await db.refresh(directory)
    
    return directory


@router.get("/directories", response_model=list[DirectoryResponse])
async def list_directories(db: AsyncSession = Depends(get_db)):
    """List all registered directories."""
    result = await db.execute(select(Directory).order_by(Directory.created_at.desc()))
    return result.scalars().all()


@router.delete("/directories/{directory_id}")
async def delete_directory(
    directory_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a directory and optionally its documents."""
    directory = await db.get(Directory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")
    
    # Delete directory (documents will have directory_id set to NULL due to SET NULL)
    await db.delete(directory)
    await db.commit()
    
    return {"message": "Directory deleted successfully"}


@router.post("/directories/{directory_id}/sync", response_model=DirectorySyncResponse)
async def sync_directory_endpoint(
    directory_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Scan directory for PDFs and register them in database."""
    directory = await db.get(Directory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")
    
    directory_path = Path(directory.path)
    if not directory_path.exists():
        raise HTTPException(status_code=400, detail="Directory path no longer exists")
    
    total_found, new_count, existing_count = await sync_directory(
        db, directory_id, directory_path
    )
    
    await db.refresh(directory)
    
    return DirectorySyncResponse(
        directory=DirectoryResponse.model_validate(directory),
        pdfs_found=total_found,
        pdfs_new=new_count,
        pdfs_existing=existing_count,
        message=f"Found {total_found} PDFs ({new_count} new, {existing_count} existing)",
    )


@router.post(
    "/directories/{directory_id}/ingest", response_model=DirectoryIngestResponse
)
async def ingest_directory_endpoint(
    directory_id: str,
    body: DirectoryIngestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Ingest PDFs from a directory (with optional limit for incremental ingestion)."""
    directory = await db.get(Directory, directory_id)
    if not directory:
        raise HTTPException(status_code=404, detail="Directory not found")
    
    directory_path = Path(directory.path)
    if not directory_path.exists():
        raise HTTPException(status_code=400, detail="Directory path no longer exists")
    
    # Get PDFs from directory
    pdfs = scan_directory_for_pdfs(directory_path)
    
    # Get documents that need ingestion (waiting or errored status)
    result = await db.execute(
        text(
            "SELECT id, filename, file_path FROM documents "
            "WHERE directory_id = :dir_id AND processed_status IN ('waiting', 'errored') "
            "ORDER BY created_at ASC"
        ),
        {"dir_id": directory_id},
    )
    docs_to_ingest = result.fetchall()
    
    # Create mapping of filename -> (doc_id, relative_path)
    doc_map = {doc.filename: (str(doc.id), doc.file_path) for doc in docs_to_ingest}
    
    # Filter PDFs to only those that have document records
    pdfs_to_process = [
        (pdf_path, rel_path)
        for pdf_path, rel_path in pdfs
        if pdf_path.name in doc_map
    ]
    
    # Apply limit if specified
    if body.limit:
        pdfs_to_process = pdfs_to_process[: body.limit]
    
    if not pdfs_to_process:
        await db.refresh(directory)
        return DirectoryIngestResponse(
            directory=DirectoryResponse.model_validate(directory),
            pdfs_ingested=0,
            chunks_created=0,
            message="No PDFs to ingest (all already processed or no documents found)",
        )
    
    # Use sync engine for ingestion (like ingest.py)
    engine = create_engine(settings.database_url_sync)
    total_chunks = 0
    
    with Session(engine) as session:
        for pdf_path, rel_path in pdfs_to_process:
            doc_id, stored_rel_path = doc_map[pdf_path.name]
            try:
                chunks = ingest_pdf_from_directory(
                    session, pdf_path, stored_rel_path or rel_path, directory_id, doc_id
                )
                total_chunks += chunks
            except Exception as e:
                # Continue with other PDFs on error
                print(f"Error ingesting {pdf_path.name}: {e}")
                continue
    
    await db.refresh(directory)
    
    return DirectoryIngestResponse(
        directory=DirectoryResponse.model_validate(directory),
        pdfs_ingested=len(pdfs_to_process),
        chunks_created=total_chunks,
        message=f"Successfully ingested {len(pdfs_to_process)} PDFs ({total_chunks} chunks)",
    )
