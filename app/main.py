import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Depends

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import init_db, get_db
from app.routes import router
from app.models import Document, Chunk, Memory, SearchTestQuestion
from app.services.search import enhanced_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    import sqlalchemy.exc

    logging.getLogger("app.main").info("lifespan: starting init_db...")
    try:
        await asyncio.wait_for(init_db(), timeout=60)
    except asyncio.TimeoutError as e:
        logging.getLogger("app.main").error("lifespan: init_db timed out")
        raise RuntimeError(
            "Database initialization timed out after 60s. "
            "Check DB connectivity and locks."
        ) from e
    except sqlalchemy.exc.DBAPIError as e:
        if "lock timeout" in str(e.orig or e).lower():
            raise RuntimeError(
                "Database migration blocked: another connection holds a lock on "
                "'documents'. Close other DB clients (uvicorn, search-eval, ingest, "
                "etc.) and try again. Or run 'make migrate' when DB is idle."
            ) from e
        raise
    logging.getLogger("app.main").info("lifespan: init_db complete, application ready")
    yield


app = FastAPI(
    title="PDF Search",
    description="Upload PDFs and search through them using natural language",
    version="0.1.0",
    lifespan=lifespan,
)

# Templates and static files
BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# CORS middleware (still useful for API access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def search_page(request: Request):
    """Search page - query box with matched sections."""
    return templates.TemplateResponse(request=request, name="search.html", context={})


@app.get("/search/result", response_class=HTMLResponse)
async def search_result_detail_page(
    request: Request,
    q: str,
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Search result detail page - shows one document's matches for a query."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Search query is required")

    response = await enhanced_search(q.strip(), db, top_k=20)
    doc_result = next(
        (r for r in response.results if r.document.id == doc_uuid),
        None,
    )

    if not doc_result:
        raise HTTPException(
            status_code=404,
            detail="Document not found in search results. Try searching again.",
        )

    return templates.TemplateResponse(
        request=request,
        name="search_result_detail.html",
        context={
            "query": q.strip(),
            "doc_result": doc_result,
            "response": response,
        },
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat page - interactive chat interface."""
    return templates.TemplateResponse(request=request, name="chat.html", context={})


@app.get("/test-questions", response_class=HTMLResponse)
async def test_questions_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Test questions page - shows generated search eval test questions."""
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(SearchTestQuestion)
        .options(
            selectinload(SearchTestQuestion.target_document),
            selectinload(SearchTestQuestion.target_chunk),
        )
        .order_by(SearchTestQuestion.created_at.desc())
    )
    questions = result.scalars().all()
    return templates.TemplateResponse(
        request=request,
        name="test_questions.html",
        context={"questions": questions},
    )


@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Documents list page - shows all uploaded documents."""
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(Document)
        .options(selectinload(Document.enrichment))
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()
    return templates.TemplateResponse(
        request=request, name="documents.html", context={"documents": documents}
    )


@app.get("/documents/{document_id}", response_class=HTMLResponse)
async def document_detail_page(
    request: Request, document_id: str, db: AsyncSession = Depends(get_db)
):
    """Document detail page - shows document info and all its chunks."""
    from uuid import UUID
    from sqlalchemy.orm import selectinload

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    # Get document with enrichment
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.enrichment))
        .where(Document.id == doc_uuid)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get chunks for this document
    chunks_result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == doc_uuid)
        .order_by(Chunk.page_number, Chunk.chunk_index)
    )
    chunks = chunks_result.scalars().all()

    # Get memories for all chunks
    memories_by_chunk = {}
    if chunks:
        chunk_ids = [chunk.id for chunk in chunks]
        memories_result = await db.execute(
            select(Memory).where(Memory.chunk_id.in_(chunk_ids))
        )
        memories = memories_result.scalars().all()

        # Group memories by chunk_id
        for memory in memories:
            if memory.chunk_id not in memories_by_chunk:
                memories_by_chunk[memory.chunk_id] = []
            memories_by_chunk[memory.chunk_id].append(memory)

    return templates.TemplateResponse(
        request=request,
        name="document_detail.html",
        context={
            "document": document,
            "chunks": chunks,
            "memories_by_chunk": memories_by_chunk,
        },
    )
