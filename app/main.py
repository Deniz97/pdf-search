from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import init_db, get_db
from app.routes import router
from app.models import Document, Chunk


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
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


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat page - interactive chat interface."""
    return templates.TemplateResponse(request=request, name="chat.html", context={})


@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Documents list page - shows all uploaded documents."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
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

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    # Get document
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
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

    return templates.TemplateResponse(
        request=request,
        name="document_detail.html",
        context={"document": document, "chunks": chunks},
    )
