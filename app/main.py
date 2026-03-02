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
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import dispose_all, init_db, get_db
from app.routes import router
from app.models import Document, Chunk, Memory, SearchTestQuestion, SearchEvalRun
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
    except sqlalchemy.exc.DBAPIError:
        raise
    logging.getLogger("app.main").info("lifespan: init_db complete, application ready")
    yield
    logging.getLogger("app.main").info("lifespan: disposing database engines...")
    dispose_all()


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


@app.get("/search-eval-runs", response_class=HTMLResponse)
async def eval_runs_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Eval runs list - table of runs with metrics."""
    runs_result = await db.execute(
        select(SearchEvalRun).order_by(SearchEvalRun.run_at.desc())
    )
    runs = runs_result.scalars().all()

    # Aggregate metrics per run (chunk/doc hit@1,2,4,8)
    run_metrics = {}
    if runs:
        run_ids = [str(r.id) for r in runs]
        agg = await db.execute(
            text("""
                SELECT run_id::text,
                    COUNT(*) AS n,
                    SUM(CASE WHEN chunk_rank <= 1 THEN 1 ELSE 0 END) AS chunk_h1,
                    SUM(CASE WHEN chunk_rank <= 2 THEN 1 ELSE 0 END) AS chunk_h2,
                    SUM(CASE WHEN chunk_rank <= 4 THEN 1 ELSE 0 END) AS chunk_h4,
                    SUM(CASE WHEN chunk_rank <= 8 THEN 1 ELSE 0 END) AS chunk_h8,
                    SUM(CASE WHEN doc_rank <= 1 THEN 1 ELSE 0 END) AS doc_h1,
                    SUM(CASE WHEN doc_rank <= 2 THEN 1 ELSE 0 END) AS doc_h2,
                    SUM(CASE WHEN doc_rank <= 4 THEN 1 ELSE 0 END) AS doc_h4,
                    SUM(CASE WHEN doc_rank <= 8 THEN 1 ELSE 0 END) AS doc_h8
                FROM search_eval_results
                WHERE run_id::text = ANY(:ids)
                GROUP BY run_id
            """),
            {"ids": run_ids},
        )
        for row in agg.fetchall():
            run_metrics[row[0]] = {
                "n": row[1],
                "chunk_h1": row[2], "chunk_h2": row[3],
                "chunk_h4": row[4], "chunk_h8": row[5],
                "doc_h1": row[6], "doc_h2": row[7],
                "doc_h4": row[8], "doc_h8": row[9],
            }

    return templates.TemplateResponse(
        request=request,
        name="eval_runs.html",
        context={"runs": runs, "run_metrics": run_metrics},
    )


@app.get("/search-eval-runs/{run_id}", response_class=HTMLResponse)
async def eval_run_detail_page(
    request: Request,
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Eval run detail - run info and all test cases as cards."""
    from uuid import UUID

    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    result = await db.execute(
        select(SearchEvalRun).where(SearchEvalRun.id == run_uuid)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Load results with question text and target doc/chunk for display
    results_rows = await db.execute(
        text("""
            SELECT r.id, r.question_id, r.query_type,
                   r.chunk_rank, r.doc_rank,
                   r.target_chunk_id, r.target_document_id,
                   r.chunk_rank_order, r.document_rank_order,
                   q.question
            FROM search_eval_results r
            JOIN search_test_questions q ON q.id = r.question_id
            WHERE r.run_id = :run_id
            ORDER BY r.chunk_rank NULLS LAST, r.doc_rank NULLS LAST
        """),
        {"run_id": run_id},
    )
    rows = results_rows.fetchall()

    # Build doc_id -> filename, chunk_id -> (content_preview, page, doc_id) for display
    doc_ids = set()
    chunk_ids = set()
    for row in rows:
        doc_ids.add(str(row[6]))  # target_document_id
        chunk_ids.add(str(row[5]))  # target_chunk_id
        for did in (row[8] or []):  # document_rank_order
            doc_ids.add(str(did))
        for cid in (row[7] or []):  # chunk_rank_order
            chunk_ids.add(str(cid))

    doc_map = {}
    chunk_map = {}
    if doc_ids:
        docs = await db.execute(
            text("SELECT id::text, filename FROM documents WHERE id::text = ANY(:ids)"),
            {"ids": list(doc_ids)},
        )
        doc_map = {r[0]: r[1] for r in docs.fetchall()}
    if chunk_ids:
        chunks = await db.execute(
            text("""
                SELECT c.id::text, c.content, c.page_number, c.document_id::text
                FROM chunks c
                WHERE c.id::text = ANY(:ids)
            """),
            {"ids": list(chunk_ids)},
        )
        for r in chunks.fetchall():
            chunk_map[r[0]] = {
                "content": (r[1] or "")[:400],
                "page": r[2],
                "document_id": r[3],
            }

    results = []
    for row in rows:
        chunk_rank_order = row[7] or []
        document_rank_order = row[8] or []
        target_doc_id = str(row[6])
        target_chunk_id = str(row[5])
        chunk_rank_items = []
        for i, cid in enumerate(chunk_rank_order):
            cid_str = str(cid)
            info = chunk_map.get(cid_str, {})
            doc_id = info.get("document_id")
            doc_name = doc_map.get(doc_id, "?") if doc_id else "?"
            chunk_rank_items.append({
                "rank": i + 1,
                "chunk_id": cid_str,
                "doc_name": doc_name,
                "is_target": cid_str == target_chunk_id,
            })
        doc_rank_items = []
        for i, did in enumerate(document_rank_order):
            did_str = str(did)
            doc_rank_items.append({
                "rank": i + 1,
                "doc_id": did_str,
                "doc_name": doc_map.get(did_str, "?"),
                "is_target": did_str == target_doc_id,
            })
        results.append({
            "id": row[0],
            "question_id": row[1],
            "query_type": row[2],
            "chunk_rank": row[3],
            "doc_rank": row[4],
            "target_chunk_id": target_chunk_id,
            "target_document_id": target_doc_id,
            "target_doc_name": doc_map.get(target_doc_id, "(deleted)"),
            "target_chunk_preview": chunk_map.get(target_chunk_id, {}).get("content", "(deleted)"),
            "target_chunk_page": chunk_map.get(target_chunk_id, {}).get("page"),
            "question": row[9],
            "chunk_rank_items": chunk_rank_items,
            "doc_rank_items": doc_rank_items,
        })

    # Compute summary metrics
    n = len(results)
    chunk_hits = {1: 0, 2: 0, 4: 0, 8: 0}
    doc_hits = {1: 0, 2: 0, 4: 0, 8: 0}
    for r in results:
        cr, dr = r["chunk_rank"], r["doc_rank"]
        for k in (1, 2, 4, 8):
            if cr is not None and cr <= k:
                chunk_hits[k] += 1
            if dr is not None and dr <= k:
                doc_hits[k] += 1

    metrics = {
        "n": n,
        "chunk_h1": chunk_hits[1], "chunk_h2": chunk_hits[2],
        "chunk_h4": chunk_hits[4], "chunk_h8": chunk_hits[8],
        "doc_h1": doc_hits[1], "doc_h2": doc_hits[2],
        "doc_h4": doc_hits[4], "doc_h8": doc_hits[8],
    }

    return templates.TemplateResponse(
        request=request,
        name="eval_run_detail.html",
        context={
            "run": run,
            "results": results,
            "metrics": metrics,
            "doc_map": doc_map,
            "chunk_map": chunk_map,
        },
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
