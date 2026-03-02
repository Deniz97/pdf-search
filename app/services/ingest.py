"""Single-document ingest service. Used by API (upload/reprocess) and CLI batch runner."""

import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm

from app.config import settings
from app.services.embeddings import get_embeddings
from app.services.pdf_processing import extract_chunks_from_pdf


def _update_status(
    session: Session,
    doc_id: str,
    status: str,
    error: str | None = None,
) -> None:
    session.execute(
        text(
            "UPDATE documents SET processed_status = :status, "
            "last_processed_at = :now, last_process_error = :error "
            "WHERE id = :id"
        ),
        {"status": status, "now": datetime.utcnow(), "error": error, "id": doc_id},
    )
    session.commit()


def ingest_pdf(
    session: Session,
    pdf_path: Path,
    reingest: bool = False,
    path_str: str | None = None,
) -> int:
    """Process a single PDF and store its chunks. Returns chunk count.

    Uses path as the unique key. path_str is the canonical string (usually resolved).
    """
    canonical_path = path_str or str(pdf_path.resolve())

    existing = session.execute(
        text("SELECT id, processed_status FROM documents WHERE path = :path"),
        {"path": canonical_path},
    ).fetchone()

    if existing and existing.processed_status == "finished" and not reingest:
        print(f"  Already ingested '{pdf_path}', skipping.")
        return 0

    if existing:
        doc_id = str(existing.id)
        session.execute(
            text("DELETE FROM chunks WHERE document_id = :id"), {"id": doc_id}
        )
        session.commit()
    else:
        doc_id = str(
            session.execute(
                text(
                    "INSERT INTO documents (id, filename, path, page_count, processed_status) "
                    "VALUES (gen_random_uuid(), :filename, :path, 0, 'waiting') RETURNING id"
                ),
                {"filename": pdf_path.name, "path": canonical_path},
            ).scalar_one()
        )
        session.commit()

    _update_status(session, doc_id, "started")

    cache_dir = Path(settings.cache_dir) / doc_id
    pdf_cache_path = cache_dir / "source.pdf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not pdf_cache_path.exists():
        shutil.copy2(pdf_path, pdf_cache_path)

    try:
        print(f"  Processing '{pdf_path}' (id: {doc_id})...")
        chunks = extract_chunks_from_pdf(str(pdf_cache_path), doc_id=doc_id)
        print(f"  Extracted {len(chunks)} chunks, starting embedding...")
        if not chunks:
            _update_status(session, doc_id, "errored", "No chunks produced")
            print(f"  WARNING: No chunks produced from '{pdf_path}'.")
            return 0

        page_count = max(c["page_number"] for c in chunks) if chunks else 0
        session.execute(
            text("UPDATE documents SET page_count = :pc WHERE id = :id"),
            {"pc": page_count, "id": doc_id},
        )
        session.commit()

        texts = [c["content"] for c in chunks]
        batch_size = 100
        print(f"  Embedding {len(chunks)} chunks in batches of {batch_size}...")
        with tqdm(
            total=len(chunks), desc=f"  Embedding {pdf_path.name}", unit="chunk"
        ) as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_chunks = chunks[i : i + batch_size]
                embeddings = get_embeddings(batch_texts)

                for chunk_data, embedding in zip(batch_chunks, embeddings):
                    session.execute(
                        text(
                            "INSERT INTO chunks (id, document_id, content, page_number, chunk_index, chunk_type, bbox, embedding) "
                            "VALUES (gen_random_uuid(), :doc_id, :content, :page_number, :chunk_index, :chunk_type, :bbox, :embedding)"
                        ),
                        {
                            "doc_id": doc_id,
                            "content": chunk_data["content"],
                            "page_number": chunk_data["page_number"],
                            "chunk_index": chunk_data["chunk_index"],
                            "chunk_type": chunk_data.get("chunk_type"),
                            "bbox": chunk_data.get("bbox"),
                            "embedding": str(embedding),
                        },
                    )
                pbar.update(len(batch_chunks))

        print(f"  Committing chunks to DB...")
        session.commit()
        _update_status(session, doc_id, "finished")
        print(f"  Ingested '{pdf_path}': {page_count} pages, {len(chunks)} chunks")
        return len(chunks)

    except Exception as e:
        session.rollback()
        _update_status(session, doc_id, "errored", str(e))
        raise
