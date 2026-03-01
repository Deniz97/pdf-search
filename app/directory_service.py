"""Directory management and PDF scanning/ingestion logic."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.config import settings
from app.embeddings import get_embeddings
from app.models import Directory, Document
from app.pdf_processing import chunk_text, extract_text_from_pdf


def scan_directory_for_pdfs(directory_path: Path) -> list[tuple[Path, str]]:
    """Recursively scan directory for PDF files.
    Returns list of (absolute_path, relative_path) tuples.
    """
    pdfs = []
    if not directory_path.exists() or not directory_path.is_dir():
        return pdfs
    
    for pdf_path in directory_path.rglob("*.pdf"):
        if pdf_path.is_file():
            try:
                relative_path = pdf_path.relative_to(directory_path)
                pdfs.append((pdf_path, str(relative_path)))
            except ValueError:
                # Path not relative to directory (shouldn't happen)
                continue
    
    return sorted(pdfs, key=lambda x: x[1])


async def sync_directory(
    db: AsyncSession, directory_id: str, directory_path: Path
) -> tuple[int, int, int]:
    """Scan directory for PDFs and register them in database.
    Returns (total_found, new_count, existing_count).
    """
    pdfs = scan_directory_for_pdfs(directory_path)
    total_found = len(pdfs)
    new_count = 0
    existing_count = 0

    # Update directory status
    await db.execute(
        text("UPDATE directories SET status = 'syncing', last_synced_at = :now WHERE id = :id"),
        {"now": datetime.utcnow(), "id": directory_id},
    )
    await db.commit()

    for pdf_path, relative_path in pdfs:
        # Check if document already exists (by filename and directory)
        existing = await db.execute(
            text(
                "SELECT id FROM documents WHERE filename = :filename AND directory_id = :dir_id"
            ),
            {"filename": pdf_path.name, "dir_id": directory_id},
        )
        existing_doc = existing.fetchone()

        if not existing_doc:
            # Create new document record
            await db.execute(
                text(
                    "INSERT INTO documents (id, filename, file_path, directory_id, page_count, processed_status) "
                    "VALUES (gen_random_uuid(), :filename, :file_path, :dir_id, 0, 'waiting')"
                ),
                {
                    "filename": pdf_path.name,
                    "file_path": relative_path,
                    "dir_id": directory_id,
                },
            )
            new_count += 1
        else:
            existing_count += 1

    await db.commit()

    # Update directory status back to active
    await db.execute(
        text("UPDATE directories SET status = 'active' WHERE id = :id"),
        {"id": directory_id},
    )
    await db.commit()

    return total_found, new_count, existing_count


def ingest_pdf_from_directory(
    session: Session,
    pdf_path: Path,
    relative_path: str,
    directory_id: str,
    doc_id: Optional[str] = None,
) -> int:
    """Process a single PDF from a directory and store its chunks. Returns chunk count."""
    
    # Find or create document record
    if doc_id:
        existing = session.execute(
            text("SELECT id, processed_status FROM documents WHERE id = :id"),
            {"id": doc_id},
        ).fetchone()
    else:
        existing = session.execute(
            text(
                "SELECT id, processed_status FROM documents WHERE filename = :name AND directory_id = :dir_id"
            ),
            {"name": pdf_path.name, "dir_id": directory_id},
        ).fetchone()

    if existing and existing.processed_status == "finished":
        return 0

    if existing:
        doc_id = str(existing.id)
        session.execute(text("DELETE FROM chunks WHERE document_id = :id"), {"id": doc_id})
        session.commit()
    else:
        doc_id = str(
            session.execute(
                text(
                    "INSERT INTO documents (id, filename, file_path, directory_id, page_count, processed_status) "
                    "VALUES (gen_random_uuid(), :filename, :file_path, :dir_id, 0, 'waiting') RETURNING id"
                ),
                {
                    "filename": pdf_path.name,
                    "file_path": relative_path,
                    "dir_id": directory_id,
                },
            ).scalar()
        )
        session.commit()

    _update_status(session, doc_id, "started")

    cache_dir = Path(settings.cache_dir) / doc_id
    pdf_cache_path = cache_dir / "source.pdf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not pdf_cache_path.exists():
        shutil.copy2(pdf_path, pdf_cache_path)

    try:
        pages = extract_text_from_pdf(str(pdf_cache_path), doc_id=doc_id)
        if not pages:
            _update_status(session, doc_id, "errored", "No text extracted")
            return 0

        session.execute(
            text("UPDATE documents SET page_count = :pc WHERE id = :id"),
            {"pc": len(pages), "id": doc_id},
        )
        session.commit()

        chunks = chunk_text(pages)
        if not chunks:
            _update_status(session, doc_id, "errored", "No chunks produced")
            return 0

        texts = [c["content"] for c in chunks]

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_chunks = chunks[i : i + batch_size]
            embeddings = get_embeddings(batch_texts)

            for chunk_data, embedding in zip(batch_chunks, embeddings):
                session.execute(
                    text(
                        "INSERT INTO chunks (id, document_id, content, page_number, chunk_index, embedding) "
                        "VALUES (gen_random_uuid(), :doc_id, :content, :page_number, :chunk_index, :embedding)"
                    ),
                    {
                        "doc_id": doc_id,
                        "content": chunk_data["content"],
                        "page_number": chunk_data["page_number"],
                        "chunk_index": chunk_data["chunk_index"],
                        "embedding": str(embedding),
                    },
                )

        session.commit()
        _update_status(session, doc_id, "finished")
        return len(chunks)

    except Exception as e:
        session.rollback()
        _update_status(session, doc_id, "errored", str(e))
        raise


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
