"""Ingest all PDFs from a directory into the vector database."""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tqdm import tqdm

from app.config import settings
from app.embeddings import get_embeddings
from app.pdf_processing import chunk_text, extract_text_from_pdf


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


def ingest_pdf(session: Session, pdf_path: Path, reingest: bool = False) -> int:
    """Process a single PDF and store its chunks. Returns chunk count.

    Args:
        session: Database session
        pdf_path: Path to PDF file
        reingest: If True, re-process even if already finished
    """

    existing = session.execute(
        text("SELECT id, processed_status FROM documents WHERE filename = :name"),
        {"name": pdf_path.name},
    ).fetchone()

    if existing and existing.processed_status == "finished" and not reingest:
        print(f"  Already ingested '{pdf_path.name}', skipping.")
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
                    "INSERT INTO documents (id, filename, page_count, processed_status) "
                    "VALUES (gen_random_uuid(), :filename, 0, 'waiting') RETURNING id"
                ),
                {"filename": pdf_path.name},
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
        print(f"  Processing '{pdf_path.name}' (id: {doc_id})...")
        pages = extract_text_from_pdf(str(pdf_cache_path), doc_id=doc_id)
        if not pages:
            _update_status(session, doc_id, "errored", "No text extracted")
            print(f"  WARNING: No text extracted from '{pdf_path.name}'.")
            return 0

        session.execute(
            text("UPDATE documents SET page_count = :pc WHERE id = :id"),
            {"pc": len(pages), "id": doc_id},
        )
        session.commit()

        chunks = chunk_text(pages)
        if not chunks:
            _update_status(session, doc_id, "errored", "No chunks produced")
            print(f"  WARNING: No chunks produced from '{pdf_path.name}'.")
            return 0

        texts = [c["content"] for c in chunks]

        batch_size = 100
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
                pbar.update(len(batch_chunks))

        session.commit()
        _update_status(session, doc_id, "finished")
        print(f"  Ingested '{pdf_path.name}': {len(pages)} pages, {len(chunks)} chunks")
        return len(chunks)

    except Exception as e:
        session.rollback()
        _update_status(session, doc_id, "errored", str(e))
        raise


def main(directory: str = "example-pdfs", reingest: bool = False):
    pdf_dir = Path(directory)
    if not pdf_dir.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in '{directory}'")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) in '{directory}':\n")
    for p in pdfs:
        print(f"  - {p.name}")
    print()

    if reingest:
        print(
            "Reingest mode: will re-process all PDFs, including already finished ones.\n"
        )

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        total_chunks = 0
        for pdf_path in pdfs:
            try:
                total_chunks += ingest_pdf(session, pdf_path, reingest=reingest)
            except Exception as e:
                print(f"  ERROR processing '{pdf_path.name}': {e}")

    print(f"\nDone. Total chunks created: {total_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector database")
    parser.add_argument(
        "directory",
        nargs="?",
        default="example-pdfs",
        help="Directory containing PDF files to ingest (default: example-pdfs)",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-process all PDFs, including those already finished",
    )
    args = parser.parse_args()
    main(directory=args.directory, reingest=args.reingest)
