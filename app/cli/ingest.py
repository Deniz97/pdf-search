"""CLI for batch ingest: ingest N|ALL, reingest N|ALL, prune."""

import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.cli.utils import setup_signal_handler
from app.database import sync_engine
from app.services.ingest import ingest_pdf


def collect_pdfs(root: Path) -> list[Path]:
    """Recursively find all PDFs under root and return them sorted by path."""
    pdfs = sorted(p.absolute() for p in root.rglob("*.pdf") if p.is_file())
    return pdfs


def get_db_paths(session: Session) -> set[str]:
    """Return set of all document paths currently in the database."""
    rows = session.execute(
        text("SELECT path FROM documents WHERE path IS NOT NULL")
    ).fetchall()
    return {str(r.path) for r in rows}


def get_db_path_to_id(session: Session) -> dict[str, str]:
    """Return mapping of path -> document id for docs with path set."""
    rows = session.execute(
        text("SELECT id, path FROM documents WHERE path IS NOT NULL")
    ).fetchall()
    return {str(r.path): str(r.id) for r in rows}


def get_db_path_to_created_at(session: Session) -> dict[str, datetime]:
    """Return mapping of path -> created_at for docs with path set (for reingest ordering)."""
    rows = session.execute(
        text("SELECT path, created_at FROM documents WHERE path IS NOT NULL")
    ).fetchall()
    return {str(r.path): r.created_at for r in rows}


def parse_count(s: str) -> int | None:
    """Parse N or ALL. Returns int for N, None for ALL."""
    if s.upper() == "ALL":
        return None
    try:
        n = int(s)
        if n < 1:
            raise ValueError("N must be >= 1")
        return n
    except ValueError:
        raise ValueError(f"Count must be a positive integer or 'ALL', got: {s!r}")


def cmd_ingest(
    root: Path,
    count: int | None,
    reingest: bool,
) -> None:
    """Ingest N or ALL PDFs. Reingest forces re-processing."""
    shutdown_requested = setup_signal_handler()

    pdfs = collect_pdfs(root)
    if not pdfs:
        print(f"No PDF files found under '{root}'")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) under '{root}':\n")
    for p in pdfs:
        print(f"  - {p}")
    print()

    if reingest:
        print(
            "Reingest mode: will re-process all selected PDFs, including already finished.\n"
        )

    with Session(sync_engine) as session:
        path_to_id = get_db_path_to_id(session) if not reingest else {}
        path_to_created_at = get_db_path_to_created_at(session) if reingest else {}

        # Build list to process
        to_process: list[Path] = []
        for p in pdfs:
            path_str = str(p.resolve())
            if reingest:
                to_process.append(p)
            elif path_str not in path_to_id:
                to_process.append(p)
            elif (
                session.execute(
                    text("SELECT processed_status FROM documents WHERE id = :id"),
                    {"id": path_to_id[path_str]},
                ).scalar_one()
                != "finished"
            ):
                to_process.append(p)

        # Reingest: sort by oldest ingested first (created_at ASC); new docs at end
        if reingest and path_to_created_at:
            to_process.sort(
                key=lambda p: path_to_created_at.get(str(p.resolve()), datetime.max)
            )

        if count is not None:
            to_process = to_process[:count]

        if not to_process:
            print("Nothing to process.")
            return

        print(f"Processing {len(to_process)} document(s)...\n")
        total_chunks = 0
        for pdf_path in to_process:
            if shutdown_requested[0]:
                break
            try:
                total_chunks += ingest_pdf(session, pdf_path, reingest=reingest)
            except Exception as e:
                print(f"  ERROR processing '{pdf_path}': {e}")

        print(f"\nDone. Total chunks created/updated: {total_chunks}")


def cmd_prune() -> None:
    """Remove DB records for documents whose files no longer exist on disk."""
    with Session(sync_engine) as session:
        path_to_id = get_db_path_to_id(session)
        deleted_count = 0
        for path_str, doc_id in list(path_to_id.items()):
            if not Path(path_str).exists():
                session.execute(
                    text("DELETE FROM documents WHERE id = :id"),
                    {"id": doc_id},
                )
                deleted_count += 1
                print(f"  Removed: {path_str}")

        if deleted_count:
            session.commit()
            print(f"\nPruned {deleted_count} document(s) with missing files.")
        else:
            print("Nothing to prune.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the vector database (recursive discovery)",
        epilog="Examples:\n"
        "  %(prog)s ingest ALL\n"
        "  %(prog)s --path /data/pdfs ingest 10\n"
        "  %(prog)s reingest ALL\n"
        "  %(prog)s prune\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=Path("example-pdfs"),
        metavar="DIR",
        help="Root directory to recursively search for PDFs (default: example-pdfs)",
    )
    parser.add_argument(
        "command",
        choices=["ingest", "reingest", "prune"],
        help="Command to run",
    )
    parser.add_argument(
        "count",
        nargs="?",
        default="ALL",
        metavar="N|ALL",
        help="Number of documents to process, or ALL (ignored for prune)",
    )
    args = parser.parse_args()

    if args.command == "prune":
        cmd_prune()
        return

    root = args.path.resolve()
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory")
        sys.exit(1)

    try:
        count = parse_count(args.count)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.command == "ingest":
        cmd_ingest(root, count, reingest=False)
    else:  # reingest
        cmd_ingest(root, count, reingest=True)


if __name__ == "__main__":
    main()
