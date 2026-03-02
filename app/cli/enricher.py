"""CLI for batch enrich: enrich N|ALL, reenrich N|ALL."""

import sys
from concurrent.futures import as_completed

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.cli.utils import managed_thread_pool, setup_signal_handler
from app.database import sync_engine
from app.services.enricher import enrich_document

DEFAULT_WORKERS = 4


def _enrich_one(doc_id: str, filename: str) -> tuple[int, str | None]:
    """Enrich a single document with its own DB session. Returns (memories_count, error)."""
    try:
        with Session(sync_engine) as session:
            return (enrich_document(session, doc_id, filename), None)
    except Exception as e:
        return (0, str(e))


def parse_count(s: str) -> int | None:
    if s.upper() == "ALL":
        return None
    try:
        n = int(s)
        if n < 1:
            raise ValueError("N must be >= 1")
        return n
    except ValueError:
        raise ValueError(f"Count must be a positive integer or 'ALL', got: {s!r}")


def cmd_enrich(
    count: int | None, reenrich: bool, workers: int = DEFAULT_WORKERS
) -> None:
    shutdown_requested = setup_signal_handler(
        "\nInterrupt received, finishing current work and shutting down..."
    )

    with Session(sync_engine) as session:
        if reenrich:
            rows = session.execute(
                text(
                    "SELECT id, filename FROM documents "
                    "WHERE processed_status = 'finished' "
                    "ORDER BY filename"
                )
            ).fetchall()
        else:
            rows = session.execute(
                text(
                    "SELECT d.id, d.filename FROM documents d "
                    "LEFT JOIN document_enrichments de ON de.document_id = d.id "
                    "WHERE d.processed_status = 'finished' "
                    "AND (de.id IS NULL OR de.enrichment_status != 'finished') "
                    "ORDER BY d.filename"
                )
            ).fetchall()

        if count is not None:
            rows = rows[:count]

        if not rows:
            print("Nothing to enrich.")
            return

        if shutdown_requested[0]:
            return

        if reenrich:
            print(
                f"Re-enriching {len(rows)} document(s) "
                f"(including already finished) with {workers} workers...\n"
            )
        else:
            print(f"Enriching {len(rows)} document(s) with {workers} workers...\n")

    total_memories = 0
    if workers <= 1:
        with Session(sync_engine) as session:
            for row in rows:
                if shutdown_requested[0]:
                    break
                try:
                    total_memories += enrich_document(
                        session, str(row.id), row.filename
                    )
                except Exception as e:
                    print(f"  ERROR enriching '{row.filename}': {e}")
    else:
        with managed_thread_pool(workers) as executor:
            futures = {
                executor.submit(_enrich_one, str(row.id), row.filename): row
                for row in rows
            }
            for future in as_completed(futures):
                if shutdown_requested[0]:
                    break
                row = futures[future]
                try:
                    memories, err = future.result()
                    total_memories += memories
                    if err:
                        print(f"  ERROR enriching '{row.filename}': {err}")
                except Exception as e:
                    print(f"  ERROR enriching '{row.filename}': {e}")

    print(f"\nDone. Total memories created: {total_memories}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich ingested documents with LLM-generated metadata and memories",
        epilog="Examples:\n"
        "  %(prog)s enrich ALL\n"
        "  %(prog)s enrich 5\n"
        "  %(prog)s reenrich ALL\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["enrich", "reenrich"],
        help="Command to run",
    )
    parser.add_argument(
        "count",
        metavar="N|ALL",
        help="Number of documents to process, or ALL",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel workers for document-level processing (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    try:
        count = parse_count(args.count)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    cmd_enrich(
        count,
        reenrich=(args.command == "reenrich"),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
