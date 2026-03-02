"""CLI for batch enrich: enrich N|ALL, reenrich N|ALL."""

import sys

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings
from app.services.enricher import enrich_document


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


def cmd_enrich(count: int | None, reenrich: bool) -> None:
    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
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

        if reenrich:
            print(
                f"Re-enriching {len(rows)} document(s) "
                "(including already finished)...\n"
            )
        else:
            print(f"Enriching {len(rows)} document(s)...\n")

        total_memories = 0
        for row in rows:
            try:
                total_memories += enrich_document(session, str(row.id), row.filename)
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
    args = parser.parse_args()

    try:
        count = parse_count(args.count)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    cmd_enrich(count, reenrich=(args.command == "reenrich"))


if __name__ == "__main__":
    main()
