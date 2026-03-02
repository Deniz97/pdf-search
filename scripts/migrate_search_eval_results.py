"""Add columns to search_eval_results for on-demand metric computation.

Run after pulling changes that add query_type, target_chunk_id, target_document_id,
chunk_rank_order, document_rank_order. Safe to run multiple times (skips if exists).
"""
import sys
from pathlib import Path

# Add project root so `app` is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import create_engine, text

from app.config import settings


def column_exists(engine, table: str, column: str) -> bool:
    with engine.connect() as conn:
        r = conn.execute(
            text("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = :t AND column_name = :c
            """),
            {"t": table, "c": column},
        )
        return r.fetchone() is not None


def main() -> int:
    engine = create_engine(settings.database_url_sync)
    try:
        with engine.connect() as conn:
            if not column_exists(engine, "search_eval_results", "query_type"):
                print("Adding query_type...")
                conn.execute(text("ALTER TABLE search_eval_results ADD COLUMN query_type VARCHAR"))
                conn.commit()
            if not column_exists(engine, "search_eval_results", "target_chunk_id"):
                print("Adding target_chunk_id...")
                conn.execute(
                    text("ALTER TABLE search_eval_results ADD COLUMN target_chunk_id UUID")
                )
                conn.commit()
            if not column_exists(engine, "search_eval_results", "target_document_id"):
                print("Adding target_document_id...")
                conn.execute(
                    text("ALTER TABLE search_eval_results ADD COLUMN target_document_id UUID")
                )
                conn.commit()
            if not column_exists(engine, "search_eval_results", "chunk_rank_order"):
                print("Adding chunk_rank_order...")
                conn.execute(
                    text("ALTER TABLE search_eval_results ADD COLUMN chunk_rank_order JSONB")
                )
                conn.commit()
            if not column_exists(engine, "search_eval_results", "document_rank_order"):
                print("Adding document_rank_order...")
                conn.execute(
                    text("ALTER TABLE search_eval_results ADD COLUMN document_rank_order JSONB")
                )
                conn.commit()
        print("Migration complete.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
