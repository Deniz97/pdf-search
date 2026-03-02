"""Terminate all other PostgreSQL backends connected to the app database.

Use when migrations hang (lock timeout) or the documents/chunks tables are
inaccessible. Closes DB viewer connections too - reconnect after.
  make unlock-documents
  make migrate
"""

import sys
from pathlib import Path

# Add project root so `app` is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import sync_engine


def main() -> int:
    with Session(sync_engine) as session:
        # Find ALL other backends connected to this database
        result = session.execute(
            text("""
                SELECT pid, state, query_start, left(query, 80) as query_preview
                FROM pg_stat_activity
                WHERE datname = current_database()
                  AND pid != pg_backend_pid()
                  AND usename IS NOT NULL
            """)
        )
        rows = result.fetchall()
        session.commit()

    if not rows:
        print("No other connections to this database.")
        return 0

    print(f"Terminating {len(rows)} other connection(s)...")
    for pid, state, query_start, query_preview in rows:
        print(f"  PID {pid}: {state} - {query_preview or '(idle)'}...")

    with Session(sync_engine) as session:
        for pid, _, _, _ in rows:
            try:
                session.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
                session.commit()
                print(f"  Terminated {pid}")
            except Exception as e:
                print(f"  Failed to terminate {pid}: {e}", file=sys.stderr)
                session.rollback()

    print("Done. Run 'make migrate' now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
