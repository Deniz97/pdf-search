import atexit
import asyncio
import logging
from collections.abc import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Async engine (FastAPI, run_search_eval)
# -----------------------------------------------------------------------------
# Timeout connection after 10s to avoid hanging when DB is unreachable.
# pool_pre_ping: avoid "connection was closed in the middle of operation" when
#   running parallel scripts or after idle connections are dropped.
engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={"timeout": 10},
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# -----------------------------------------------------------------------------
# Sync engine (CLI scripts, reprocess endpoint, migrate/unlock scripts)
# -----------------------------------------------------------------------------
# Shared by all sync DB usage. pool_pre_ping + pool_recycle for robustness.
sync_engine = create_engine(
    settings.database_url_sync,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=10,
    pool_recycle=300,
)

# -----------------------------------------------------------------------------
# Disposal: run on process exit (scripts) or lifespan shutdown (web app)
# -----------------------------------------------------------------------------
_disposed = False


def dispose_all() -> None:
    """Dispose both engines. Safe to call multiple times."""
    global _disposed
    if _disposed:
        return
    _disposed = True
    sync_engine.dispose()
    try:
        asyncio.run(engine.dispose())
    except Exception:
        pass  # e.g. event loop already closed, connections from different loop


atexit.register(dispose_all)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with async_session() as session:
        yield session


async def reset_db():
    """Drop all tables to reset the database."""
    import app.models  # noqa: F401 — ensure models are registered with Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def init_db():
    import app.models  # noqa: F401 — ensure models are registered with Base.metadata

    log.info("init_db: connecting...")
    async with engine.begin() as conn:
        log.info("init_db: connected, setting timeouts")
        # Fail fast: 30s statement timeout, 10s lock timeout
        await conn.execute(
            __import__("sqlalchemy").text("SET statement_timeout = '30000'")
        )
        await conn.execute(__import__("sqlalchemy").text("SET lock_timeout = '10000'"))
        log.info("init_db: creating vector extension")
        await conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        log.info("init_db: create_all tables")
        await conn.run_sync(Base.metadata.create_all)
    log.info("init_db: done")
