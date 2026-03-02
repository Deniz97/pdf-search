import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

log = logging.getLogger(__name__)

# Timeout connection after 10s to avoid hanging when DB is unreachable
engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={"timeout": 10},
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


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
        await conn.execute(
            __import__("sqlalchemy").text("SET lock_timeout = '10000'")
        )
        log.info("init_db: creating vector extension")
        await conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        log.info("init_db: create_all tables")
        await conn.run_sync(Base.metadata.create_all)
    log.info("init_db: done")
