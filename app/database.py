from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
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

    async with engine.begin() as conn:
        await conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await conn.run_sync(Base.metadata.create_all)
        # Migration: add path column if missing (for existing DBs)
        await conn.execute(
            __import__("sqlalchemy").text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS path VARCHAR UNIQUE"
            )
        )
        # Migration: add created_at if missing, set default, backfill NULLs
        await conn.execute(
            __import__("sqlalchemy").text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT now()"
            )
        )
        await conn.execute(
            __import__("sqlalchemy").text(
                "ALTER TABLE documents ALTER COLUMN created_at SET DEFAULT now()"
            )
        )
        await conn.execute(
            __import__("sqlalchemy").text(
                "UPDATE documents SET created_at = now() WHERE created_at IS NULL"
            )
        )
        await conn.execute(
            __import__("sqlalchemy").text(
                "ALTER TABLE documents ALTER COLUMN created_at SET NOT NULL"
            )
        )
        # Migration: add chunk_type and bbox for PP-Structure blocks
        await conn.execute(
            __import__("sqlalchemy").text(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_type VARCHAR"
            )
        )
        await conn.execute(
            __import__("sqlalchemy").text(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS bbox JSONB"
            )
        )
