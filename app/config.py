from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5433/pdf_search"
    )
    database_url_sync: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5433/pdf_search"
    )
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chat_model: str = "gpt-4o-mini"
    chunk_sentences: int = 5
    chunk_overlap: int = 1
    upload_dir: str = "uploads"
    cache_dir: str = "cache"

    model_config = {"env_file": ".env"}


settings = Settings()
