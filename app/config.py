from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5433/pdf_search"
    )
    database_url_sync: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5433/pdf_search"
    )
    openai_api_key: str = ""
    cohere_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chat_model: str = "gpt-4o-mini"
    enrichment_model: str = "gpt-3.5-turbo"  # faster/cheaper for STM + metadata (iterates over all chunks)
    rerank_model: str = "rerank-v3.5"
    rerank_top_n_multiplier: int = 4
    chunk_min_words: int = 100
    chunk_min_sentences: int = 10
    overlap_min_words: int = 10
    overlap_min_sentences: int = 1
    upload_dir: str = "uploads"
    cache_dir: str = "cache"
    pdf_dpi: int = 250
    ocr_backend: str = "tesseract"  # "tesseract" (fast, default) or "ppstructure" (layout+table, slow)
    stm_max_size: int = 15
    ltm_sample_max_chars: int = 8000

    model_config = {"env_file": ".env"}


settings = Settings()
