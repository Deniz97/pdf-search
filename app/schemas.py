from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    page_count: int
    processed_status: str
    last_processed_at: datetime | None
    last_process_error: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChunkResult(BaseModel):
    content: str
    page_number: int | None
    chunk_index: int
    chunk_type: str | None = None  # text, title, table
    score: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of results to return"
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[ChunkResult]


class UploadResponse(BaseModel):
    document: DocumentResponse
    chunks_created: int
    message: str


class SearchOnlyRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(
        default=10, ge=1, le=50, description="Number of results to return"
    )


class ChunkResultWithDocument(ChunkResult):
    document_id: UUID
    document_filename: str


class SearchOnlyResponse(BaseModel):
    query: str
    results: list[ChunkResultWithDocument]


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    message: ChatMessage
    tool_calls: list[dict] | None = None


# ----- Enhanced search response types -----


class CueResult(BaseModel):
    """A generated embedding cue with its relevance score."""

    text: str
    score: float


class RegexResult(BaseModel):
    """A generated regex pattern with its relevance score."""

    text: str  # The regex pattern
    score: float


class ContextDto(BaseModel):
    """A context memory that matched the search."""

    content: str
    chunk_id: UUID


class FactDto(BaseModel):
    """A fact memory that matched the search."""

    content: str
    chunk_id: UUID


class DocumentDto(BaseModel):
    """Document metadata for search results."""

    id: UUID
    filename: str
    page_count: int
    processed_status: str
    last_processed_at: datetime | None
    last_process_error: str | None
    created_at: datetime
    title: str | None = None
    description: str | None = None
    tags: list[str] | None = None


class ChunkMatchDto(BaseModel):
    """A matched chunk with ID and display fields."""

    chunk_id: UUID
    content: str
    page_number: int | None
    chunk_index: int
    chunk_type: str | None = None  # text, title, table
    score: float


class DocumentResult(BaseModel):
    """Search result for a single document with all match details."""

    document: DocumentDto
    matched_chunks: list[ChunkMatchDto]  # chunk_id + content, page, score for display
    matched_contexts: list[ContextDto]
    matched_facts: list[FactDto]
    description_matched: bool
    matched_tags: list[str]


class EnhancedSearchResponse(BaseModel):
    """Full enhanced search response with generated signals and document results."""

    user_query: str
    generated_cues: list[CueResult]
    generated_regexes: list[RegexResult]
    results: list[DocumentResult]
    chunk_rank_order: list[UUID] = []  # chunk IDs in retrieval order (for eval)
    document_rank_order: list[
        UUID
    ] = []  # doc IDs in order of first occurrence (for eval)
