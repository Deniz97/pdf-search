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
