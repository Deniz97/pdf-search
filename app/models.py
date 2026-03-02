import uuid

from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.config import settings
from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    path = Column(
        String, nullable=True, unique=True
    )  # Resolved filesystem path; unique per doc
    page_count = Column(Integer, nullable=False, default=0)
    processed_status = Column(String, nullable=False, default="waiting")
    last_processed_at = Column(DateTime, nullable=True)
    last_process_error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    enrichment = relationship(
        "DocumentEnrichment", back_populates="document", uselist=False
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String, nullable=True)  # text, title, table, figure
    bbox = Column(JSON, nullable=True)  # [x1, y1, x2, y2] in page coords
    embedding = Column(Vector(settings.embedding_dimensions))


class DocumentEnrichment(Base):
    __tablename__ = "document_enrichments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    description_embedding = Column(Vector(settings.embedding_dimensions))
    tags_embedding = Column(Vector(settings.embedding_dimensions))
    enrichment_status = Column(String, nullable=False, default="pending")
    last_enriched_at = Column(DateTime, nullable=True)
    last_enrich_error = Column(Text, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="enrichment")


class Memory(Base):
    __tablename__ = "memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dimensions))


class SearchTestQuestion(Base):
    """A test question with known target chunk and document for search evaluation."""

    __tablename__ = "search_test_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    query_type = Column(
        String, nullable=False
    )  # e.g. direct, paraphrase, keyword, conceptual
    target_chunk_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_memory_id = Column(
        UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    target_chunk = relationship("Chunk")
    target_document = relationship("Document")


class SearchEvalRun(Base):
    """A single run of search evaluation."""

    __tablename__ = "search_eval_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_at = Column(DateTime, server_default=func.now(), nullable=False)
    top_k = Column(Integer, nullable=False, default=10)
    notes = Column(Text, nullable=True)


class DownloadedPdf(Base):
    """Tracks PDFs downloaded by scripts/download_pdfs.py to avoid re-downloading."""

    __tablename__ = "downloaded_pdfs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(Text, nullable=False, unique=True)
    search_query = Column(Text, nullable=False)
    local_path = Column(String, nullable=False)
    downloaded_at = Column(DateTime, server_default=func.now(), nullable=False)


class SearchEvalResult(Base):
    """Per-question result from a search eval run.

    Stores all data needed to compute metrics on demand (no JOINs required).
    For regeneration: DELETE FROM search_eval_results WHERE run_id = :id
    """

    __tablename__ = "search_eval_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("search_eval_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id = Column(
        UUID(as_uuid=True),
        ForeignKey("search_test_questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_rank = Column(Integer, nullable=True)  # 1-indexed, None if not in top_k
    doc_rank = Column(Integer, nullable=True)  # 1-indexed, None if not in top_k
    # Denormalized for on-demand metric computation
    query_type = Column(String, nullable=False)
    target_chunk_id = Column(UUID(as_uuid=True), nullable=False)
    target_document_id = Column(UUID(as_uuid=True), nullable=False)
    chunk_rank_order = Column(JSON, nullable=True)  # [chunk_id, ...] as strings
    document_rank_order = Column(JSON, nullable=True)  # [doc_id, ...] as strings
