"""Enhanced search with multi-signal retrieval boosted by enrichment data.

Expands the user query via LLM into 8 signal types (4 embedding cues +
4 regex patterns), each with a relevance score. Signals match against
memories (chunk-level) and document_enrichments (document-level), producing
boosts that are added to the base cosine similarity before Cohere reranking.
"""

import json
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from openai import OpenAI


@dataclass
class SearchResult:
    """A single search result from enhanced_search_in_document (document-scoped search)."""

    chunk_id: uuid.UUID
    content: str
    page_number: int | None
    chunk_index: int
    chunk_type: str | None
    document_id: uuid.UUID
    document_filename: str
    score: float


from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models import Document, Memory
from app.services.embeddings import get_embedding, get_embeddings, rerank

_client = OpenAI(api_key=settings.openai_api_key)

MEMORY_CUE_WEIGHT = 0.15
DOC_CUE_WEIGHT = 0.10
REGEX_WEIGHT = 0.10

QUERY_EXPANSION_SYSTEM_PROMPT = """\
You are a search query expansion assistant. Given a user search query, \
generate signals to find relevant documents in a knowledge base.

Generate 8 types of signals. For each item, assign a relevance score \
(0.0 to 1.0) indicating how important this signal is for answering the query.

Signal types:
1. context_cues — Short phrases about background, setting, or topic \
(matched via embedding against extracted contexts from document chunks)
2. fact_cues — Short phrases about specific facts, data, or claims \
(matched via embedding against extracted facts from document chunks)
3. description_cues — Short phrases that might match a document's \
generated description (matched via embedding)
4. tag_cues — Single words or short phrases matching document tags \
(matched via embedding)
5. context_regex — Case-insensitive regex patterns to match context text
6. fact_regex — Case-insensitive regex patterns to match factual text
7. description_regex — Case-insensitive regex patterns for document descriptions
8. tag_regex — Case-insensitive regex patterns for document tags

Keep regexes simple and valid. Generate 1-3 items per cue type and 0-2 \
per regex type. Omit a type with an empty list if not applicable.

Respond with ONLY valid JSON (no markdown fences):
{
  "context_cues": [{"text": "...", "relevance": 0.9}],
  "fact_cues": [{"text": "...", "relevance": 0.8}],
  "description_cues": [{"text": "...", "relevance": 0.7}],
  "tag_cues": [{"text": "...", "relevance": 0.6}],
  "context_regex": [{"pattern": "...", "relevance": 0.5}],
  "fact_regex": [{"pattern": "...", "relevance": 0.4}],
  "description_regex": [{"pattern": "...", "relevance": 0.3}],
  "tag_regex": [{"pattern": "...", "relevance": 0.2}]
}"""


async def _has_enrichment_data(db: AsyncSession) -> bool:
    result = await db.execute(
        sa_text(
            "SELECT EXISTS("
            "  SELECT 1 FROM document_enrichments"
            "  WHERE enrichment_status = 'finished'"
            ")"
        )
    )
    return bool(result.scalar_one())


def _expand_query(query: str) -> dict:
    response = _client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": QUERY_EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Search query: {query}"},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def _valid_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


# ---------- embedding cue helpers ----------


async def _memory_cue_boosts(
    db: AsyncSession,
    embedding: list[float],
    memory_type: str,
    relevance: float,
    chunk_boosts: defaultdict[str, float],
    limit: int = 20,
) -> None:
    result = await db.execute(
        sa_text(
            "SELECT chunk_id, 1 - (embedding <=> :emb) AS similarity "
            "FROM memories WHERE type = :type "
            "ORDER BY embedding <=> :emb LIMIT :lim"
        ),
        {"emb": str(embedding), "type": memory_type, "lim": limit},
    )
    for row in result.fetchall():
        chunk_boosts[str(row.chunk_id)] += (
            max(row.similarity, 0) * relevance * MEMORY_CUE_WEIGHT
        )


async def _doc_cue_boosts(
    db: AsyncSession,
    embedding: list[float],
    column: str,
    relevance: float,
    doc_boosts: defaultdict[str, float],
    doc_desc_matched: set[str] | None = None,
    limit: int = 10,
) -> None:
    assert column in ("description_embedding", "tags_embedding")
    result = await db.execute(
        sa_text(
            f"SELECT document_id, 1 - ({column} <=> :emb) AS similarity "
            f"FROM document_enrichments "
            f"WHERE enrichment_status = 'finished' "
            f"ORDER BY {column} <=> :emb LIMIT :lim"
        ),
        {"emb": str(embedding), "lim": limit},
    )
    for row in result.fetchall():
        doc_id = str(row.document_id)
        doc_boosts[doc_id] += max(row.similarity, 0) * relevance * DOC_CUE_WEIGHT
        if doc_desc_matched is not None and column == "description_embedding":
            doc_desc_matched.add(doc_id)


# ---------- regex helpers ----------


async def _memory_regex_boosts(
    db: AsyncSession,
    pattern: str,
    memory_type: str,
    relevance: float,
    chunk_boosts: defaultdict[str, float],
) -> None:
    try:
        result = await db.execute(
            sa_text(
                "SELECT DISTINCT chunk_id FROM memories "
                "WHERE type = :type AND content ~* :pat"
            ),
            {"type": memory_type, "pat": pattern},
        )
        for row in result.fetchall():
            chunk_boosts[str(row.chunk_id)] += relevance * REGEX_WEIGHT
    except Exception:
        pass


async def _doc_desc_regex_boosts(
    db: AsyncSession,
    pattern: str,
    relevance: float,
    doc_boosts: defaultdict[str, float],
    doc_desc_matched: set[str] | None = None,
) -> None:
    try:
        result = await db.execute(
            sa_text(
                "SELECT document_id FROM document_enrichments "
                "WHERE enrichment_status = 'finished' AND description ~* :pat"
            ),
            {"pat": pattern},
        )
        for row in result.fetchall():
            doc_id = str(row.document_id)
            doc_boosts[doc_id] += relevance * REGEX_WEIGHT
            if doc_desc_matched is not None:
                doc_desc_matched.add(doc_id)
    except Exception:
        pass


async def _doc_tag_regex_boosts(
    db: AsyncSession,
    pattern: str,
    relevance: float,
    doc_boosts: defaultdict[str, float],
    doc_tags_matched: dict[str, set[str]] | None = None,
) -> None:
    """Apply tag regex boosts. Optionally populate doc_tags_matched with (doc_id -> {tag})."""
    try:
        result = await db.execute(
            sa_text(
                "SELECT de.document_id, tag::text AS tag "
                "FROM document_enrichments de, "
                "     json_array_elements_text(de.tags) AS tag "
                "WHERE de.enrichment_status = 'finished' AND tag ~* :pat"
            ),
            {"pat": pattern},
        )
        for row in result.fetchall():
            doc_id = str(row.document_id)
            doc_boosts[doc_id] += relevance * REGEX_WEIGHT
            if doc_tags_matched is not None:
                if doc_id not in doc_tags_matched:
                    doc_tags_matched[doc_id] = set()
                doc_tags_matched[doc_id].add(row.tag)
    except Exception:
        pass


# ---------- main entry point ----------


def _signals_to_cues_and_regexes(signals: dict) -> tuple[list[dict], list[dict]]:
    """Extract generated_cues and generated_regexes from LLM signals."""
    cues: list[dict] = []
    regexes: list[dict] = []
    for sig_type in ("context_cues", "fact_cues", "description_cues", "tag_cues"):
        for item in signals.get(sig_type, []):
            t = item.get("text", "").strip()
            if t:
                cues.append({"text": t, "score": float(item.get("relevance", 0.5))})
    for sig_type in (
        "context_regex",
        "fact_regex",
        "description_regex",
        "tag_regex",
    ):
        for item in signals.get(sig_type, []):
            pat = item.get("pattern", "").strip()
            if pat and _valid_regex(pat):
                regexes.append(
                    {"text": pat, "score": float(item.get("relevance", 0.5))}
                )
    return cues, regexes


async def enhanced_search(
    query: str,
    db: AsyncSession,
    top_k: int = 10,
):
    """Search chunks with enrichment-based boosting and Cohere reranking.

    Returns EnhancedSearchResponse with user_query, generated_cues, generated_regexes,
    and results (DocumentResult per document).
    Falls back to basic vector+rerank when no enrichment data exists.
    """
    from app.schemas import (
        ChunkMatchDto,
        ContextDto,
        DocumentDto,
        DocumentResponse,
        DocumentResult,
        EnhancedSearchResponse,
        FactDto,
        CueResult,
        RegexResult,
    )

    query_embedding = get_embedding(query)
    fetch_limit = top_k * settings.rerank_top_n_multiplier
    generated_cues: list[dict] = []
    generated_regexes: list[dict] = []
    signals: dict = {}
    doc_desc_matched: set[str] = set()
    doc_tags_matched: dict[str, set[str]] = {}

    # 1. Base vector search
    result = await db.execute(
        sa_text(
            "SELECT c.id AS chunk_id, c.content, c.page_number, c.chunk_index, c.chunk_type, "
            "       c.document_id, d.filename, "
            "       c.embedding <=> :emb AS distance "
            "FROM chunks c "
            "JOIN documents d ON d.id = c.document_id "
            "ORDER BY c.embedding <=> :emb "
            "LIMIT :lim"
        ),
        {"emb": str(query_embedding), "lim": fetch_limit},
    )
    rows = result.fetchall()
    if not rows:
        return EnhancedSearchResponse(
            user_query=query,
            generated_cues=[],
            generated_regexes=[],
            results=[],
        )

    candidates = [
        {
            "chunk_id": row.chunk_id,
            "content": row.content,
            "page_number": row.page_number,
            "chunk_index": row.chunk_index,
            "chunk_type": getattr(row, "chunk_type", None),
            "document_id": row.document_id,
            "document_filename": row.filename,
            "score": round(1 - row.distance, 4),
        }
        for row in rows
    ]

    # 2. Check for enrichment data
    has_enrichment = await _has_enrichment_data(db)

    if has_enrichment:
        # 3. Expand query into 8 signal types
        try:
            signals = _expand_query(query)
            generated_cues, generated_regexes = _signals_to_cues_and_regexes(signals)
        except Exception:
            signals = {}

        # 4. Collect and embed all cue texts
        cue_texts: list[str] = []
        cue_meta: list[tuple[str, float, int]] = []

        for sig_type in ("context_cues", "fact_cues", "description_cues", "tag_cues"):
            for item in signals.get(sig_type, []):
                t = item.get("text", "").strip()
                r = float(item.get("relevance", 0.5))
                if t:
                    cue_meta.append((sig_type, r, len(cue_texts)))
                    cue_texts.append(t)

        chunk_boosts: defaultdict[str, float] = defaultdict(float)
        doc_boosts: defaultdict[str, float] = defaultdict(float)

        # 5. Embedding-cue boosts
        if cue_texts:
            cue_embeddings = get_embeddings(cue_texts)
            for sig_type, relevance, idx in cue_meta:
                emb = cue_embeddings[idx]
                if sig_type == "context_cues":
                    await _memory_cue_boosts(
                        db, emb, "context", relevance, chunk_boosts
                    )
                elif sig_type == "fact_cues":
                    await _memory_cue_boosts(db, emb, "fact", relevance, chunk_boosts)
                elif sig_type == "description_cues":
                    await _doc_cue_boosts(
                        db,
                        emb,
                        "description_embedding",
                        relevance,
                        doc_boosts,
                        doc_desc_matched=doc_desc_matched,
                    )
                elif sig_type == "tag_cues":
                    await _doc_cue_boosts(
                        db, emb, "tags_embedding", relevance, doc_boosts
                    )

        # 6. Regex boosts
        for sig_type in (
            "context_regex",
            "fact_regex",
            "description_regex",
            "tag_regex",
        ):
            for item in signals.get(sig_type, []):
                pat = item.get("pattern", "")
                relevance = float(item.get("relevance", 0.5))
                if not pat or not _valid_regex(pat):
                    continue
                if sig_type == "context_regex":
                    await _memory_regex_boosts(
                        db, pat, "context", relevance, chunk_boosts
                    )
                elif sig_type == "fact_regex":
                    await _memory_regex_boosts(db, pat, "fact", relevance, chunk_boosts)
                elif sig_type == "description_regex":
                    await _doc_desc_regex_boosts(
                        db, pat, relevance, doc_boosts, doc_desc_matched
                    )
                elif sig_type == "tag_regex":
                    await _doc_tag_regex_boosts(
                        db, pat, relevance, doc_boosts, doc_tags_matched
                    )

        # 7. Apply boosts to candidate scores
        for c in candidates:
            boost = chunk_boosts.get(str(c["chunk_id"]), 0.0) + doc_boosts.get(
                str(c["document_id"]), 0.0
            )
            c["score"] = round(c["score"] + boost, 4)

    # 8. Sort and rerank (either with or without boosts)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    pool = candidates[:fetch_limit]
    reranked = rerank(query, pool, top_n=top_k)

    # 9. Group by document and build DocumentResult

    doc_to_chunks: dict[uuid.UUID, list[dict]] = {}
    for item in reranked:
        doc_id = item["document_id"]
        if doc_id not in doc_to_chunks:
            doc_to_chunks[doc_id] = []
        doc_to_chunks[doc_id].append(item)

    if not doc_to_chunks:
        return EnhancedSearchResponse(
            user_query=query,
            generated_cues=[CueResult(**c) for c in generated_cues],
            generated_regexes=[RegexResult(**r) for r in generated_regexes],
            results=[],
            chunk_rank_order=[],
            document_rank_order=[],
        )

    chunk_rank_order = [item["chunk_id"] for item in reranked]
    doc_seen: set[uuid.UUID] = set()
    document_rank_order: list[uuid.UUID] = []
    for item in reranked:
        doc_id = item["document_id"]
        if doc_id not in doc_seen:
            doc_seen.add(doc_id)
            document_rank_order.append(doc_id)

    # Fetch documents with enrichment
    doc_ids = list(doc_to_chunks.keys())
    docs_result = await db.execute(
        select(Document)
        .options(selectinload(Document.enrichment))
        .where(Document.id.in_(doc_ids))
    )
    docs_list = docs_result.scalars().all()
    docs_by_id: dict[uuid.UUID, Document] = {
        cast(uuid.UUID, d.id): d for d in docs_list
    }

    # Fetch memories for all matched chunks
    chunk_ids = [c["chunk_id"] for chunks in doc_to_chunks.values() for c in chunks]
    mem_result = await db.execute(select(Memory).where(Memory.chunk_id.in_(chunk_ids)))
    memories = mem_result.scalars().all()
    memories_by_chunk: dict[uuid.UUID, list] = {}
    for m in memories:
        cid = cast(uuid.UUID, m.chunk_id)
        if cid not in memories_by_chunk:
            memories_by_chunk[cid] = []
        memories_by_chunk[cid].append(m)

    results: list[DocumentResult] = []
    for doc_id, chunks in doc_to_chunks.items():
        doc = docs_by_id.get(doc_id)
        if not doc:
            continue

        enrichment = doc.enrichment
        title = enrichment.title if enrichment else None
        description = enrichment.description if enrichment else None
        tags = enrichment.tags if enrichment and enrichment.tags else None
        if tags is None:
            tags = []

        # Sort by score descending so best match is first
        sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
        matched_chunks = [
            ChunkMatchDto(
                chunk_id=c["chunk_id"],
                content=c["content"],
                page_number=c["page_number"],
                chunk_index=c["chunk_index"],
                chunk_type=c.get("chunk_type"),
                score=c["score"],
            )
            for c in sorted_chunks
        ]
        matched_chunk_ids = [c["chunk_id"] for c in chunks]

        contexts: list[ContextDto] = []
        facts: list[FactDto] = []
        for cid in matched_chunk_ids:
            for m in memories_by_chunk.get(cid, []):
                if m.type == "context":
                    contexts.append(ContextDto(content=m.content, chunk_id=m.chunk_id))
                elif m.type == "fact":
                    facts.append(FactDto(content=m.content, chunk_id=m.chunk_id))

        description_matched = str(doc_id) in doc_desc_matched
        matched_tags = list(doc_tags_matched.get(str(doc_id), []))

        base = DocumentResponse.model_validate(doc)
        doc_dto = DocumentDto(
            **base.model_dump(),
            title=title,
            description=description,
            tags=tags,
        )
        results.append(
            DocumentResult(
                document=doc_dto,
                matched_chunks=matched_chunks,
                matched_contexts=contexts,
                matched_facts=facts,
                description_matched=description_matched,
                matched_tags=matched_tags,
            )
        )

    return EnhancedSearchResponse(
        user_query=query,
        generated_cues=[CueResult(**c) for c in generated_cues],
        generated_regexes=[RegexResult(**r) for r in generated_regexes],
        results=results,
        chunk_rank_order=chunk_rank_order,
        document_rank_order=document_rank_order,
    )


def _dicts_to_search_results(items: list[dict]) -> list[SearchResult]:
    """Convert rerank output dicts to SearchResult dataclass instances."""
    return [
        SearchResult(
            chunk_id=item["chunk_id"],
            content=item["content"],
            page_number=item["page_number"],
            chunk_index=item["chunk_index"],
            chunk_type=item.get("chunk_type"),
            document_id=item["document_id"],
            document_filename=item["document_filename"],
            score=item["score"],
        )
        for item in items
    ]


async def enhanced_search_in_document(
    query: str,
    document_id: uuid.UUID,
    db: AsyncSession,
    top_k: int = 50,
) -> list[SearchResult]:
    """Search chunks within a specific document with enrichment-based boosting and Cohere reranking.

    Returns list of SearchResult dataclass instances.
    Falls back to basic vector+rerank when no enrichment data exists.
    """
    query_embedding = get_embedding(query)
    fetch_limit = top_k * settings.rerank_top_n_multiplier

    # 1. Base vector search filtered by document_id
    result = await db.execute(
        sa_text(
            "SELECT c.id AS chunk_id, c.content, c.page_number, c.chunk_index, c.chunk_type, "
            "       c.document_id, d.filename, "
            "       c.embedding <=> :emb AS distance "
            "FROM chunks c "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE c.document_id = :doc_id "
            "ORDER BY c.embedding <=> :emb "
            "LIMIT :lim"
        ),
        {"emb": str(query_embedding), "doc_id": str(document_id), "lim": fetch_limit},
    )
    rows = result.fetchall()
    if not rows:
        return []

    candidates = [
        {
            "chunk_id": row.chunk_id,
            "content": row.content,
            "page_number": row.page_number,
            "chunk_index": row.chunk_index,
            "chunk_type": getattr(row, "chunk_type", None),
            "document_id": row.document_id,
            "document_filename": row.filename,
            "score": round(1 - row.distance, 4),
        }
        for row in rows
    ]

    # 2. Check for enrichment data; fall back to plain rerank if absent
    if not await _has_enrichment_data(db):
        return _dicts_to_search_results(rerank(query, candidates, top_n=top_k))

    # 3. Expand query into 8 signal types
    try:
        signals = _expand_query(query)
    except Exception:
        return _dicts_to_search_results(rerank(query, candidates, top_n=top_k))

    # 4. Collect and embed all cue texts in one batch
    cue_texts: list[str] = []
    cue_meta: list[tuple[str, float, int]] = []  # (signal_type, relevance, idx)

    for sig_type in ("context_cues", "fact_cues", "description_cues", "tag_cues"):
        for item in signals.get(sig_type, []):
            t = item.get("text", "").strip()
            r = float(item.get("relevance", 0.5))
            if t:
                cue_meta.append((sig_type, r, len(cue_texts)))
                cue_texts.append(t)

    chunk_boosts: defaultdict[str, float] = defaultdict(float)
    doc_boosts: defaultdict[str, float] = defaultdict(float)

    # 5. Embedding-cue boosts (only for chunks in this document)
    if cue_texts:
        cue_embeddings = get_embeddings(cue_texts)
        for sig_type, relevance, idx in cue_meta:
            emb = cue_embeddings[idx]
            if sig_type == "context_cues":
                await _memory_cue_boosts(db, emb, "context", relevance, chunk_boosts)
            elif sig_type == "fact_cues":
                await _memory_cue_boosts(db, emb, "fact", relevance, chunk_boosts)
            elif sig_type == "description_cues":
                await _doc_cue_boosts(
                    db, emb, "description_embedding", relevance, doc_boosts
                )
            elif sig_type == "tag_cues":
                await _doc_cue_boosts(db, emb, "tags_embedding", relevance, doc_boosts)

    # 6. Regex boosts (only for chunks in this document)
    for sig_type in ("context_regex", "fact_regex", "description_regex", "tag_regex"):
        for item in signals.get(sig_type, []):
            pat = item.get("pattern", "")
            relevance = float(item.get("relevance", 0.5))
            if not pat or not _valid_regex(pat):
                continue
            if sig_type == "context_regex":
                await _memory_regex_boosts(db, pat, "context", relevance, chunk_boosts)
            elif sig_type == "fact_regex":
                await _memory_regex_boosts(db, pat, "fact", relevance, chunk_boosts)
            elif sig_type == "description_regex":
                await _doc_desc_regex_boosts(db, pat, relevance, doc_boosts)
            elif sig_type == "tag_regex":
                await _doc_tag_regex_boosts(db, pat, relevance, doc_boosts)

    # 7. Apply boosts to candidate scores (filter boosts to only this document's chunks)
    doc_chunk_ids = {str(c["chunk_id"]) for c in candidates}
    # Filter chunk_boosts to only include chunks in this document
    filtered_chunk_boosts = {
        chunk_id: boost
        for chunk_id, boost in chunk_boosts.items()
        if chunk_id in doc_chunk_ids
    }
    for c in candidates:
        boost = filtered_chunk_boosts.get(str(c["chunk_id"]), 0.0)
        # Only apply doc boosts if this is the target document
        if str(c["document_id"]) == str(document_id):
            boost += doc_boosts.get(str(c["document_id"]), 0.0)
        c["score"] = round(c["score"] + boost, 4)

    # 8. Sort by boosted score, then Cohere rerank the top pool
    candidates.sort(key=lambda x: x["score"], reverse=True)
    pool = candidates[:fetch_limit]

    return _dicts_to_search_results(rerank(query, pool, top_n=top_k))
