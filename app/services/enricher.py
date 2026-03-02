"""Single-document enrichment service. Used by CLI batch runner."""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm

from app.config import settings
from app.services.embeddings import get_embeddings

client = OpenAI(api_key=settings.openai_api_key)


@dataclass
class STMItem:
    stm_id: int
    type: str
    content: str
    source_chunk_id: str


@dataclass
class LTMItem:
    type: str
    content: str
    source_chunk_id: str


@dataclass
class ShortTermMemory:
    items: list[STMItem] = field(default_factory=list)
    _next_id: int = 0

    def next_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def add(self, type: str, content: str, chunk_id: str) -> None:
        self.items.append(STMItem(self.next_id(), type, content, chunk_id))

    def remove(self, stm_id: int) -> STMItem | None:
        for i, item in enumerate(self.items):
            if item.stm_id == stm_id:
                return self.items.pop(i)
        return None

    def update(self, stm_id: int, content: str, chunk_id: str) -> bool:
        for item in self.items:
            if item.stm_id == stm_id:
                item.content = content
                item.source_chunk_id = chunk_id
                return True
        return False

    def drop_oldest(self, max_size: int) -> list[STMItem]:
        dropped = []
        while len(self.items) > max_size:
            dropped.append(self.items.pop(0))
        return dropped

    def to_prompt_list(self) -> str:
        if not self.items:
            return "(empty)"
        lines = []
        for item in self.items:
            lines.append(f"  [id={item.stm_id}] ({item.type}) {item.content}")
        return "\n".join(lines)


STM_SYSTEM_PROMPT = """\
You are a document analysis assistant processing a document chunk by chunk. \
You maintain a short-term memory of two types of items:

- **context**: Background information, setting, topic, theme, or narrative thread.
- **fact**: Specific claims, data points, definitions, statistics, or conclusions.

Given the current short-term memory and a new chunk of text, decide what to \
add, remove, or update. Be selective — keep only items that remain relevant. \
Prefer updating existing items over removing+adding when the meaning evolves.

Respond with ONLY valid JSON (no markdown fences):
{
  "add": [{"type": "context"|"fact", "content": "..."}],
  "remove": [<stm_id>, ...],
  "update": [{"id": <stm_id>, "content": "new content"}]
}

All three keys must be present. Use empty lists if no action is needed for that category."""

METADATA_SYSTEM_PROMPT = """\
You are a creative document analyst. Given a collection of contexts and facts \
extracted from a document, generate:

1. A creative, evocative title for the document
2. A rich description (2-4 sentences) that captures its essence
3. A list of relevant tags (5-15 tags, mix of broad and specific)

Be creative and insightful. Respond with ONLY valid JSON (no markdown fences):
{
  "title": "...",
  "description": "...",
  "tags": ["tag1", "tag2", ...]
}"""


def call_llm_json(system: str, user: str) -> dict:
    response = client.chat.completions.create(
        model=settings.enrichment_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def process_chunk_with_stm(
    stm: ShortTermMemory, chunk_content: str, chunk_id: str
) -> list[STMItem]:
    """Process one chunk: call LLM, apply STM ops, return any dropped items."""
    user_msg = (
        f"Current short-term memory:\n{stm.to_prompt_list()}\n\n"
        f"New chunk:\n{chunk_content}"
    )
    result = call_llm_json(STM_SYSTEM_PROMPT, user_msg)

    for item in result.get("remove", []):
        stm.remove(item)

    for item in result.get("update", []):
        stm.update(item["id"], item["content"], chunk_id)

    for item in result.get("add", []):
        stm.add(item["type"], item["content"], chunk_id)

    return stm.drop_oldest(settings.stm_max_size)


def sample_ltm_for_metadata(ltm_items: list[LTMItem]) -> list[LTMItem]:
    """If combined text is too long, randomly sample items to fit."""
    total = sum(len(item.content) for item in ltm_items)
    if total <= settings.ltm_sample_max_chars:
        return ltm_items

    shuffled = list(ltm_items)
    random.shuffle(shuffled)
    sampled = []
    chars = 0
    for item in shuffled:
        if chars + len(item.content) > settings.ltm_sample_max_chars:
            continue
        sampled.append(item)
        chars += len(item.content)
    return sampled


def generate_metadata(ltm_items: list[LTMItem]) -> dict:
    """Use LTM to generate document title, description, and tags."""
    sampled = sample_ltm_for_metadata(ltm_items)
    if not sampled:
        return {"title": "Untitled", "description": "", "tags": []}

    lines = []
    for item in sampled:
        lines.append(f"[{item.type}] {item.content}")
    user_msg = "Extracted contexts and facts:\n\n" + "\n".join(lines)

    return call_llm_json(METADATA_SYSTEM_PROMPT, user_msg)


def _update_enrichment_status(
    session: Session,
    doc_id: str,
    status: str,
    error: str | None = None,
) -> None:
    existing = session.execute(
        text("SELECT id FROM document_enrichments WHERE document_id = :doc_id"),
        {"doc_id": doc_id},
    ).fetchone()

    if existing:
        session.execute(
            text(
                "UPDATE document_enrichments SET enrichment_status = :status, "
                "last_enriched_at = :now, last_enrich_error = :error "
                "WHERE document_id = :doc_id"
            ),
            {
                "status": status,
                "now": datetime.utcnow(),
                "error": error,
                "doc_id": doc_id,
            },
        )
    else:
        session.execute(
            text(
                "INSERT INTO document_enrichments (id, document_id, enrichment_status, "
                "last_enriched_at, last_enrich_error) "
                "VALUES (gen_random_uuid(), :doc_id, :status, :now, :error)"
            ),
            {
                "doc_id": doc_id,
                "status": status,
                "now": datetime.utcnow(),
                "error": error,
            },
        )
    session.commit()


def enrich_document(session: Session, doc_id: str, filename: str) -> int:
    """Enrich a single document. Returns the number of memories created."""
    _update_enrichment_status(session, doc_id, "started")

    # Clear previous memories for this document
    session.execute(
        text("DELETE FROM memories WHERE document_id = :doc_id"),
        {"doc_id": doc_id},
    )
    session.commit()

    chunks = session.execute(
        text(
            "SELECT id, content, chunk_index FROM chunks "
            "WHERE document_id = :doc_id ORDER BY chunk_index"
        ),
        {"doc_id": doc_id},
    ).fetchall()

    if not chunks:
        _update_enrichment_status(session, doc_id, "errored", "No chunks found")
        print(f"  WARNING: No chunks for document '{filename}'.")
        return 0

    try:
        stm = ShortTermMemory()
        ltm: list[LTMItem] = []

        print(f"  Processing {len(chunks)} chunks for '{filename}'...")
        for chunk in tqdm(chunks, desc=f"  Enriching {filename}", unit="chunk"):
            dropped = process_chunk_with_stm(stm, chunk.content, str(chunk.id))
            for item in dropped:
                ltm.append(LTMItem(item.type, item.content, item.source_chunk_id))

        # Move remaining STM to LTM
        for item in stm.items:
            ltm.append(LTMItem(item.type, item.content, item.source_chunk_id))

        if not ltm:
            _update_enrichment_status(
                session, doc_id, "errored", "No memories produced"
            )
            print(f"  WARNING: No memories produced for '{filename}'.")
            return 0

        # Generate document metadata
        print(f"  Generating metadata for '{filename}'...")
        metadata = generate_metadata(ltm)
        title = metadata.get("title", "Untitled")
        description = metadata.get("description", "")
        tags = metadata.get("tags", [])

        # Embed description and tags
        desc_text = description or "empty"
        tags_text = ", ".join(tags) if tags else "empty"
        desc_emb, tags_emb = get_embeddings([desc_text, tags_text])

        # Upsert document enrichment
        existing = session.execute(
            text("SELECT id FROM document_enrichments WHERE document_id = :doc_id"),
            {"doc_id": doc_id},
        ).fetchone()

        if existing:
            session.execute(
                text(
                    "UPDATE document_enrichments SET title = :title, description = :desc, "
                    "tags = :tags, description_embedding = :desc_emb, "
                    "tags_embedding = :tags_emb, enrichment_status = 'finished', "
                    "last_enriched_at = :now, last_enrich_error = NULL "
                    "WHERE document_id = :doc_id"
                ),
                {
                    "title": title,
                    "desc": description,
                    "tags": json.dumps(tags),
                    "desc_emb": str(desc_emb),
                    "tags_emb": str(tags_emb),
                    "now": datetime.utcnow(),
                    "doc_id": doc_id,
                },
            )
        else:
            session.execute(
                text(
                    "INSERT INTO document_enrichments "
                    "(id, document_id, title, description, tags, "
                    "description_embedding, tags_embedding, enrichment_status, "
                    "last_enriched_at) "
                    "VALUES (gen_random_uuid(), :doc_id, :title, :desc, :tags, "
                    ":desc_emb, :tags_emb, 'finished', :now)"
                ),
                {
                    "doc_id": doc_id,
                    "title": title,
                    "desc": description,
                    "tags": json.dumps(tags),
                    "desc_emb": str(desc_emb),
                    "tags_emb": str(tags_emb),
                    "now": datetime.utcnow(),
                },
            )
        session.commit()

        # Embed and store memories
        print(f"  Embedding {len(ltm)} memories for '{filename}'...")
        batch_size = 100
        memory_count = 0
        for i in range(0, len(ltm), batch_size):
            batch = ltm[i : i + batch_size]
            texts = [item.content for item in batch]
            embeddings = get_embeddings(texts)

            for item, emb in zip(batch, embeddings):
                session.execute(
                    text(
                        "INSERT INTO memories (id, document_id, chunk_id, type, content, embedding) "
                        "VALUES (gen_random_uuid(), :doc_id, :chunk_id, :type, :content, :embedding)"
                    ),
                    {
                        "doc_id": doc_id,
                        "chunk_id": item.source_chunk_id,
                        "type": item.type,
                        "content": item.content,
                        "embedding": str(emb),
                    },
                )
                memory_count += 1

        session.commit()
        print(
            f"  Enriched '{filename}': {memory_count} memories, "
            f"title='{title}', {len(tags)} tags"
        )
        return memory_count

    except Exception as e:
        session.rollback()
        _update_enrichment_status(session, doc_id, "errored", str(e))
        raise
