"""Generate test questions for search evaluation.

For each document, randomly samples chunks and uses an LLM to generate one search
query per sampled chunk. Ensures equal number of questions per document.

  make test-questions              Generate 10 questions per document (default)
  make test-questions N=5          Generate 5 questions per document
"""

import argparse
import json
import random
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tqdm import tqdm

from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

DEFAULT_WORKERS = 4

QUERY_GENERATION_SYSTEM_PROMPT = """\
You are a search evaluation assistant. Given a document chunk and optionally a related \
memory (context or fact extracted from that chunk), generate exactly ONE search query \
that a user might type to find this information.

Choose ONE type from: direct, paraphrase, keyword, conceptual, fact-seeking, summary-style
- **direct**: Literal or near-literal phrasing from the chunk
- **paraphrase**: Same meaning, different wording
- **keyword**: Sparse keywords or key phrases only
- **conceptual**: Abstract or conceptual question the chunk answers
- **fact-seeking**: Question asking for a fact or detail
- **summary-style**: "What does X say about Y?" or "Where is Z mentioned?"

The query should be a realistic, self-contained search string (1-15 words typically).

Mild bias: Prefer queries that anchor on something distinctive in this chunk (a name, number, \
specific claim, or phrasing) so that the same chunk is likelier to rank above nearby chunks. \
Stay natural—don't force unnatural specificity.

Respond with ONLY valid JSON (no markdown fences):
{
  "queries": [
    {"text": "the search query string", "type": "direct|paraphrase|keyword|conceptual|fact-seeking|summary-style"}
  ]
}"""


def get_documents_with_chunks(session: Session) -> list[dict]:
    """Fetch all finished documents with their chunks and memories.
    Returns list of {document_id, filename, chunks: [{chunk_id, content, memories}]}.
    """
    # Get all chunks for finished documents, with memories
    rows = session.execute(
        text("""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content AS chunk_content,
                d.filename,
                m.id AS memory_id,
                m.type AS memory_type,
                m.content AS memory_content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN memories m ON m.chunk_id = c.id
            WHERE d.processed_status = 'finished'
        """),
    ).fetchall()
    # Group by document, then by chunk
    by_doc: dict[uuid.UUID, dict] = {}
    by_chunk: dict[uuid.UUID, dict] = {}
    for r in rows:
        did = r.document_id
        cid = r.chunk_id
        if did not in by_doc:
            by_doc[did] = {"document_id": did, "filename": r.filename, "chunks": {}}
        if cid not in by_chunk:
            chunk_data = {
                "chunk_id": cid,
                "document_id": did,
                "chunk_content": r.chunk_content,
                "filename": r.filename,
                "memories": [],
            }
            by_doc[did]["chunks"][cid] = chunk_data
            by_chunk[cid] = chunk_data
        if r.memory_id:
            by_chunk[cid]["memories"].append(
                {"id": r.memory_id, "type": r.memory_type, "content": r.memory_content}
            )
    # Convert to list of docs with list of chunks
    result = []
    for doc in by_doc.values():
        doc["chunks"] = list(doc["chunks"].values())
        result.append(doc)
    return result


def generate_queries(chunk_content: str, memories: list[dict]) -> list[tuple[str, str]]:
    """Use LLM to generate diverse queries. Returns list of (text, query_type)."""
    memory_desc = ""
    if memories:
        mem_str = "\n".join(
            f"- [{m['type']}] {m['content'][:300]}" for m in memories[:3]
        )
        memory_desc = f"\nRelated memories:\n{mem_str}"
    chunk_preview = chunk_content[:800].strip()
    if len(chunk_content) > 800:
        chunk_preview += "..."

    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": QUERY_GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Chunk text:\n{chunk_preview}{memory_desc}",
            },
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    queries = data.get("queries", [])
    for q in queries:
        text_val = q.get("text", "").strip()
        qtype = q.get("type", "direct").lower()
        if text_val and qtype:
            return [(text_val, qtype)]
    return []


def _process_one_doc(
    doc: dict,
    per_doc: int,
    existing_count: int,
    existing_chunk_ids: set[str],
    engine,
) -> int:
    """Process one document with its own DB session. Returns number of questions created."""
    doc_id = str(doc["document_id"])
    chunks = doc["chunks"]
    if not chunks:
        return 0

    need = per_doc - existing_count
    if need <= 0:
        return 0

    available = [c for c in chunks if str(c["chunk_id"]) not in existing_chunk_ids]
    if not available:
        return 0
    n_sample = min(need, len(available))
    sampled = random.sample(available, n_sample)

    created = 0
    with Session(engine) as session:
        for chunk in sampled:
            queries = generate_queries(
                chunk["chunk_content"],
                chunk["memories"],
            )
            if not queries:
                continue
            text_val, qtype = queries[0]
            memory = chunk["memories"][0] if chunk["memories"] else None
            session.execute(
                text("""
                    INSERT INTO search_test_questions
                    (id, question, query_type, target_chunk_id, target_document_id, source_memory_id)
                    VALUES (:id, :question, :query_type, :target_chunk_id, :target_document_id, :source_memory_id)
                """),
                {
                    "id": str(uuid.uuid4()),
                    "question": text_val,
                    "query_type": qtype,
                    "target_chunk_id": str(chunk["chunk_id"]),
                    "target_document_id": doc_id,
                    "source_memory_id": str(memory["id"]) if memory else None,
                },
            )
            created += 1
        session.commit()
    return created


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate test questions for search evaluation (equal per document)"
    )
    parser.add_argument(
        "n",
        nargs="?",
        default=10,
        type=int,
        help="Number of questions to generate per document (default 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel workers for document-level processing (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()
    per_doc = max(1, args.n)
    workers = args.workers

    engine = create_engine(settings.database_url_sync)
    with Session(engine) as session:
        docs = get_documents_with_chunks(session)
        if not docs:
            print("No documents with chunks found. Run ingest first.", file=sys.stderr)
            return 1

        # Build existing: doc_id -> (count of questions, set of chunk_ids with questions)
        existing_rows = session.execute(
            text(
                "SELECT target_document_id, target_chunk_id FROM search_test_questions"
            )
        ).fetchall()
        existing_count_by_doc: dict[str, int] = {}
        existing_chunks_by_doc: dict[str, set[str]] = {}
        for r in existing_rows:
            did = str(r.target_document_id)
            cid = str(r.target_chunk_id)
            existing_count_by_doc[did] = existing_count_by_doc.get(did, 0) + 1
            if did not in existing_chunks_by_doc:
                existing_chunks_by_doc[did] = set()
            existing_chunks_by_doc[did].add(cid)

    # Build list of (doc, existing_count, existing_chunk_ids) for docs that need work
    to_process: list[tuple[dict, int, set[str]]] = []
    for doc in docs:
        doc_id = str(doc["document_id"])
        existing_count = existing_count_by_doc.get(doc_id, 0)
        existing_chunks = existing_chunks_by_doc.get(doc_id, set())
        if per_doc - existing_count <= 0:
            continue
        available = [c for c in doc["chunks"] if str(c["chunk_id"]) not in existing_chunks]
        if not available:
            continue
        to_process.append((doc, existing_count, existing_chunks))

    if not to_process:
        print(
            f"No documents need questions (target {per_doc} per doc, {len(docs)} documents)."
        )
        return 0

    created = 0
    if workers <= 1:
        for doc, existing_count, existing_chunk_ids in tqdm(to_process, desc="Documents"):
            created += _process_one_doc(
                doc, per_doc, existing_count, existing_chunk_ids, engine
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_one_doc,
                    doc,
                    per_doc,
                    existing_count,
                    existing_chunk_ids,
                    engine,
                ): doc
                for doc, existing_count, existing_chunk_ids in to_process
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Documents"):
                created += future.result()

    print(
        f"Created {created} test questions ({per_doc} per document, {len(docs)} documents)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
