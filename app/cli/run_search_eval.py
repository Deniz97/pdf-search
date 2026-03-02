"""Run search evaluation over generated test questions.

Loads questions from search_test_questions, runs each through enhanced_search
(using same top_k=10 as live UI search), records chunk/document ranks, and
reports accuracy for top1, top2, top4, top8.

- Results are persisted immediately after each question (resilient to crashes).
- Use --resume-run-id to skip already-processed questions.
- To regenerate: DELETE FROM search_eval_results WHERE run_id = :id

Writes sanity-check output to eval_results/[timestamp]/:
  - configs.md: eval configuration
  - example_result_1.md, example_result_2.md, example_result_3.md: sample Q&A
  - results.md: detailed metrics with precision/recall per document and per chunk

  make search-eval
"""

import argparse
import asyncio
import logging
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.cli.utils import setup_signal_handler
from app.database import async_session, sync_engine
from app.services.search import enhanced_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Same as live UI search (form default, SearchOnlyRequest default)
SEARCH_TOP_K = 10

# Metrics to report
K_VALUES = (1, 2, 4, 8)

# Concurrent questions per batch
BATCH_SIZE = 4

# Retries for transient DB errors when persisting
PERSIST_RETRIES = 3
PERSIST_RETRY_DELAY = 2.0


def _chunks(items: list, size: int):
    """Yield successive chunks of size from items."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _persist_result(
    engine,
    run_id: str,
    result: dict,
) -> None:
    """Persist a single result to DB. Commits immediately. Retries on transient errors."""
    import json

    for attempt in range(PERSIST_RETRIES):
        try:
            with Session(engine) as sync_session:
                sync_session.execute(
                    text("""
                        INSERT INTO search_eval_results
                        (id, run_id, question_id, chunk_rank, doc_rank,
                         query_type, target_chunk_id, target_document_id,
                         chunk_rank_order, document_rank_order)
                        VALUES (:id, :run_id, :question_id, :chunk_rank, :doc_rank,
                                :query_type, :target_chunk_id, :target_document_id,
                                CAST(:chunk_rank_order AS jsonb), CAST(:document_rank_order AS jsonb))
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "run_id": run_id,
                        "question_id": str(result["question_id"]),
                        "chunk_rank": result["chunk_rank"],
                        "doc_rank": result["doc_rank"],
                        "query_type": result["query_type"],
                        "target_chunk_id": str(result["target_chunk_id"]),
                        "target_document_id": str(result["target_doc_id"]),
                        "chunk_rank_order": json.dumps(result["chunk_rank_order"]),
                        "document_rank_order": json.dumps(
                            result["document_rank_order"]
                        ),
                    },
                )
                sync_session.commit()
            return
        except Exception as e:
            if attempt < PERSIST_RETRIES - 1:
                log.warning(
                    "Persist failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    PERSIST_RETRIES,
                    e,
                    PERSIST_RETRY_DELAY,
                )
                time.sleep(PERSIST_RETRY_DELAY)
            else:
                raise


async def _process_one_question(row: tuple, idx: int, total: int) -> dict:
    """Process a single question with its own DB session. Returns result dict for aggregation."""
    q_id, question, query_type, target_chunk_id, target_doc_id = row
    q_preview = (question[:50] + "…") if len(question) > 50 else question

    async with async_session() as db:
        t0 = time.perf_counter()
        response = await enhanced_search(question, db, top_k=SEARCH_TOP_K)
        elapsed = time.perf_counter() - t0

    chunk_rank = None
    doc_rank = None
    target_chunk_str = str(target_chunk_id)
    target_doc_str = str(target_doc_id)

    for i, cid in enumerate(response.chunk_rank_order):
        if str(cid) == target_chunk_str:
            chunk_rank = i + 1
            break
    for i, did in enumerate(response.document_rank_order):
        if str(did) == target_doc_str:
            doc_rank = i + 1
            break

    c_status = f"chunk#{chunk_rank}" if chunk_rank else "chunk:miss"
    d_status = f"doc#{doc_rank}" if doc_rank else "doc:miss"
    log.info(
        "[%d/%d] %s → %s | %s | %.1fs",
        idx,
        total,
        q_preview,
        c_status,
        d_status,
        elapsed,
    )

    chunk_rank_order = [str(cid) for cid in response.chunk_rank_order]
    document_rank_order = [str(did) for did in response.document_rank_order]

    return {
        "question_id": q_id,
        "question": question,
        "query_type": query_type,
        "target_chunk_id": str(target_chunk_id),
        "target_doc_id": str(target_doc_id),
        "chunk_rank": chunk_rank,
        "doc_rank": doc_rank,
        "chunk_rank_order": chunk_rank_order,
        "document_rank_order": document_rank_order,
        "timings": getattr(response, "timings", None) or {},
    }


def _sample_rows(
    rows: list,
    documents: int | None,
    questions_per_document: int | None,
    limit: int | None,
) -> list:
    """Apply document-based sampling and optional total limit."""
    if documents is not None or questions_per_document is not None:
        by_doc: dict[str, list] = defaultdict(list)
        for row in rows:
            target_doc_id = str(row[4])  # target_document_id
            by_doc[target_doc_id].append(row)
        doc_ids = sorted(by_doc.keys())
        if documents is not None:
            doc_ids = doc_ids[:documents]
        sampled = []
        for doc_id in doc_ids:
            doc_rows = by_doc[doc_id]
            if questions_per_document is not None:
                doc_rows = doc_rows[:questions_per_document]
            sampled.extend(doc_rows)
        rows = sampled
    if limit is not None:
        rows = rows[:limit]
    return rows


def _load_results_from_db_simple(engine, run_id: str) -> list[dict]:
    """Load results when question may not be in JOIN (e.g. new columns, simple schema)."""
    with Session(engine) as sync_session:
        rows = sync_session.execute(
            text("""
                SELECT r.question_id, r.query_type,
                       r.target_chunk_id, r.target_document_id,
                       r.chunk_rank, r.doc_rank,
                       r.chunk_rank_order, r.document_rank_order
                FROM search_eval_results r
                WHERE r.run_id = :run_id
            """),
            {"run_id": run_id},
        ).fetchall()

        # Get question text from search_test_questions for example output
        question_ids = [str(r[0]) for r in rows]
        q_texts: dict[str, str] = {}
        if question_ids:
            q_rows = sync_session.execute(
                text(
                    "SELECT id::text, question FROM search_test_questions WHERE id::text = ANY(:ids)"
                ),
                {"ids": question_ids},
            ).fetchall()
            for qid, qtext in q_rows:
                q_texts[qid] = qtext

    results = []
    for row in rows:
        (
            q_id,
            query_type,
            target_chunk_id,
            target_doc_id,
            chunk_rank,
            doc_rank,
            cro,
            dro,
        ) = row
        q_id_str = str(q_id)
        results.append(
            {
                "question_id": q_id,
                "question": q_texts.get(q_id_str, "(stored result)"),
                "query_type": query_type or "unknown",
                "target_chunk_id": str(target_chunk_id),
                "target_doc_id": str(target_doc_id),
                "chunk_rank": chunk_rank,
                "doc_rank": doc_rank,
                "chunk_rank_order": cro or [],
                "document_rank_order": dro or [],
            }
        )
    return results


async def run_eval(
    notes: str | None = None,
    limit: int | None = None,
    documents: int | None = None,
    questions_per_document: int | None = None,
    workers: int = BATCH_SIZE,
    shutdown_requested: list | None = None,
    resume_run_id: str | None = None,
) -> dict:
    """Run evaluation and return summary metrics.
    Results are persisted after each question. Use resume_run_id to skip already-processed questions.
    """
    with Session(sync_engine) as sync_session:
        log.info("Loading test questions...")
        rows = sync_session.execute(
            text("""
                SELECT id, question, query_type, target_chunk_id, target_document_id
                FROM search_test_questions
            """)
        ).fetchall()

        if not rows:
            return {"error": "No test questions found. Run: make generate-questions"}

        rows = _sample_rows(rows, documents, questions_per_document, limit)
        if documents is not None or questions_per_document is not None:
            doc_str = f"{documents} docs" if documents else "all docs"
            qpd_str = (
                f"{questions_per_document} q/doc"
                if questions_per_document
                else "all q/doc"
            )
            log.info("Sampled: %s × %s → %d questions", doc_str, qpd_str, len(rows))
        elif limit:
            log.info("Limited to %d questions (--limit)", len(rows))
        else:
            log.info("Loaded %d questions", len(rows))

        # Create or resume run
        if resume_run_id:
            run_id = uuid.UUID(resume_run_id)
            existing = sync_session.execute(
                text("SELECT id FROM search_eval_runs WHERE id = :id"),
                {"id": str(run_id)},
            ).fetchone()
            if not existing:
                return {"error": f"Run ID {resume_run_id} not found. Cannot resume."}
            done_ids = {
                str(r[0])
                for r in sync_session.execute(
                    text(
                        "SELECT question_id::text FROM search_eval_results WHERE run_id = :run_id"
                    ),
                    {"run_id": str(run_id)},
                ).fetchall()
            }
            rows = [r for r in rows if str(r[0]) not in done_ids]
            log.info(
                "Resuming run %s: %d already done, %d remaining",
                run_id,
                len(done_ids),
                len(rows),
            )
            if not rows:
                log.info(
                    "All questions already processed. Computing metrics from DB..."
                )
                full_results = _load_results_from_db_simple(sync_engine, str(run_id))
                return _compute_and_write_output(
                    sync_engine,
                    run_id,
                    full_results,
                    workers,
                    notes,
                    limit,
                    documents,
                    questions_per_document,
                    timing_samples=[],
                )
        else:
            run_id = uuid.uuid4()
            sync_session.execute(
                text("""
                    INSERT INTO search_eval_runs (id, top_k, notes)
                    VALUES (:id, :top_k, :notes)
                """),
                {"id": str(run_id), "top_k": SEARCH_TOP_K, "notes": notes},
            )
            sync_session.commit()

    log.info("Run ID: %s | top_k: %d | workers: %d", run_id, SEARCH_TOP_K, workers)
    log.info("Running searches (persisting each result immediately)...")
    shutdown = shutdown_requested or [False]
    total = len(rows)
    start_time = time.perf_counter()
    processed = 0
    timing_samples: list[dict[str, float]] = []

    for batch in _chunks(rows, workers):
        if shutdown[0]:
            break
        start_idx = processed + 1
        batch_tasks = [
            _process_one_question(row, start_idx + i, total)
            for i, row in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*batch_tasks)

        for r in batch_results:
            _persist_result(sync_engine, str(run_id), r)
            if r.get("timings"):
                timing_samples.append(r["timings"])
            processed += 1

    total_elapsed = time.perf_counter() - start_time
    if shutdown[0]:
        log.info("Interrupted. %d results persisted.", processed)
    log.info(
        "Searches complete in %.1fs (%.1f q/s)",
        total_elapsed,
        processed / total_elapsed if total_elapsed > 0 else 0,
    )
    log.info("Loading results from DB for metrics...")
    full_results = _load_results_from_db_simple(sync_engine, str(run_id))
    return _compute_and_write_output(
        sync_engine,
        run_id,
        full_results,
        workers,
        notes,
        limit,
        documents,
        questions_per_document,
        timing_samples=timing_samples,
    )


def _compute_average_timings(
    timing_samples: list[dict[str, float]],
) -> dict[str, float]:
    """Compute average duration (seconds) per step across samples."""
    if not timing_samples:
        return {}
    all_keys = set()
    for t in timing_samples:
        all_keys.update(k for k in t.keys() if k != "total")
    avg: dict[str, float] = {}
    for k in sorted(all_keys):
        vals = [t[k] for t in timing_samples if k in t and isinstance(t[k], (int, float))]
        if vals:
            avg[k] = sum(vals) / len(vals)
    if avg:
        avg["total"] = sum(v for k, v in avg.items() if k != "total")
    return avg


def _compute_and_write_output(
    engine,
    run_id,
    full_results: list[dict],
    workers: int,
    notes: str | None,
    limit: int | None,
    documents: int | None,
    questions_per_document: int | None,
    timing_samples: list[dict[str, float]] | None = None,
) -> dict:
    """Compute metrics from full_results and write eval output files."""
    n = len(full_results)
    if n == 0:
        return {"error": "No results to compute metrics."}

    avg_timings = _compute_average_timings(timing_samples or [])

    chunk_hits = {k: 0 for k in K_VALUES}
    doc_hits = {k: 0 for k in K_VALUES}
    by_type: dict[str, dict] = {}

    for r in full_results:
        chunk_rank = r["chunk_rank"]
        doc_rank = r["doc_rank"]
        query_type = r["query_type"]
        for k in K_VALUES:
            if chunk_rank is not None and chunk_rank <= k:
                chunk_hits[k] += 1
            if doc_rank is not None and doc_rank <= k:
                doc_hits[k] += 1
        if query_type not in by_type:
            by_type[query_type] = {
                "n": 0,
                **{f"chunk@{k}": 0 for k in K_VALUES},
                **{f"doc@{k}": 0 for k in K_VALUES},
            }
        bt = by_type[query_type]
        bt["n"] += 1
        for k in K_VALUES:
            if chunk_rank is not None and chunk_rank <= k:
                bt[f"chunk@{k}"] += 1
            if doc_rank is not None and doc_rank <= k:
                bt[f"doc@{k}"] += 1

    chunk_metrics = {}
    doc_metrics = {}
    for k in K_VALUES:
        chunk_metrics[f"hit@{k}"] = chunk_hits[k]
        chunk_metrics[f"recall@{k}"] = chunk_hits[k] / n if n else 0
        chunk_metrics[f"precision@{k}"] = chunk_hits[k] / (k * n) if n and k else 0.0
        doc_metrics[f"hit@{k}"] = doc_hits[k]
        doc_metrics[f"recall@{k}"] = doc_hits[k] / n if n else 0
        doc_metrics[f"precision@{k}"] = doc_hits[k] / (k * n) if n and k else 0.0

    by_doc: dict[str, dict] = defaultdict(
        lambda: {
            "n": 0,
            **{f"chunk_hit@{k}": 0 for k in K_VALUES},
            **{f"doc_hit@{k}": 0 for k in K_VALUES},
        }
    )
    for r in full_results:
        doc_id = r["target_doc_id"]
        by_doc[doc_id]["n"] += 1
        for k in K_VALUES:
            if r["chunk_rank"] is not None and r["chunk_rank"] <= k:
                by_doc[doc_id][f"chunk_hit@{k}"] += 1
            if r["doc_rank"] is not None and r["doc_rank"] <= k:
                by_doc[doc_id][f"doc_hit@{k}"] += 1

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("eval_results") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing sanity-check output to %s/", out_dir)

    with Session(engine) as sync_session:
        n_docs, n_chunks = sync_session.execute(
            text(
                "SELECT (SELECT COUNT(*) FROM documents), (SELECT COUNT(*) FROM chunks)"
            )
        ).fetchone()

    config_lines = [
        "# Eval Configuration",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Timestamp**: {ts}",
        f"- **top_k**: {SEARCH_TOP_K}",
        f"- **workers**: {workers}",
        f"- **n_questions**: {n}",
        f"- **total_documents**: {n_docs}",
        f"- **total_chunks**: {n_chunks}",
    ]
    if limit is not None:
        config_lines.append(f"- **limit**: {limit}")
    if documents is not None:
        config_lines.append(f"- **documents**: {documents}")
    if questions_per_document is not None:
        config_lines.append(f"- **questions_per_document**: {questions_per_document}")
    if notes:
        config_lines.append(f"- **notes**: {notes}")
    (out_dir / "configs.md").write_text("\n".join(config_lines), encoding="utf-8")

    for i, r in enumerate(full_results[:3], 1):
        lines = [
            "# Example Result {}",
            "",
            "## Question",
            "",
            r["question"],
            "",
            "## Correct Answer",
            "",
            f"- **Document ID**: `{r['target_doc_id']}`",
            f"- **Chunk ID**: `{r['target_chunk_id']}`",
            "",
            "## Search Results",
            "",
            "### Document IDs (in order)",
            "",
        ]
        for j, did in enumerate(r["document_rank_order"], 1):
            mark = " ✓" if did == r["target_doc_id"] else ""
            lines.append(f"{j}. `{did}`{mark}")
        lines.extend(["", "### Chunk IDs (in order)", ""])
        for j, cid in enumerate(r["chunk_rank_order"], 1):
            mark = " ✓" if cid == r["target_chunk_id"] else ""
            lines.append(f"{j}. `{cid}`{mark}")
        (out_dir / f"example_result_{i}.md").write_text(
            "\n".join(lines).format(i), encoding="utf-8"
        )

    doc_ids = list(by_doc.keys())
    doc_id_to_filename: dict[str, str] = {}
    if doc_ids:
        with Session(engine) as sync_session:
            result = sync_session.execute(
                text(
                    "SELECT id::text, filename FROM documents WHERE id::text = ANY(:ids)"
                ),
                {"ids": doc_ids},
            )
            for row in result.fetchall():
                doc_id_to_filename[row[0]] = row[1]

    res_lines = [
        "# Search Eval Results",
        "",
        f"Run ID: `{run_id}` | Questions: {n} | top_k: {SEARCH_TOP_K}",
        "",
    ]

    if avg_timings:
        res_lines.extend(
            [
                "## Average Step Timings (seconds)",
                "",
                "| Step | Avg (s) | Avg (ms) |",
                "|------|---------|----------|",
            ]
        )
        for k in sorted(avg_timings.keys(), key=lambda x: (0 if x == "total" else 1, x)):
            v = avg_timings[k]
            res_lines.append(f"| {k} | {v:.3f} | {v * 1000:.0f} |")
        res_lines.extend(["", ""])

    res_lines.extend(
        [
            "## Overall Metrics",
            "",
            "### Chunk-level",
            "",
            "| k | hit@k | recall@k | precision@k |",
            "|---|-------|----------|--------------|",
        ]
    )
    for k in K_VALUES:
        c = chunk_metrics
        res_lines.append(
            f"| {k} | {c[f'hit@{k}']} | {c[f'recall@{k}']:.1%} | {c[f'precision@{k}']:.1%} |"
        )
    res_lines.extend(
        [
            "",
            "### Document-level",
            "",
            "| k | hit@k | recall@k | precision@k |",
            "|---|-------|----------|--------------|",
        ]
    )
    for k in K_VALUES:
        d = doc_metrics
        res_lines.append(
            f"| {k} | {d[f'hit@{k}']} | {d[f'recall@{k}']:.1%} | {d[f'precision@{k}']:.1%} |"
        )
    res_lines.extend(
        [
            "",
            "## Per-Document Metrics",
            "",
            "| Document | filename | n | chunk hit@1 | chunk hit@2 | chunk hit@4 | chunk hit@8 | doc hit@1 | doc hit@2 | doc hit@4 | doc hit@8 |",
            "|----------|----------|---|-------------|-------------|-------------|-------------|-----------|-----------|-----------|-----------|",
        ]
    )
    for doc_id in sorted(by_doc.keys()):
        bd = by_doc[doc_id]
        fn = doc_id_to_filename.get(doc_id, "—").replace("|", " ")
        row = [
            doc_id[:8] + "…",
            (fn[:20] + "…") if len(fn) > 20 else fn,
            str(bd["n"]),
        ]
        for k in K_VALUES:
            row.append(str(bd[f"chunk_hit@{k}"]))
        for k in K_VALUES:
            row.append(str(bd[f"doc_hit@{k}"]))
        res_lines.append("| " + " | ".join(row) + " |")
    res_lines.extend(["", "## By Query Type", ""])
    for qtype, bt in sorted(by_type.items()):
        n_q = bt["n"]
        res_lines.append(f"### {qtype} (n={n_q})")
        res_lines.append("")
        res_lines.append("| k | chunk recall@k | doc recall@k |")
        res_lines.append("|---|---------------|--------------|")
        for k in K_VALUES:
            c_r = bt[f"chunk@{k}"] / n_q if n_q else 0
            d_r = bt[f"doc@{k}"] / n_q if n_q else 0
            res_lines.append(f"| {k} | {c_r:.1%} | {d_r:.1%} |")
        res_lines.append("")

    (out_dir / "results.md").write_text("\n".join(res_lines), encoding="utf-8")

    return {
        "run_id": str(run_id),
        "n_questions": n,
        "chunk": chunk_metrics,
        "document": doc_metrics,
        "by_query_type": by_type,
        "output_dir": str(out_dir),
        "avg_timings": avg_timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run search evaluation over test questions (uses top_k=10, same as live UI)"
    )
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default=None,
        help="Resume an existing run; skips already-processed questions",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional notes for this eval run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of questions (applied after document sampling)",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=None,
        help="Limit to N documents (takes first N docs by id)",
    )
    parser.add_argument(
        "--questions-per-document",
        type=int,
        default=None,
        help="Max questions per document",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=BATCH_SIZE,
        help=f"Concurrent questions per batch (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    shutdown_requested = setup_signal_handler(
        "\nInterrupt received, finishing current batch and shutting down..."
    )

    log.info(
        "Starting search eval (top_k=%d, workers=%d, metrics: %s)",
        SEARCH_TOP_K,
        args.workers,
        list(K_VALUES),
    )
    if args.resume_run_id:
        log.info("Resume run ID: %s", args.resume_run_id)
    if args.documents is not None:
        log.info("Documents: %d", args.documents)
    if args.questions_per_document is not None:
        log.info("Questions per document: %d", args.questions_per_document)
    if args.limit is not None:
        log.info("Limit: %d questions", args.limit)
    if args.notes:
        log.info("Notes: %s", args.notes)

    result = asyncio.run(
        run_eval(
            notes=args.notes,
            limit=args.limit,
            documents=args.documents,
            questions_per_document=args.questions_per_document,
            workers=args.workers,
            shutdown_requested=shutdown_requested,
            resume_run_id=args.resume_run_id,
        )
    )

    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1

    n = result["n_questions"]
    print(f"\nSearch Eval Run: {result['run_id']}")
    print(f"Questions: {n} | top_k: {SEARCH_TOP_K} (live UI default)")
    if result.get("output_dir"):
        print(f"Output: {result['output_dir']}/")
    print("\n--- Chunk-level ---")
    c = result["chunk"]
    for k in K_VALUES:
        print(f"  hit@{k}: {c[f'hit@{k}']:3d}  ({c[f'recall@{k}']:.1%})")
    print("\n--- Document-level ---")
    d = result["document"]
    for k in K_VALUES:
        print(f"  hit@{k}: {d[f'hit@{k}']:3d}  ({d[f'recall@{k}']:.1%})")

    if result["by_query_type"]:
        print("\n--- By query type ---")
        for qtype, bt in sorted(result["by_query_type"].items()):
            n_q = bt["n"]
            parts = (
                [f"chunk@{k}={bt[f'chunk@{k}'] / n_q:.1%}" for k in K_VALUES]
                if n_q
                else ["n=0"]
            )
            print(f"  {qtype}: n={n_q}  {'  '.join(parts)}")

    if result.get("avg_timings"):
        print("\n--- Average step timings (ms) ---")
        for k in sorted(
            result["avg_timings"].keys(),
            key=lambda x: (0 if x == "total" else 1, x),
        ):
            v = result["avg_timings"][k]
            print(f"  {k}: {v * 1000:.0f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
