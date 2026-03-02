"""Run search evaluation over generated test questions.

Loads questions from search_test_questions, runs each through enhanced_search
(using same top_k=10 as live UI search), records chunk/document ranks, and
reports accuracy for top1, top2, top4, top8.

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

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.cli.utils import setup_signal_handler
from app.config import settings
from app.database import async_session
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


def _chunks(items: list, size: int):
    """Yield successive chunks of size from items."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


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


async def run_eval(
    notes: str | None = None,
    limit: int | None = None,
    documents: int | None = None,
    questions_per_document: int | None = None,
    workers: int = BATCH_SIZE,
    shutdown_requested: list | None = None,
) -> dict:
    """Run evaluation and return summary metrics."""
    engine = create_engine(settings.database_url_sync)
    try:
        with Session(engine) as sync_session:
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
                qpd_str = f"{questions_per_document} q/doc" if questions_per_document else "all q/doc"
                log.info("Sampled: %s × %s → %d questions", doc_str, qpd_str, len(rows))
            elif limit:
                log.info("Limited to %d questions (--limit)", len(rows))
            else:
                log.info("Loaded %d questions", len(rows))

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
        log.info("Running searches (LLM query expansion + rerank per question)...")
        shutdown = shutdown_requested or [False]
        full_results: list[dict] = []
        chunk_hits = {k: 0 for k in K_VALUES}
        doc_hits = {k: 0 for k in K_VALUES}
        by_type: dict[str, dict] = {}
        total = len(rows)
        start_time = time.perf_counter()

        for batch in _chunks(rows, workers):
            if shutdown[0]:
                break
            start_idx = len(full_results) + 1
            batch_tasks = [
                _process_one_question(row, start_idx + i, total)
                for i, row in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*batch_tasks)

            for r in batch_results:
                full_results.append(r)

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

        total_elapsed = time.perf_counter() - start_time
        if shutdown[0]:
            log.info("Interrupted. Saving %d results collected so far...", len(full_results))
        log.info(
            "Searches complete in %.1fs (%.1f q/s)",
            total_elapsed,
            total / total_elapsed if total_elapsed > 0 else 0,
        )
        log.info("Writing %d results to search_eval_results...", len(full_results))

        # Batch insert results
        with Session(engine) as sync_session:
            for r in full_results:
                sync_session.execute(
                    text("""
                        INSERT INTO search_eval_results
                        (id, run_id, question_id, chunk_rank, doc_rank)
                        VALUES (:id, :run_id, :question_id, :chunk_rank, :doc_rank)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "run_id": str(run_id),
                        "question_id": str(r["question_id"]),
                        "chunk_rank": r["chunk_rank"],
                        "doc_rank": r["doc_rank"],
                    },
                )
            sync_session.commit()

        log.info("Done. Computing metrics...")
        n = len(full_results)
        chunk_metrics = {}
        doc_metrics = {}
        for k in K_VALUES:
            chunk_metrics[f"hit@{k}"] = chunk_hits[k]
            chunk_metrics[f"recall@{k}"] = chunk_hits[k] / n if n else 0
            # Precision@k: 1/k when relevant item in top-k, else 0
            chunk_metrics[f"precision@{k}"] = (
                chunk_hits[k] / (k * n) if n and k else 0.0
            )
            doc_metrics[f"hit@{k}"] = doc_hits[k]
            doc_metrics[f"recall@{k}"] = doc_hits[k] / n if n else 0
            doc_metrics[f"precision@{k}"] = (
                doc_hits[k] / (k * n) if n and k else 0.0
            )

        # Per-document metrics
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

        # Write eval_results/[timestamp]/*.md
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path("eval_results") / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Writing sanity-check output to %s/", out_dir)

        # configs.md
        config_lines = [
            "# Eval Configuration",
            "",
            f"- **Run ID**: `{run_id}`",
            f"- **Timestamp**: {ts}",
            f"- **top_k**: {SEARCH_TOP_K}",
            f"- **workers**: {workers}",
            f"- **n_questions**: {n}",
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

        # example_result_1.md, example_result_2.md, example_result_3.md
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
            lines.extend(
                [
                    "",
                    "### Chunk IDs (in order)",
                    "",
                ]
            )
            for j, cid in enumerate(r["chunk_rank_order"], 1):
                mark = " ✓" if cid == r["target_chunk_id"] else ""
                lines.append(f"{j}. `{cid}`{mark}")
            (out_dir / f"example_result_{i}.md").write_text(
                "\n".join(lines).format(i), encoding="utf-8"
            )

        # Fetch doc filenames for per-document table
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

        # results.md
        res_lines = [
            "# Search Eval Results",
            "",
            f"Run ID: `{run_id}` | Questions: {n} | top_k: {SEARCH_TOP_K}",
            "",
            "## Overall Metrics",
            "",
            "### Chunk-level",
            "",
            "| k | hit@k | recall@k | precision@k |",
            "|---|-------|----------|--------------|",
        ]
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

        res_lines.extend(
            [
                "",
                "## By Query Type",
                "",
            ]
        )
        for qtype, bt in sorted(by_type.items()):
            n_q = bt["n"]
            res_lines.append(f"### {qtype} (n={n_q})")
            res_lines.append("")
            res_lines.append(
                "| k | chunk recall@k | doc recall@k |"
            )
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
        }
    finally:
        engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run search evaluation over test questions (uses top_k=10, same as live UI)"
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

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
