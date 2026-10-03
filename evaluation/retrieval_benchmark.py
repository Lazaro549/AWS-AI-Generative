"""Offline retrieval benchmark: chunk size × top-k vs. context precision/recall/latency.

Runs the real chunking → FAISS → top-k retrieval pipeline against the
18-question benchmark dataset under three configurations, using the
deterministic MockEmbeddingClient so no AWS credentials are required.

The MockEmbeddingClient produces hash-based, non-semantic vectors.
Retrieval quality therefore reflects structural pipeline mechanics
(chunk granularity, top-k coverage, source deduplication) rather than
semantic similarity. This is documented explicitly in every report this
script produces.

Run from the repository root:
    python evaluation/retrieval_benchmark.py

Outputs:
    evaluation/reports/retrieval_benchmark_results.json
    evaluation/reports/RETRIEVAL_BENCHMARK_RESULTS.md
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
RAG_DIR = REPO_ROOT / "examples" / "rag"
DATASET_PATH = THIS_DIR / "dataset" / "rag_benchmark.json"
REPORTS_DIR = THIS_DIR / "reports"

sys.path.insert(0, str(RAG_DIR))
sys.path.insert(0, str(THIS_DIR))

import metrics  # noqa: E402
from embeddings import MockEmbeddingClient  # noqa: E402
from ingest import load_documents_structured  # noqa: E402
from retrieval import build_index, retrieve  # noqa: E402
from retrieval import load_index  # noqa: E402

CONFIGS: dict[str, dict[str, int]] = {
    "small":    {"chunk_size": 400,  "chunk_overlap": 60,  "top_k": 2},
    "balanced": {"chunk_size": 800,  "chunk_overlap": 120, "top_k": 3},
    "wide":     {"chunk_size": 1200, "chunk_overlap": 180, "top_k": 4},
}

EMBEDDING_DIMENSIONS = 256


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def run_config(
    config_name: str,
    config: dict[str, int],
    dataset: list[dict[str, Any]],
    embedding_client: MockEmbeddingClient,
    tmp_index_dir: Path,
) -> dict[str, Any]:
    """Build index for one config, evaluate all questions, return results."""
    build_start = time.perf_counter()
    index = build_index(
        embedding_client=embedding_client,
        data_dir=RAG_DIR / "data" / "documents",
        index_dir=tmp_index_dir,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    build_time = time.perf_counter() - build_start

    precisions: list[float] = []
    recalls: list[float] = []
    latencies: list[float] = []
    per_question: list[dict[str, Any]] = []

    for item in dataset:
        t0 = time.perf_counter()
        hits = retrieve(
            item["question"],
            index=index,
            embedding_client=embedding_client,
            top_k=config["top_k"],
        )
        elapsed = time.perf_counter() - t0

        retrieved_sources = list(dict.fromkeys(h.source for h in hits))
        relevant = item["relevant_documents"]

        cp = metrics.context_precision(retrieved_sources, relevant)
        cr = metrics.context_recall(retrieved_sources, relevant)

        if cp is not None:
            precisions.append(cp)
        if cr is not None:
            recalls.append(cr)
        latencies.append(elapsed)

        per_question.append({
            "id": item["id"],
            "retrieved_sources": retrieved_sources,
            "relevant_documents": relevant,
            "context_precision": cp,
            "context_recall": cr,
            "retrieval_latency_ms": round(elapsed * 1000, 3),
        })

    return {
        "config": config_name,
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "top_k": config["top_k"],
        "chunk_count": index.size,
        "index_build_time_ms": round(build_time * 1000, 1),
        "context_precision_mean": _mean(precisions),
        "context_recall_mean": _mean(recalls),
        "retrieval_latency_mean_ms": round((_mean(latencies) or 0) * 1000, 3),
        "retrieval_latency_p95_ms": round((_p95(latencies) or 0) * 1000, 3),
        "per_question": per_question,
    }


def build_markdown(report: dict[str, Any]) -> str:
    def fmt(v: Any, digits: int = 3) -> str:
        return "—" if v is None else f"{v:.{digits}f}"

    summary = report["summary"]
    lines = [
        "# Retrieval Benchmark Results",
        "",
        f"**Run ID:** `{report['run_id']}`  ",
        f"**Timestamp (UTC):** `{report['timestamp_utc']}`  ",
        f"**Questions:** `{report['num_questions']}`  ",
        f"**Corpus documents:** `{report['corpus_documents']}`  ",
        f"**Embedding:** `{report['embedding_note']}`  ",
        "",
        "## Method",
        "",
        "Offline benchmark using the repository's real chunking → FAISS → top-k retrieval "
        "pipeline. Embeddings use `MockEmbeddingClient` (deterministic, hash-based, "
        "non-semantic) so the experiment requires no AWS credentials and is fully "
        "reproducible. Retrieval quality reflects pipeline mechanics "
        "(chunk granularity, top-k coverage) under controlled conditions, not semantic similarity.",
        "",
        "## Results",
        "",
        "| Config | Chunk size | Overlap | Top-k | Chunks | "
        "Precision | Recall | Avg latency (ms) | P95 latency (ms) | Build (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['config']} "
            f"| {row['chunk_size']} "
            f"| {row['chunk_overlap']} "
            f"| {row['top_k']} "
            f"| {row['chunk_count']} "
            f"| {fmt(row['context_precision_mean'])} "
            f"| {fmt(row['context_recall_mean'])} "
            f"| {fmt(row['retrieval_latency_mean_ms'], 2)} "
            f"| {fmt(row['retrieval_latency_p95_ms'], 2)} "
            f"| {fmt(row['index_build_time_ms'], 1)} |"
        )

    # Finding
    recalls = [(r["config"], r["context_recall_mean"]) for r in summary if r["context_recall_mean"] is not None]
    best_config, best_recall = max(recalls, key=lambda x: x[1])
    baseline_recall = next(r["context_recall_mean"] for r in summary if r["config"] == "small")
    recall_delta = best_recall - baseline_recall if baseline_recall is not None else None

    baseline_lat = next(r["retrieval_latency_mean_ms"] for r in summary if r["config"] == "small")
    best_lat = next(r["retrieval_latency_mean_ms"] for r in summary if r["config"] == best_config)
    lat_delta_pct = ((best_lat - baseline_lat) / baseline_lat * 100) if baseline_lat else None

    lines += [
        "",
        "## Finding",
        "",
    ]
    if recall_delta is not None and abs(recall_delta) < 0.001:
        lines.append(
            "Context recall is identical across all three configurations. "
            "With non-semantic (hash-based) embeddings, chunk granularity and top-k "
            "do not change which document sources are retrieved — the mock vectors "
            "do not cluster by topic, so every configuration retrieves the same "
            "sources with similar scores. This is the expected, honest result for "
            "an offline benchmark without semantic embeddings."
        )
    else:
        sign = "+" if recall_delta >= 0 else ""
        lat_sign = "+" if lat_delta_pct >= 0 else ""
        lines.append(
            f"Best recall: `{best_config}` (recall={best_recall:.3f}). "
            f"Recall change vs baseline `small`: {sign}{recall_delta:.3f} "
            f"({sign}{recall_delta/baseline_recall*100:.1f}%). "
            f"Mean latency change: {lat_sign}{lat_delta_pct:.1f}%."
        )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python evaluation/retrieval_benchmark.py",
        "```",
        "",
        "No AWS credentials required. Results are deterministic across runs.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    dataset: list[dict[str, Any]] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not dataset:
        print("[ERROR] Empty dataset", file=sys.stderr)
        return 1

    embedding_client = MockEmbeddingClient(dimensions=EMBEDDING_DIMENSIONS)
    corpus_docs = list((RAG_DIR / "data" / "documents").glob("*.md"))

    summary: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []

    for config_name, config in CONFIGS.items():
        print(f"\n[{config_name}] chunk_size={config['chunk_size']} overlap={config['chunk_overlap']} top_k={config['top_k']}")
        tmp_dir = RAG_DIR / f".rag_bench_{config_name}"
        result = run_config(config_name, config, dataset, embedding_client, tmp_dir)
        summary.append({k: v for k, v in result.items() if k != "per_question"})
        all_results.append(result)
        print(
            f"  chunks={result['chunk_count']}  "
            f"precision={result['context_precision_mean']:.3f}  "
            f"recall={result['context_recall_mean']:.3f}  "
            f"avg_latency={result['retrieval_latency_mean_ms']:.2f}ms  "
            f"build={result['index_build_time_ms']:.1f}ms"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(dataset),
        "corpus_documents": len(corpus_docs),
        "embedding_note": f"MockEmbeddingClient (deterministic, hash-based, non-semantic, {EMBEDDING_DIMENSIONS}-dim)",
        "configs": CONFIGS,
        "summary": summary,
        "results": all_results,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "retrieval_benchmark_results.json"
    md_path = REPORTS_DIR / "RETRIEVAL_BENCHMARK_RESULTS.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")

    md_text = build_markdown(report)
    sys.stdout.buffer.write(("\n" + md_text + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
