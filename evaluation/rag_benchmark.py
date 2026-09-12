"""Live RAG benchmark: multiple documents, retrieval configurations, and Bedrock models.

This benchmark intentionally uses the repository's real RAG path:

    documents -> chunking -> Titan embeddings -> FAISS -> top-k context -> Bedrock

It runs every question against every retrieval configuration and generation model,
then writes JSON + Markdown results. It never fabricates a score when a live AWS
call fails.

Run locally with AWS credentials:
    python evaluation/rag_benchmark.py

Override the defaults with environment variables:
    BEDROCK_BENCHMARK_MODELS=global.anthropic.claude-haiku-4-5-20251001-v1:0,global.anthropic.claude-sonnet-4-5-20250929-v1:0
    RAG_BENCHMARK_CONFIGS=small,balanced,wide
    BEDROCK_INPUT_PRICE_PER_1K_USD=...
    BEDROCK_OUTPUT_PRICE_PER_1K_USD=...
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
RAG_DIR = REPO_ROOT / "examples" / "rag"
DATASET_PATH = THIS_DIR / "dataset" / "rag_benchmark.json"
REPORTS_DIR = THIS_DIR / "reports"

sys.path.insert(0, str(RAG_DIR))
sys.path.insert(0, str(THIS_DIR))

import context as context_module  # noqa: E402
import metrics  # noqa: E402
from embeddings import get_embedding_client  # noqa: E402
from retrieval import build_index, retrieve  # noqa: E402


CONFIGS: dict[str, dict[str, int]] = {
    "small": {"chunk_size": 400, "chunk_overlap": 60, "top_k": 2},
    "balanced": {"chunk_size": 800, "chunk_overlap": 120, "top_k": 3},
    "wide": {"chunk_size": 1200, "chunk_overlap": 180, "top_k": 4},
}

DEFAULT_MODELS = [
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
]


def load_dataset() -> list[dict[str, Any]]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Benchmark dataset must be a non-empty JSON array")
    return data


def selected_configs() -> dict[str, dict[str, int]]:
    raw = os.getenv("RAG_BENCHMARK_CONFIGS", "small,balanced,wide")
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown benchmark configs: {unknown}")
    return {name: CONFIGS[name] for name in names}


def selected_models() -> list[str]:
    raw = os.getenv("BEDROCK_BENCHMARK_MODELS", ",".join(DEFAULT_MODELS))
    models = [x.strip() for x in raw.split(",") if x.strip()]
    if not models:
        raise ValueError("BEDROCK_BENCHMARK_MODELS must contain at least one model")
    return models


def invoke_model(client: Any, model_id: str, prompt: str) -> tuple[str, dict[str, Any], float]:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": int(os.getenv("BEDROCK_BENCHMARK_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("BEDROCK_BENCHMARK_TEMPERATURE", "0")),
        "messages": [{"role": "user", "content": prompt}],
    }
    start = time.perf_counter()
    response = client.invoke_model(modelId=model_id, body=json.dumps(body))
    elapsed = time.perf_counter() - start
    payload = json.loads(response["body"].read())
    answer = payload["content"][0]["text"]
    usage = payload.get("usage", {})
    return answer, usage, elapsed


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, int(len(values) * 0.95) - 1)]


def estimate_cost(usage: dict[str, Any]) -> float | None:
    input_price = os.getenv("BEDROCK_INPUT_PRICE_PER_1K_USD")
    output_price = os.getenv("BEDROCK_OUTPUT_PRICE_PER_1K_USD")
    if not input_price or not output_price:
        return None
    try:
        input_tokens = float(usage.get("input_tokens"))
        output_tokens = float(usage.get("output_tokens"))
        return (input_tokens / 1000) * float(input_price) + (output_tokens / 1000) * float(output_price)
    except (TypeError, ValueError):
        return None


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RAG Benchmark Results",
        "",
        f"**Run ID:** `{report['run_id']}`  ",
        f"**Timestamp (UTC):** `{report['timestamp_utc']}`  ",
        f"**Questions:** `{report['num_questions']}`  ",
        f"**Corpus documents:** `{report['corpus_documents']}`  ",
        "",
        "## Method",
        "",
        "Live Amazon Bedrock generation over the repository's multi-document RAG pipeline. "
        "Retrieval uses Titan Embeddings + FAISS; each configuration changes chunk size, overlap, and top-k. "
        "Models are evaluated on the exact same question set and retrieved context for a fair comparison.",
        "",
        "## Aggregate results",
        "",
        "| Retrieval config | Model | Retrieval precision | Retrieval recall | Faithfulness | Answer F1 | Avg E2E (s) | P95 E2E (s) | Avg cost/query | Success |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        def fmt(v: Any, digits: int = 3) -> str:
            return "—" if v is None else f"{v:.{digits}f}"
        cost = fmt(row["avg_cost_usd"], 6)
        lines.append(
            f"| {row['config']} | `{row['model']}` | {fmt(row['context_precision'])} | "
            f"{fmt(row['context_recall'])} | {fmt(row['faithfulness'])} | {fmt(row['answer_f1'])} | "
            f"{fmt(row['avg_e2e_seconds'])} | {fmt(row['p95_e2e_seconds'])} | ${cost} | "
            f"{row['success_rate']:.1%} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The benchmark intentionally publishes measured values only. Failed AWS requests are reported as failures and are excluded from metric means; they are never converted into zeroes or hidden.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python evaluation/rag_benchmark.py",
        "```",
        "",
        "For cost estimates, provide the current Bedrock input/output price variables before running. Prices are not hard-coded because they vary by model and service tier.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    dataset = load_dataset()
    configs = selected_configs()
    models = selected_models()
    region = os.getenv("AWS_REGION", "us-east-1")
    embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

    client = boto3.client("bedrock-runtime", region_name=region)
    embedding_client = get_embedding_client("bedrock", client=client, model_id=embedding_model)

    results: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for config_name, config in configs.items():
        index_dir = RAG_DIR / f".rag_benchmark_{config_name}"
        print(f"\n[{config_name}] building index: {config}")
        build_index(
            embedding_client=embedding_client,
            index_dir=index_dir,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            on_progress=print,
        )
        from retrieval import load_index
        index = load_index(index_dir)
        if index is None:
            raise RuntimeError(f"Could not reload benchmark index: {index_dir}")

        for model_id in models:
            print(f"[{config_name}] model={model_id}")
            precisions: list[float | None] = []
            recalls: list[float | None] = []
            faithfulness: list[float | None] = []
            answer_f1: list[float | None] = []
            e2e: list[float] = []
            costs: list[float] = []
            successes = 0
            model_results: list[dict[str, Any]] = []

            for item in dataset:
                q_start = time.perf_counter()
                try:
                    hits = retrieve(item["question"], index=index, embedding_client=embedding_client, top_k=config["top_k"])
                    retrieved_docs = list(dict.fromkeys(hit.source for hit in hits))
                    context = context_module.assemble_context(hits)
                    prompt = context_module.build_prompt(item["question"], context)
                    answer, usage, generation_latency = invoke_model(client, model_id, prompt)
                    total_latency = time.perf_counter() - q_start

                    cp = metrics.context_precision(retrieved_docs, item["relevant_documents"])
                    cr = metrics.context_recall(retrieved_docs, item["relevant_documents"])
                    f = metrics.faithfulness(answer, context)
                    af1 = metrics.answer_relevancy(answer, item["expected_answer"])
                    cost = estimate_cost(usage)
                    if cost is not None:
                        costs.append(cost)
                    if cp is not None: precisions.append(cp)
                    if cr is not None: recalls.append(cr)
                    if f is not None: faithfulness.append(f)
                    if af1 is not None: answer_f1.append(af1)
                    e2e.append(total_latency)
                    successes += 1
                    model_results.append({
                        "id": item["id"], "question": item["question"], "retrieved_documents": retrieved_docs,
                        "retrieval_scores": [round(h.score, 4) for h in hits], "answer": answer,
                        "usage": usage, "generation_latency_seconds": generation_latency,
                        "e2e_latency_seconds": total_latency, "context_precision": cp,
                        "context_recall": cr, "faithfulness": f, "answer_f1": af1, "cost_usd": cost,
                    })
                except Exception as exc:
                    total_latency = time.perf_counter() - q_start
                    model_results.append({"id": item["id"], "question": item["question"], "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "e2e_latency_seconds": total_latency})
                    print(f"  [FAIL] {item['id']}: {type(exc).__name__}: {exc}")

            summary.append({
                "config": config_name, "model": model_id,
                "context_precision": mean(precisions), "context_recall": mean(recalls),
                "faithfulness": mean(faithfulness), "answer_f1": mean(answer_f1),
                "avg_e2e_seconds": mean(e2e), "p95_e2e_seconds": p95(e2e),
                "avg_cost_usd": mean(costs), "success_rate": successes / len(dataset),
            })
            results.append({"config": config_name, "model": model_id, "results": model_results})

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(dataset),
        "corpus_documents": len(list((RAG_DIR / "data" / "documents").glob("*.md"))),
        "region": region,
        "embedding_model": embedding_model,
        "configs": configs,
        "models": models,
        "summary": summary,
        "results": results,
        "status": "complete",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "rag_benchmark_latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / "RAG_BENCHMARK_RESULTS.md").write_text(build_markdown(report), encoding="utf-8")
    print("\n" + build_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
