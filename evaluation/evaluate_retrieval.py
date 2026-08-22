"""Evaluate the retrieval stage of the repository's RAG example
(examples/rag/ingest.py) independently from generation.

Retrieval in this repository has no vector search or chunking: every call
to ``load_documents()`` reads and concatenates every file under
``examples/rag/data/`` (see examples/rag/ingest.py). This script measures
that behaviour as it exists today -- it does not add retrieval logic that
is not in the repository. See evaluation/README.md, "Limitations", for
what that means for Context Precision/Recall right now.

Run from the repository root:

    python evaluation/evaluate_retrieval.py
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / "examples" / "rag"
DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset" / "rag_questions.json"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def load_module_from_path(name: str, path: Path):
    """Load a module from a file path, exactly like tests/test_rag.py does.

    The example scripts under examples/rag/ are written to be run
    directly (bare ``from ingest import ...`` style imports), not
    imported as an installed package, so evaluation loads them the same
    way the existing test suite already does.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{name}' from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_ingest_module(rag_dir: Path = RAG_DIR):
    """Load examples/rag/ingest.py."""
    return load_module_from_path("ingest", rag_dir / "ingest.py")


def list_data_files(data_dir: Path, repo_root: Path = REPO_ROOT) -> list[str]:
    """List the files ``ingest.load_documents()`` would load, as
    repo-root-relative paths.

    ``load_documents()`` returns a single concatenated string with no
    per-file identifiers. This mirrors its exact selection rule (files
    directly under ``data_dir``, sorted, non-recursive -- see
    examples/rag/ingest.py) purely to give the evaluation metrics
    something to compare against; it does not change what gets loaded.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        return []
    return [
        p.relative_to(repo_root).as_posix() for p in sorted(data_dir.iterdir()) if p.is_file()
    ]


def retrieve_with_timing(ingest_module: Any) -> tuple[str, list[str], float]:
    """Run the real retrieval step once and time it.

    Returns ``(context_text, retrieved_document_ids, latency_seconds)``.
    """
    start = time.perf_counter()
    context = ingest_module.load_documents()
    elapsed = time.perf_counter() - start
    retrieved_ids = list_data_files(ingest_module.DATA_DIR)
    return context, retrieved_ids, elapsed


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the evaluation dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Evaluation dataset must be a non-empty JSON array: {dataset_path}")
    return data


def evaluate(
    dataset: list[dict[str, Any]], dataset_path: Path = DEFAULT_DATASET
) -> dict[str, Any]:
    """Run retrieval evaluation over every question in ``dataset``."""
    ingest_module = load_ingest_module()

    per_question: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    precision_values: list[float | None] = []
    recall_values: list[float | None] = []
    latencies: list[float] = []

    for question in dataset:
        qid = question.get("id", "<missing-id>")
        try:
            context, retrieved_ids, elapsed = retrieve_with_timing(ingest_module)
            relevant_ids = question.get("relevant_documents", [])

            precision = metrics.context_precision(retrieved_ids, relevant_ids)
            recall = metrics.context_recall(retrieved_ids, relevant_ids)

            precision_values.append(precision)
            recall_values.append(recall)
            latencies.append(elapsed)

            per_question.append(
                {
                    "id": qid,
                    "question": question.get("question"),
                    "retrieved_documents": retrieved_ids,
                    "relevant_documents": relevant_ids,
                    "context_precision": precision,
                    "context_recall": recall,
                    "retrieval_latency_seconds": elapsed,
                    "context_length_chars": len(context),
                }
            )
        except Exception as exc:  # noqa: BLE001 - one bad question must not stop the run
            failures.append(
                {
                    "id": qid,
                    "stage": "retrieval",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    aggregate_metrics = {
        "context_precision": metrics.aggregate_scores(precision_values),
        "context_recall": metrics.aggregate_scores(recall_values),
    }
    latency = {"retrieval_latency": metrics.latency_stats(latencies)}

    configuration = {
        "retrieval_source": "examples/rag/ingest.py:load_documents",
        "data_dir": ingest_module.DATA_DIR.relative_to(REPO_ROOT).as_posix(),
        "requires_aws_credentials": False,
    }

    return metrics.build_report(
        evaluation_type="retrieval",
        dataset_path=dataset_path,
        per_question_results=per_question,
        aggregate_metrics=aggregate_metrics,
        latency=latency,
        failures=failures,
        configuration=configuration,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality and latency.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    report = evaluate(dataset, dataset_path=args.dataset)
    report_path = metrics.save_report(report, args.reports_dir, prefix="retrieval_report")

    metrics.print_summary(report, title="RAG Retrieval Evaluation")
    print(f"\nReport saved to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
