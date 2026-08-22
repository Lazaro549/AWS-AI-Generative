"""Metric calculations for evaluating the repository's RAG implementation.

Every function here is pure (no I/O, no AWS calls) and independently
testable -- see tests/test_evaluation_metrics.py. Metrics that cannot be
computed from the given inputs return ``None`` instead of a fabricated
score (e.g. ``0.0`` or ``1.0``), so that missing evaluation data is never
silently reported as a perfect -- or a failing -- result.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, hand-picked stopword list, kept in this file rather than pulled
# from a new dependency (this repository's runtime dependencies are only
# boto3 and python-dotenv -- see pyproject.toml).
_STOPWORDS = frozenset(
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "with", "by", "as", "and", "or",
        "but", "if", "it", "its", "this", "that", "these", "those", "from",
        "into", "such", "can", "does", "do", "did", "has", "have", "had",
        "not", "no", "so", "than", "then", "also", "which", "who", "what",
        "when", "where", "how", "why", "i", "you", "he", "she", "we", "they",
        "them", "his", "her", "their", "our", "your",
    ]
)


def content_tokens(text: str) -> list[str]:
    """Lowercase, alphanumeric tokens from ``text`` with stopwords removed.

    This is the basis for every lexical-overlap metric below (faithfulness,
    answer relevancy). It is a heuristic, not semantic understanding: it
    catches word-level grounding but not paraphrases or synonyms -- see
    evaluation/README.md, "Limitations".
    """
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences on '.', '!' or '?' plus whitespace."""
    if not text or not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in (part.strip() for part in parts) if p]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def context_precision(retrieved: Sequence[str], relevant: Sequence[str]) -> float | None:
    """Fraction of retrieved documents that are actually relevant.

    Returns ``None`` (undefined) when nothing was retrieved, instead of
    treating an empty retrieval as a perfect or a zero score.
    """
    if not retrieved:
        return None
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in retrieved if doc_id in relevant_set)
    return hits / len(retrieved)


def context_recall(retrieved: Sequence[str], relevant: Sequence[str]) -> float | None:
    """Fraction of the known-relevant documents that were retrieved.

    Returns ``None`` (undefined) when the question has no ground-truth
    relevant documents recorded, since recall cannot be computed without
    something to recall against.
    """
    if not relevant:
        return None
    retrieved_set = set(retrieved)
    hits = sum(1 for doc_id in relevant if doc_id in retrieved_set)
    return hits / len(relevant)


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

#: A sentence in the generated answer is considered "supported" by the
#: retrieved context when at least this fraction of its content tokens
#: also appear in the context.
DEFAULT_FAITHFULNESS_THRESHOLD = 0.5


def faithfulness(
    answer: str,
    context: str,
    threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD,
) -> float | None:
    """Deterministic, lexical-overlap approximation of faithfulness.

    The answer is split into sentences. Each sentence that has at least
    one content token is "checkable"; a checkable sentence is "supported"
    when at least ``threshold`` of its content tokens also occur in the
    retrieved context. The score is the fraction of checkable sentences
    that are supported, so unsupported claims and hallucinated sentences
    pull the score down.

    This is a word-overlap heuristic, not entailment -- see
    evaluation/README.md, "Limitations", for what that means in practice
    and for the optional LLM-judge alternative (``--judge llm`` in
    evaluate_generation.py).

    Returns ``None`` only when there is no answer text at all to evaluate
    (e.g. generation failed) -- as opposed to a real ``0.0``, which means
    an answer exists but nothing in it is grounded in the context.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return None

    context_token_set = set(content_tokens(context))
    if not context_token_set:
        # A real answer exists but there is no context to ground it in:
        # a genuine faithfulness failure, not missing evaluation data.
        return 0.0

    checkable = 0
    supported = 0
    for sentence in sentences:
        sentence_tokens = set(content_tokens(sentence))
        if not sentence_tokens:
            continue  # nothing substantive to check (e.g. "Yes.")
        checkable += 1
        overlap = len(sentence_tokens & context_token_set) / len(sentence_tokens)
        if overlap >= threshold:
            supported += 1

    if checkable == 0:
        return None
    return supported / checkable


def token_f1(prediction: str, reference: str) -> float | None:
    """SQuAD-style token-overlap F1 between two texts.

    Returns ``None`` when ``reference`` has no content tokens, since there
    is nothing meaningful to compare against. Returns a real ``0.0`` (not
    ``None``) when ``prediction`` is empty but a reference exists: a blank
    answer genuinely fails to match real reference content.
    """
    reference_tokens = content_tokens(reference)
    if not reference_tokens:
        return None

    prediction_tokens = content_tokens(prediction)
    if not prediction_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_relevancy(answer: str, expected_answer: str) -> float | None:
    """Deterministic proxy for answer relevancy.

    True answer relevancy asks whether an answer addresses the *question*.
    Without embeddings or an LLM available, this approximates it with
    token-F1 overlap against the dataset's ``expected_answer`` -- a
    reference answer that, by construction, does address the question.
    See evaluation/README.md, "Limitations", for the reasoning and for
    the more direct LLM-judge alternative.
    """
    return token_f1(answer, expected_answer)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_scores(values: Sequence[float | None]) -> dict[str, Any]:
    """Summarize a list of optional scores without hiding missing data."""
    scored = [v for v in values if v is not None]
    return {
        "mean": (sum(scored) / len(scored)) if scored else None,
        "min": min(scored) if scored else None,
        "max": max(scored) if scored else None,
        "count_scored": len(scored),
        "count_undefined": len(values) - len(scored),
    }


def latency_stats(samples: Sequence[float]) -> dict[str, Any]:
    """Basic latency statistics (mean/min/max/p95), in seconds."""
    if not samples:
        return {
            "count": 0,
            "mean_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "p95_seconds": None,
        }
    ordered = sorted(samples)
    n = len(ordered)
    p95_index = min(n - 1, math.ceil(0.95 * n) - 1)
    return {
        "count": n,
        "mean_seconds": sum(ordered) / n,
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "p95_seconds": ordered[p95_index],
    }


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    input_price_per_1k: float | None,
    output_price_per_1k: float | None,
) -> float | None:
    """Estimate USD cost from token usage and a price rate.

    This repository does not embed Amazon Bedrock pricing anywhere --
    prices vary by model and region and change over time -- so this
    function never assumes a default rate. It returns ``None``
    ("unavailable") unless both real token usage *and* an explicit price
    were supplied by the caller. See evaluation/README.md, "Cost/query".
    """
    if input_tokens is None or output_tokens is None:
        return None
    if input_price_per_1k is None or output_price_per_1k is None:
        return None
    return (input_tokens / 1000) * input_price_per_1k + (
        output_tokens / 1000
    ) * output_price_per_1k


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_report(
    *,
    evaluation_type: str,
    dataset_path: Path,
    per_question_results: list[dict[str, Any]],
    aggregate_metrics: dict[str, Any],
    latency: dict[str, Any],
    failures: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the standard machine-readable evaluation report."""
    return {
        "evaluation_type": evaluation_type,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(dataset_path),
        "num_questions_evaluated": len(per_question_results),
        "aggregate_metrics": aggregate_metrics,
        "latency": latency,
        "per_question_results": per_question_results,
        "failures": failures,
        "configuration": configuration,
    }


def save_report(report: dict[str, Any], reports_dir: Path, prefix: str) -> Path:
    """Write ``report`` as indented JSON under ``reports_dir``; return its path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = reports_dir / f"{prefix}_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_summary(report: dict[str, Any], title: str) -> None:
    """Print a concise, human-readable summary of ``report`` to the terminal."""
    print(f"\n=== {title} ===")
    print(f"Dataset: {report['dataset']}")
    print(f"Timestamp (UTC): {report['timestamp_utc']}")
    print(f"Questions evaluated: {report['num_questions_evaluated']}")

    print("\nAggregate metrics:")
    for name, stats in report["aggregate_metrics"].items():
        print(
            f"  {name:<18} mean={_fmt(stats['mean'])}"
            f"  (scored {stats['count_scored']}, undefined {stats['count_undefined']})"
        )

    print("\nLatency:")
    for name, stats in report["latency"].items():
        print(
            f"  {name:<18} mean={_fmt(stats.get('mean_seconds'), 4)}s"
            f"  p95={_fmt(stats.get('p95_seconds'), 4)}s"
            f"  (n={stats.get('count', 0)})"
        )

    failures = report["failures"]
    print(f"\nFailures: {len(failures)}")
    for failure in failures[:5]:
        print(f"  - {failure.get('id')} [{failure.get('stage')}]: {failure.get('error')}")
    if len(failures) > 5:
        print(f"  ... and {len(failures) - 5} more (see the JSON report)")
