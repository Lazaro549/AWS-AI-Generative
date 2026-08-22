"""Tests for evaluation/metrics.py: pure metric functions and edge cases.

These tests do not touch the network or AWS in any way.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluation"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def metrics():
    module = _load("metrics", EVAL_DIR / "metrics.py")
    yield module
    sys.modules.pop("metrics", None)


# --- context_precision / context_recall -------------------------------------


def test_context_precision_all_relevant(metrics):
    assert metrics.context_precision(["a", "b"], ["a", "b"]) == 1.0


def test_context_precision_partial_overlap(metrics):
    assert metrics.context_precision(["a", "b"], ["a"]) == 0.5


def test_context_precision_empty_retrieved_is_undefined(metrics):
    assert metrics.context_precision([], ["a"]) is None


def test_context_precision_no_relevant_known_is_zero(metrics):
    assert metrics.context_precision(["a"], []) == 0.0


def test_context_recall_full_recall(metrics):
    assert metrics.context_recall(["a", "b", "c"], ["a", "b"]) == 1.0


def test_context_recall_partial_recall(metrics):
    assert metrics.context_recall(["a"], ["a", "b"]) == 0.5


def test_context_recall_empty_relevant_is_undefined(metrics):
    assert metrics.context_recall(["a"], []) is None


def test_context_recall_nothing_retrieved_is_zero(metrics):
    assert metrics.context_recall([], ["a"]) == 0.0


# --- token_f1 / answer_relevancy ---------------------------------------------


def test_token_f1_identical_text_is_one(metrics):
    text = "Amazon Bedrock is serverless"
    assert metrics.token_f1(text, text) == 1.0


def test_token_f1_no_overlap_is_zero(metrics):
    prediction = "completely unrelated text"
    reference = "Amazon Bedrock foundation models"
    assert metrics.token_f1(prediction, reference) == 0.0


def test_token_f1_empty_reference_is_undefined(metrics):
    assert metrics.token_f1("some answer", "") is None


def test_token_f1_empty_prediction_is_zero(metrics):
    assert metrics.token_f1("", "Amazon Bedrock foundation models") == 0.0


def test_token_f1_partial_overlap_between_zero_and_one(metrics):
    prediction = "Amazon Bedrock provides foundation models"
    reference = "Amazon Bedrock provides access to foundation models from Anthropic"
    score = metrics.token_f1(prediction, reference)
    assert score is not None
    assert 0.0 < score < 1.0


def test_answer_relevancy_uses_token_f1(metrics):
    answer = "Amazon Bedrock provides access to foundation models."
    expected = "Amazon Bedrock provides access to foundation models."
    assert metrics.answer_relevancy(answer, expected) == 1.0


# --- faithfulness -------------------------------------------------------------


def test_faithfulness_fully_grounded_answer(metrics):
    context = "Amazon Bedrock provides access to foundation models from Anthropic and Meta."
    answer = "Amazon Bedrock provides access to foundation models."
    assert metrics.faithfulness(answer, context) == 1.0


def test_faithfulness_hallucinated_sentence_is_penalized(metrics):
    context = "Amazon Bedrock provides access to foundation models."
    answer = (
        "Amazon Bedrock provides access to foundation models. "
        "It was launched in 1995 by a team in Antarctica."
    )
    score = metrics.faithfulness(answer, context)
    assert score is not None
    assert 0.0 <= score < 1.0


def test_faithfulness_empty_answer_is_undefined(metrics):
    assert metrics.faithfulness("", "some context") is None


def test_faithfulness_empty_context_is_zero(metrics):
    assert metrics.faithfulness("A real claim goes here.", "") == 0.0


def test_faithfulness_respects_custom_threshold(metrics):
    context = "Amazon Bedrock supports chatbots and summarization."
    # The answer sentence has 5 content tokens; only "bedrock" overlaps
    # with the context, so overlap ratio = 1/5 = 0.2.
    answer = "Bedrock also handles translation tasks well."
    assert metrics.faithfulness(answer, context, threshold=0.5) == 0.0
    assert metrics.faithfulness(answer, context, threshold=0.1) == 1.0


# --- aggregate_scores / latency_stats ------------------------------------------


def test_aggregate_scores_mixed_values(metrics):
    result = metrics.aggregate_scores([1.0, 0.5, None, 0.0])
    assert result["count_scored"] == 3
    assert result["count_undefined"] == 1
    assert result["mean"] == pytest.approx(0.5)


def test_aggregate_scores_all_undefined(metrics):
    result = metrics.aggregate_scores([None, None])
    assert result["mean"] is None
    assert result["count_scored"] == 0
    assert result["count_undefined"] == 2


def test_aggregate_scores_empty_list(metrics):
    result = metrics.aggregate_scores([])
    assert result["mean"] is None
    assert result["count_scored"] == 0
    assert result["count_undefined"] == 0


def test_latency_stats_empty(metrics):
    result = metrics.latency_stats([])
    assert result["count"] == 0
    assert result["mean_seconds"] is None


def test_latency_stats_basic(metrics):
    result = metrics.latency_stats([0.1, 0.2, 0.3])
    assert result["count"] == 3
    assert result["mean_seconds"] == pytest.approx(0.2)
    assert result["min_seconds"] == pytest.approx(0.1)
    assert result["max_seconds"] == pytest.approx(0.3)


# --- estimate_cost --------------------------------------------------------------


def test_estimate_cost_requires_usage_and_pricing(metrics):
    assert metrics.estimate_cost(None, None, None, None) is None
    assert metrics.estimate_cost(100, 50, None, None) is None
    assert metrics.estimate_cost(None, None, 0.003, 0.015) is None


def test_estimate_cost_computes_when_available(metrics):
    cost = metrics.estimate_cost(1000, 500, 0.003, 0.015)
    assert cost == pytest.approx(0.003 + 0.0075)


# --- report assembly ------------------------------------------------------------


def test_build_report_shape(metrics, tmp_path):
    report = metrics.build_report(
        evaluation_type="retrieval",
        dataset_path=tmp_path / "dataset.json",
        per_question_results=[{"id": "q1"}],
        aggregate_metrics={
            "context_precision": {"mean": 1.0, "count_scored": 1, "count_undefined": 0}
        },
        latency={"retrieval_latency": {"count": 1, "mean_seconds": 0.01}},
        failures=[],
        configuration={"model_id": "test"},
    )
    assert report["evaluation_type"] == "retrieval"
    assert report["num_questions_evaluated"] == 1
    assert "timestamp_utc" in report


def test_save_report_writes_json_file(metrics, tmp_path):
    report = {"evaluation_type": "retrieval", "num_questions_evaluated": 0}
    path = metrics.save_report(report, tmp_path, prefix="retrieval_report")
    assert path.exists()
    assert path.name.startswith("retrieval_report_")
    assert path.suffix == ".json"
