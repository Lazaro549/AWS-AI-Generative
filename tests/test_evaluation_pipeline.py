"""Integration-style tests for the evaluation/ scripts.

Retrieval evaluation needs no AWS access (it only reads local files), so
it is tested directly against the real repository data. Generation
evaluation is tested with a mocked Bedrock client -- exactly like
tests/test_rag.py's ``rag_query`` fixture -- so this suite never requires
live AWS credentials.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "evaluation"
DATASET_PATH = EVAL_DIR / "dataset" / "rag_questions.json"

_MODULE_NAMES = ("metrics", "evaluate_retrieval", "evaluate_generation", "ingest", "rag_query")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _clear_modules():
    for name in _MODULE_NAMES:
        sys.modules.pop(name, None)


@pytest.fixture
def retrieval_module():
    _load("metrics", EVAL_DIR / "metrics.py")
    module = _load("evaluate_retrieval", EVAL_DIR / "evaluate_retrieval.py")
    yield module
    _clear_modules()


class FakeBody:
    """Mimics botocore's StreamingBody: a one-shot .read()."""

    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()


class FakeBedrockClient:
    def __init__(self, answer_text="Amazon Bedrock provides access to foundation models."):
        self.calls = []
        self._answer_text = answer_text

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "body": FakeBody(
                {
                    "content": [{"text": self._answer_text}],
                    "usage": {"input_tokens": 42, "output_tokens": 7},
                }
            )
        }


@pytest.fixture
def generation_module(monkeypatch):
    fake_client = FakeBedrockClient()
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: fake_client)

    _load("metrics", EVAL_DIR / "metrics.py")
    _load("evaluate_retrieval", EVAL_DIR / "evaluate_retrieval.py")
    module = _load("evaluate_generation", EVAL_DIR / "evaluate_generation.py")

    yield module, fake_client

    _clear_modules()


# --- dataset sanity checks ----------------------------------------------------


def test_dataset_file_is_valid_json_with_expected_shape():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 5
    for entry in data:
        assert set(entry) >= {"id", "question", "expected_answer", "relevant_documents"}
        assert isinstance(entry["relevant_documents"], list)
        assert entry["relevant_documents"], f"{entry['id']} has no relevant_documents"


# --- retrieval evaluation (real data, no AWS) ----------------------------------


def test_retrieval_evaluation_runs_against_real_data(retrieval_module):
    dataset = retrieval_module.load_dataset(DATASET_PATH)
    report = retrieval_module.evaluate(dataset, dataset_path=DATASET_PATH)

    assert report["evaluation_type"] == "retrieval"
    assert report["num_questions_evaluated"] == len(dataset)
    assert report["failures"] == []
    assert report["aggregate_metrics"]["context_precision"]["count_scored"] == len(dataset)
    # The current retrieval mechanism always returns every file in data/,
    # and every dataset question lists that same file as relevant.
    assert report["aggregate_metrics"]["context_precision"]["mean"] == 1.0
    assert report["aggregate_metrics"]["context_recall"]["mean"] == 1.0
    assert report["latency"]["retrieval_latency"]["count"] == len(dataset)


def test_retrieval_evaluation_handles_missing_dataset_file(retrieval_module, tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        retrieval_module.load_dataset(missing)


# --- generation evaluation (mocked Bedrock) ------------------------------------


def test_generation_evaluation_with_mocked_bedrock(generation_module, monkeypatch):
    monkeypatch.delenv("BEDROCK_INPUT_PRICE_PER_1K_USD", raising=False)
    monkeypatch.delenv("BEDROCK_OUTPUT_PRICE_PER_1K_USD", raising=False)

    module, fake_client = generation_module
    dataset = module.load_dataset(DATASET_PATH)[:3]  # keep the mocked run fast

    report = module.evaluate(dataset, dataset_path=DATASET_PATH, judge="deterministic")

    assert report["evaluation_type"] == "generation"
    assert report["failures"] == []
    assert report["num_questions_evaluated"] == 3
    assert len(fake_client.calls) == 3

    for result in report["per_question_results"]:
        assert result["generated_answer"]
        assert result["input_tokens"] == 42
        assert result["output_tokens"] == 7
        assert result["end_to_end_latency_seconds"] >= 0

    # No pricing was configured via environment variables, so cost must be
    # reported as unavailable rather than a fabricated number.
    assert report["cost_per_query"]["status"] == "unavailable"
    assert report["cost_per_query"]["mean_usd"] is None


def test_generation_evaluation_computes_cost_when_pricing_is_configured(
    generation_module, monkeypatch
):
    monkeypatch.setenv("BEDROCK_INPUT_PRICE_PER_1K_USD", "0.003")
    monkeypatch.setenv("BEDROCK_OUTPUT_PRICE_PER_1K_USD", "0.015")

    module, _fake_client = generation_module
    dataset = module.load_dataset(DATASET_PATH)[:1]

    report = module.evaluate(dataset, dataset_path=DATASET_PATH)

    assert report["cost_per_query"]["status"] == "estimated"
    assert report["cost_per_query"]["mean_usd"] is not None
    assert report["per_question_results"][0]["cost_usd"] is not None


def test_generation_evaluation_reports_failures_without_crashing(monkeypatch):
    class BrokenClient:
        def invoke_model(self, **kwargs):
            raise RuntimeError("simulated Bedrock outage")

    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: BrokenClient())

    _load("metrics", EVAL_DIR / "metrics.py")
    _load("evaluate_retrieval", EVAL_DIR / "evaluate_retrieval.py")
    module = _load("evaluate_generation", EVAL_DIR / "evaluate_generation.py")
    try:
        dataset = module.load_dataset(DATASET_PATH)[:2]
        report = module.evaluate(dataset, dataset_path=DATASET_PATH)

        assert report["num_questions_evaluated"] == 0
        assert len(report["failures"]) == 2
        assert all(f["stage"] == "generation" for f in report["failures"])
    finally:
        _clear_modules()


def test_generation_evaluation_llm_judge_parses_response(monkeypatch):
    judge_payload = {"faithfulness": 0.9, "answer_relevancy": 0.8, "rationale": "Looks accurate."}

    class JudgeAwareClient:
        def __init__(self):
            self.call_count = 0

        def invoke_model(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                # First call: the generation call itself.
                payload = {
                    "content": [{"text": "Amazon Bedrock provides foundation models."}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            else:
                # Second call: the LLM judge.
                payload = {"content": [{"text": json.dumps(judge_payload)}]}
            return {"body": FakeBody(payload)}

    fake_client = JudgeAwareClient()
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: fake_client)

    _load("metrics", EVAL_DIR / "metrics.py")
    _load("evaluate_retrieval", EVAL_DIR / "evaluate_retrieval.py")
    module = _load("evaluate_generation", EVAL_DIR / "evaluate_generation.py")
    try:
        dataset = module.load_dataset(DATASET_PATH)[:1]
        report = module.evaluate(dataset, dataset_path=DATASET_PATH, judge="llm")

        assert report["failures"] == []
        result = report["per_question_results"][0]
        assert result["faithfulness"] == pytest.approx(0.9)
        assert result["answer_relevancy"] == pytest.approx(0.8)
        assert result["rationale"] == "Looks accurate."
    finally:
        _clear_modules()
