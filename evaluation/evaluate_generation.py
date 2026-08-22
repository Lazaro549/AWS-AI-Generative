"""Evaluate the generation stage of the repository's RAG example
(examples/rag/query.py): faithfulness, answer relevancy, end-to-end
latency and, when available, token usage / cost.

Producing real answers requires a live call to Amazon Bedrock through
examples/rag/query.py's existing client and configuration -- this script
does not introduce a different model provider. When AWS credentials are
not available, each question fails independently (see the report's
"failures" list) instead of crashing the whole run.

Run from the repository root:

    python evaluation/evaluate_generation.py
    python evaluation/evaluate_generation.py --judge llm
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import metrics
from evaluate_retrieval import (
    DEFAULT_DATASET,
    DEFAULT_REPORTS_DIR,
    RAG_DIR,
    load_dataset,
    load_ingest_module,
    load_module_from_path,
    retrieve_with_timing,
)


def load_query_module(rag_dir: Path = RAG_DIR):
    """Load examples/rag/query.py.

    ``query.py`` contains a bare ``from ingest import load_documents``, so
    ``ingest`` must already be registered in ``sys.modules`` before this
    runs -- ``load_ingest_module`` (called by ``evaluate`` below) does
    that, exactly like tests/test_rag.py's ``rag_query`` fixture.
    """
    return load_module_from_path("rag_query", rag_dir / "query.py")


def _instrument_client_for_usage(client: Any) -> dict[str, Any]:
    """Wrap a bedrock-runtime client's ``invoke_model`` so the response's
    token usage can be captured for cost estimation, without changing the
    response that ``query.ask()`` itself parses.

    This wraps the client *instance* held by the dynamically loaded copy
    of query.py used for this evaluation run only -- it does not modify
    examples/rag/query.py on disk.
    """
    captured: dict[str, Any] = {"usage": None}
    original_invoke_model = client.invoke_model

    def instrumented_invoke_model(*args: Any, **kwargs: Any) -> Any:
        response = original_invoke_model(*args, **kwargs)
        try:
            raw_body = response["body"].read()
            parsed = json.loads(raw_body)
            captured["usage"] = parsed.get("usage")
            response["body"] = io.BytesIO(raw_body)  # let ask() read it again
        except Exception as exc:  # noqa: BLE001 - telemetry must never break generation
            print(f"[WARN] Could not capture Bedrock token usage: {exc}", file=sys.stderr)
        return response

    client.invoke_model = instrumented_invoke_model
    return captured


def _float_env(name: str) -> float | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


_JUDGE_PROMPT_TEMPLATE = """\
You are an evaluation judge for a Retrieval-Augmented Generation (RAG) system.

Given a question, the context that was retrieved for it, and the answer that
was generated, score the answer on two dimensions:

1. faithfulness: does every claim in the answer follow from the given
   context, with no unsupported additions or hallucinations? 1.0 = fully
   supported, 0.0 = not supported at all.
2. answer_relevancy: does the answer directly and completely address the
   question asked? 1.0 = fully relevant, 0.0 = does not address the
   question at all.

Question:
{question}

Context:
{context}

Answer:
{answer}

Respond with ONLY a JSON object in this exact format, with no extra text:
{{"faithfulness": <float 0.0-1.0>, "answer_relevancy": <float 0.0-1.0>,
"rationale": "<one short sentence>"}}"""


def _llm_judge(question: str, answer: str, context: str, query_module: Any) -> dict[str, Any]:
    """Score faithfulness and answer relevancy using the same Bedrock
    client already configured in examples/rag/query.py.

    Never raises: a failed or malformed judge call is reported as an
    error, not a fabricated score.
    """
    judge_model_id = os.getenv("BEDROCK_JUDGE_MODEL_ID", query_module.MODEL_ID)
    prompt = _JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = query_module.client.invoke_model(
            modelId=judge_model_id, body=json.dumps(body)
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"].strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        parsed = json.loads(text)
        return {
            "faithfulness": float(parsed["faithfulness"]),
            "answer_relevancy": float(parsed["answer_relevancy"]),
            "rationale": parsed.get("rationale"),
        }
    except Exception as exc:  # noqa: BLE001 - a bad judge call must not crash the run
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def evaluate(
    dataset: list[dict[str, Any]],
    dataset_path: Path = DEFAULT_DATASET,
    judge: str = "deterministic",
) -> dict[str, Any]:
    """Run generation evaluation over every question in ``dataset``."""
    ingest_module = load_ingest_module()
    query_module = load_query_module()
    usage_capture = _instrument_client_for_usage(query_module.client)

    input_price = _float_env("BEDROCK_INPUT_PRICE_PER_1K_USD")
    output_price = _float_env("BEDROCK_OUTPUT_PRICE_PER_1K_USD")

    per_question: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    faithfulness_values: list[float | None] = []
    relevancy_values: list[float | None] = []
    e2e_latencies: list[float] = []
    gen_latencies: list[float] = []
    cost_values: list[float | None] = []

    for question in dataset:
        qid = question.get("id", "<missing-id>")
        try:
            context, _retrieved_ids, retrieval_latency = retrieve_with_timing(ingest_module)

            usage_capture["usage"] = None
            gen_start = time.perf_counter()
            answer = query_module.ask(question["question"], _docs=context)
            gen_elapsed = time.perf_counter() - gen_start
            e2e_elapsed = retrieval_latency + gen_elapsed

            expected_answer = question.get("expected_answer", "")
            judge_meta: dict[str, Any] = {"rationale": None, "judge_error": None}

            if judge == "llm":
                judged = _llm_judge(question["question"], answer, context, query_module)
                faith_score = judged.get("faithfulness")
                relevancy_score = judged.get("answer_relevancy")
                judge_meta["rationale"] = judged.get("rationale")
                judge_meta["judge_error"] = judged.get("error")
            else:
                faith_score = metrics.faithfulness(answer, context)
                relevancy_score = metrics.answer_relevancy(answer, expected_answer)

            usage = usage_capture["usage"] or {}
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            cost = metrics.estimate_cost(input_tokens, output_tokens, input_price, output_price)

            faithfulness_values.append(faith_score)
            relevancy_values.append(relevancy_score)
            e2e_latencies.append(e2e_elapsed)
            gen_latencies.append(gen_elapsed)
            cost_values.append(cost)

            per_question.append(
                {
                    "id": qid,
                    "question": question["question"],
                    "expected_answer": expected_answer,
                    "generated_answer": answer,
                    "faithfulness": faith_score,
                    "answer_relevancy": relevancy_score,
                    "judge": judge,
                    **judge_meta,
                    "retrieval_latency_seconds": retrieval_latency,
                    "generation_latency_seconds": gen_elapsed,
                    "end_to_end_latency_seconds": e2e_elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                }
            )
        except Exception as exc:  # noqa: BLE001 - one bad question must not stop the run
            failures.append(
                {
                    "id": qid,
                    "stage": "generation",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    aggregate_metrics = {
        "faithfulness": metrics.aggregate_scores(faithfulness_values),
        "answer_relevancy": metrics.aggregate_scores(relevancy_values),
    }
    latency = {
        "generation_latency": metrics.latency_stats(gen_latencies),
        "end_to_end_latency": metrics.latency_stats(e2e_latencies),
    }

    scored_costs = [c for c in cost_values if c is not None]
    cost_summary = {
        "mean_usd": (sum(scored_costs) / len(scored_costs)) if scored_costs else None,
        "priced_queries": len(scored_costs),
        "total_queries": len(cost_values),
        "status": "estimated" if scored_costs else "unavailable",
        "note": (
            None
            if scored_costs
            else (
                "Token usage and/or per-token pricing were not available. Set "
                "BEDROCK_INPUT_PRICE_PER_1K_USD and BEDROCK_OUTPUT_PRICE_PER_1K_USD "
                "to enable cost estimation; see evaluation/README.md."
            )
        ),
    }

    configuration = {
        "region": query_module.REGION,
        "model_id": query_module.MODEL_ID,
        "max_tokens": query_module.MAX_TOKENS,
        "temperature": query_module.TEMPERATURE,
        "judge_mode": judge,
        "judge_model_id": (
            os.getenv("BEDROCK_JUDGE_MODEL_ID", query_module.MODEL_ID) if judge == "llm" else None
        ),
        "pricing_configured": input_price is not None and output_price is not None,
    }

    report = metrics.build_report(
        evaluation_type="generation",
        dataset_path=dataset_path,
        per_question_results=per_question,
        aggregate_metrics=aggregate_metrics,
        latency=latency,
        failures=failures,
        configuration=configuration,
    )
    report["cost_per_query"] = cost_summary
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG generation quality, latency and cost."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--judge", choices=["deterministic", "llm"], default="deterministic")
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    report = evaluate(dataset, dataset_path=args.dataset, judge=args.judge)
    report_path = metrics.save_report(report, args.reports_dir, prefix="generation_report")

    metrics.print_summary(report, title="RAG Generation Evaluation")
    cost = report["cost_per_query"]
    if cost["status"] == "estimated":
        print(
            f"\nCost/query (USD): mean=${cost['mean_usd']:.5f} "
            f"({cost['priced_queries']}/{cost['total_queries']} priced)"
        )
    else:
        print(f"\nCost/query (USD): unavailable -- {cost['note']}")
    print(f"\nReport saved to: {report_path}")

    if not report["per_question_results"] and report["failures"]:
        print(
            "\n[WARNING] Every question failed -- see 'failures' above and in the "
            "report. This is expected without AWS Bedrock access; see "
            "evaluation/README.md.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
