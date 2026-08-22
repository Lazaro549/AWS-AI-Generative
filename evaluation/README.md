# RAG Evaluation

## 📌 Purpose

This directory is an independent, objective evaluation framework for the
RAG example in [`examples/rag/`](../examples/rag/) (`ingest.py` +
`query.py`). It measures retrieval quality, generation quality, latency,
and cost/query, and produces both a machine-readable JSON report and a
concise terminal summary.

It does not modify `examples/rag/ingest.py` or `examples/rag/query.py` --
it evaluates them exactly as they exist in this repository, run through
their own real code paths.

## 🧪 Evaluation methodology

Retrieval and generation are evaluated **independently**, mirroring how a
RAG pipeline is normally debugged (a bad answer can come from bad
retrieval, bad generation, or both):

- `evaluate_retrieval.py` calls `examples/rag/ingest.py`'s
  `load_documents()` directly and scores what comes back against each
  dataset question's `relevant_documents`. This needs no AWS access.
- `evaluate_generation.py` calls `examples/rag/query.py`'s `ask()`
  function to produce a real answer from Amazon Bedrock, then scores that
  answer against the retrieved context (faithfulness) and against the
  dataset's `expected_answer` (answer relevancy). This needs the same
  Bedrock access as `query.py` itself.

Both scripts load `ingest.py` / `query.py` dynamically from their file
paths, the same way `tests/test_rag.py` already does. These example
scripts are written to be run directly (`query.py` uses a bare
`from ingest import load_documents`), not imported as an installed
package, so the evaluation scripts follow that existing convention rather
than inventing a different import mechanism.

## 📊 Dataset format (`dataset/rag_questions.json`)

A JSON array of objects:

```json
{
  "id": "rag-001",
  "question": "What is Amazon Bedrock?",
  "expected_answer": "...",
  "relevant_documents": ["examples/rag/data/sample.txt"]
}
```

- `id` -- unique string identifier, used in reports and test output.
- `question` -- sent to `ask()` for generation evaluation.
- `expected_answer` -- a human-written reference answer, used to score
  answer relevancy.
- `relevant_documents` -- repo-root-relative paths, under
  `examples/rag/data/`, that should ground the answer; used to score
  context precision/recall.

The initial dataset has 10 questions, all grounded in
`examples/rag/data/sample.txt` -- the only document currently in the
repository's RAG corpus. See "Limitations" below for what that means.

## 📈 Metrics (`metrics.py`)

| Metric | What it measures | How it's computed |
|---|---|---|
| Context Precision | Are retrieved documents relevant? | `\|retrieved ∩ relevant\| / \|retrieved\|` |
| Context Recall | Was everything needed retrieved? | `\|retrieved ∩ relevant\| / \|relevant\|` |
| Faithfulness | Is the answer grounded in the retrieved context? | Deterministic: fraction of answer sentences whose content words sufficiently overlap the context. Optional: Bedrock LLM judge. |
| Answer Relevancy | Does the answer address the question? | Deterministic: token-F1 overlap between the generated answer and `expected_answer` (a proxy -- see Limitations). Optional: Bedrock LLM judge, scored directly against the question. |
| Retrieval latency | Time to retrieve context | Wall-clock time around `load_documents()` |
| End-to-end latency | Time from question to final answer | Wall-clock time around retrieval + `ask()` |
| Cost/query | Approximate USD cost | Real token usage captured from the live Bedrock response, multiplied by a price you supply. Reported as `unavailable` otherwise -- no price is invented (see below). |

Every metric function returns `None` (not `0.0` or `1.0`) when it cannot
be computed from the given input (e.g. an empty retrieval, or no
reference answer) -- see the docstrings in `metrics.py` for the exact
rule per metric. Aggregates report both a `mean` (over defined scores
only) and a `count_undefined`, so missing data stays visible in the
report instead of being silently folded into an average.

## ▶️ Running retrieval evaluation

No AWS credentials are required -- retrieval is local file I/O only.

```bash
python evaluation/evaluate_retrieval.py
```

Optional flags: `--dataset PATH`, `--reports-dir PATH`.

## ▶️ Running generation evaluation

Requires the same Bedrock access as `python examples/rag/query.py` --
see "AWS credentials and configuration" below.

```bash
python evaluation/evaluate_generation.py
python evaluation/evaluate_generation.py --judge llm   # optional LLM judge
```

If a question fails (no AWS credentials, throttling, a malformed
response, ...) it is recorded in the report's `failures` list with the
error message, and the run continues with the remaining questions -- one
failed call never crashes the whole evaluation.

## 📄 How reports are generated

Each run writes one timestamped JSON report to `evaluation/reports/`:

- `retrieval_report_<UTC timestamp>.json`
- `generation_report_<UTC timestamp>.json`

Each report includes `timestamp_utc`, `num_questions_evaluated`,
`aggregate_metrics`, `latency`, `per_question_results`, `failures`, and
`configuration` (model id, region, max tokens, temperature, judge mode --
never credentials). A concise human-readable version is also printed to
the terminal. `reports/.gitkeep` keeps the otherwise-empty directory
tracked in git; generated reports are run artifacts, not source, so you
may want to `.gitignore` the timestamped files themselves.

## 🔐 AWS credentials and configuration

This evaluation reuses `examples/rag/query.py`'s existing configuration:
`AWS_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_MAX_TOKENS`,
`BEDROCK_TEMPERATURE` (environment variables, falling back to
`examples/rag/config.json`), and standard AWS credential resolution
(`aws configure`, `~/.aws/credentials`, or `AWS_PROFILE`). No credentials
are read from or written to this repository, and no evaluation code
prints or logs secret values.

Two additional environment variables are specific to this evaluation and
default to unset (no behaviour changes unless you set them):

- `BEDROCK_JUDGE_MODEL_ID` -- model used for `--judge llm` (defaults to
  the same `BEDROCK_MODEL_ID` used for generation).
- `BEDROCK_INPUT_PRICE_PER_1K_USD` / `BEDROCK_OUTPUT_PRICE_PER_1K_USD` --
  your own current Bedrock price per 1,000 tokens, used only to turn
  captured token usage into a dollar estimate. Leave unset to keep cost
  reporting as `unavailable`.

Unit tests never call AWS: `tests/test_evaluation_pipeline.py` mocks
`boto3.client`, exactly like the existing `tests/test_rag.py` does.

## ⚠️ Limitations

- **Single-document corpus.** `examples/rag/data/` currently contains one
  file (`sample.txt`), and `load_documents()` always returns *every* file
  in that folder, concatenated -- there is no similarity search or
  chunking in this repository. As a result, every dataset question's
  `relevant_documents` is that same file, and Context Precision/Recall
  are expected to be `1.0` today, by construction. These metrics become
  discriminative the moment the corpus grows to multiple files and
  `ingest.py` gains real retrieval logic -- this evaluation is written
  against that future, not only against today's trivial case.
- **Deterministic faithfulness/relevancy are lexical heuristics.** Word
  overlap is not entailment: a well-worded paraphrase can score lower
  than it should, and an answer that reuses context words without truly
  answering the question can score higher than it should. The optional
  `--judge llm` mode is more semantically aware but costs a Bedrock call
  per question, and shares the generator's own model unless
  `BEDROCK_JUDGE_MODEL_ID` points at a different one -- so it is subject
  to self-evaluation bias.
- **Answer Relevancy is measured against `expected_answer`, not the raw
  question**, because a deterministic way to judge relevance to an
  open-ended question (without embeddings or an LLM) is unreliable on its
  own -- an answer could reuse the question's words without answering it.
  `--judge llm` scores relevancy directly against the question instead.
- **`prompts/rag.txt`** defines a stricter, context-only prompt with an
  explicit "I don't have enough information" fallback, but
  `examples/rag/query.py` builds its own inline prompt and does not use
  it. This evaluation scores the prompt `query.py` actually sends, not
  `prompts/rag.txt` -- see the recommendation in the final summary.
- **Cost/query** depends on the Bedrock response including a `usage`
  field (true for the Claude 3 models this repository is configured for)
  and on you supplying current pricing; without both it is reported as
  `unavailable` rather than guessed.
- **End-to-end latency** measures retrieval + generation explicitly (via
  `ask()`'s existing `_docs` parameter) instead of relying on
  `query.py`'s internal `functools.lru_cache` for document loading, so
  that latency is comparable across every evaluated question rather than
  only the first one paying the full retrieval cost.

## ➕ Adding new evaluation questions

1. Add real content to `examples/rag/data/` if the question needs
   information that isn't in the corpus yet.
2. Append an entry to `dataset/rag_questions.json` with a unique `id`,
   the `question`, a human-written `expected_answer`, and the
   repo-root-relative paths of the files it should be grounded in, under
   `relevant_documents`.
3. Run both evaluation scripts again -- no code changes are required.
