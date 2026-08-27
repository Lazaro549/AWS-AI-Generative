"""Interactive RAG CLI: retrieve relevant context, then ask Amazon Bedrock.

Retrieval strategy
-------------------
If a vector index has been built (see build_index.py), ``ask()``
performs real semantic retrieval: embed the question, search the index,
keep the top ``TOP_K`` chunks, and build a source-annotated context from
those chunks only -- never the whole corpus.

If no index exists yet -- a fresh checkout, or AWS credentials/Bedrock
access are not configured -- ``ask()`` falls back to the original
behaviour of this example (every document under examples/rag/data/,
concatenated via ingest.load_documents()), so the example keeps working
out of the box and existing callers (tests/test_rag.py,
evaluation/evaluate_generation.py) keep seeing the exact same, already
well-defined result.

Building the index is a deliberately separate, explicit step (also
triggered automatically once, below, when this file is run directly as
the CLI) rather than something ``ask()`` does lazily on first use. Two
reasons: it mirrors how a real RAG deployment separates offline
ingestion from online serving, and it keeps ask() free of any embedding
call in the common unit-test scenario (a mocked Bedrock client with no
index present), which is what lets tests/test_rag.py's
``test_rag_ask_calls_bedrock`` keep asserting exactly one Bedrock call
per question.
"""

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import boto3
from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ingest import load_documents
import context as context_module
import retrieval as retrieval_module
from embeddings import EmbeddingError, get_embedding_client

load_dotenv()

_config_path = Path(__file__).parent / "config.json"
try:
    config = json.loads(_config_path.read_text())
except FileNotFoundError:
    config = {}

REGION = os.getenv("AWS_REGION", config.get("region", "us-east-1"))
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0"))
MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", config.get("max_tokens", "512")))
TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", config.get("temperature", "0.3")))

# --- New, RAG-specific configuration (retrieval pipeline) -------------------
# Same reuse pattern as REGION/MODEL_ID above: an env var, if set,
# overrides config.json, which overrides a hard-coded default.
EMBEDDING_MODEL_ID = os.getenv(
    "BEDROCK_EMBEDDING_MODEL_ID", config.get("embedding_model_id", "amazon.titan-embed-text-v2:0")
)
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", config.get("chunk_size", "800")))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", config.get("chunk_overlap", "120")))
TOP_K = int(os.getenv("RAG_TOP_K", config.get("top_k", "3")))
INDEX_DIR = _THIS_DIR / config.get("index_dir", ".rag_index")

client = boto3.client("bedrock-runtime", region_name=REGION)


@lru_cache(maxsize=1)
def _cached_documents() -> str:
    return load_documents()


def _context_for_question(question: str) -> tuple[str, list]:
    """Return (context_text, retrieval_results) for ``question``.

    Real top-k retrieval when a vector index exists; the original
    full-corpus fallback (and an empty results list) otherwise -- see
    the module docstring for why building the index is never triggered
    from here.
    """
    index = retrieval_module.load_index(INDEX_DIR)
    if index is None:
        return _cached_documents(), []

    try:
        embedding_client = get_embedding_client("bedrock", client=client, model_id=EMBEDDING_MODEL_ID)
        results = retrieval_module.retrieve(
            question, index=index, embedding_client=embedding_client, top_k=TOP_K
        )
    except (EmbeddingError, retrieval_module.RetrievalError) as exc:
        print(f"[WARN] Falling back to full-corpus context ({exc})", file=sys.stderr)
        return _cached_documents(), []

    return context_module.assemble_context(results), results


def ask(question: str, _docs: str | None = None) -> str:
    if not question or not question.strip():
        raise ValueError("question must not be empty.")

    context = _docs if _docs is not None else _context_for_question(question)[0]
    prompt = context_module.build_prompt(question, context)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        print(f"Error calling Bedrock: {e}")
        raise


if __name__ == "__main__":
    if retrieval_module.load_index(INDEX_DIR) is None:
        print("No vector index found -- building one now (one-time step)...")
        try:
            embedding_client = get_embedding_client("bedrock", client=client, model_id=EMBEDDING_MODEL_ID)
            retrieval_module.build_index(
                embedding_client=embedding_client,
                index_dir=INDEX_DIR,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                on_progress=print,
            )
        except Exception as exc:  # noqa: BLE001 - config/AWS errors must not crash the CLI
            print(
                f"[WARN] Could not build the vector index ({type(exc).__name__}: {exc}). "
                "Falling back to full-corpus context for this session.",
                file=sys.stderr,
            )

    question = input("Ask a question: ")
    if not question or not question.strip():
        print("[ERROR] question must not be empty.")
        sys.exit(1)

    context_text, results = _context_for_question(question)
    if results:
        print("\nRetrieved context:")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.source} \u2014 score: {r.score:.2f}")

    answer = ask(question, _docs=context_text)
    print("\nAnswer:\n", answer)
