"""Explicit, offline step: build (or rebuild) the local vector index for
the RAG example from the documents under examples/rag/data/documents/.

Run from anywhere (paths are resolved relative to this file):

    python examples/rag/build_index.py

This never runs automatically as a side effect of asking a question --
see query.py's module docstring for why indexing and querying are kept
as separate steps. Re-run this after adding, editing, or removing
documents in examples/rag/data/documents/.
"""

import json
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from embeddings import EmbeddingError, get_embedding_client
from retrieval import RetrievalError, build_index

# Minimal, standalone config read (same env-var-overrides-config.json
# pattern as query.py) so this script does not need to import query.py
# just for configuration -- doing so would also construct query.py's
# module-level Bedrock client as an import side effect.
try:
    _config = json.loads((_THIS_DIR / "config.json").read_text())
except FileNotFoundError:
    _config = {}

REGION = os.getenv("AWS_REGION", _config.get("region", "us-east-1"))
EMBEDDING_MODEL_ID = os.getenv(
    "BEDROCK_EMBEDDING_MODEL_ID", _config.get("embedding_model_id", "amazon.titan-embed-text-v2:0")
)
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", _config.get("chunk_size", "800")))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", _config.get("chunk_overlap", "120")))
INDEX_DIR = _THIS_DIR / _config.get("index_dir", ".rag_index")


def main() -> int:
    import boto3

    try:
        client = boto3.client("bedrock-runtime", region_name=REGION)
        embedding_client = get_embedding_client("bedrock", client=client, model_id=EMBEDDING_MODEL_ID)
        build_index(
            embedding_client=embedding_client,
            index_dir=INDEX_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            on_progress=print,
        )
    except (EmbeddingError, RetrievalError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface AWS/config errors clearly, no raw traceback
        print(f"[ERROR] Could not build the vector index: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Index build complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
