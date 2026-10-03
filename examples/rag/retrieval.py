"""Retrieval orchestration: documents -> chunks -> embeddings -> vector
index -> similarity search -> ranked, source-aware results.

This module ties ingest.py, chunking.py, embeddings.py and
vector_index.py together into the two operations a caller needs:

- ``build_index(...)`` -- an explicit, offline step (see build_index.py
  and query.py's __main__ block) that embeds every chunk of the corpus
  and persists a searchable index.
- ``retrieve(...)``    -- embeds a single query and searches an
  already-built index, returning ranked, source-aware results.

Deliberately NOT done here: automatically building a missing index as a
side effect of answering a question. Indexing and querying are kept as
separate steps, the way a real RAG deployment separates offline
ingestion from online serving -- see query.py's module docstring for
the full reasoning (it also keeps the existing, mocked-Bedrock unit
tests fast and deterministic, since they never trigger an embedding
pass over the whole corpus just by asking one question).
"""

import sys as _sys
from pathlib import Path as _Path

_THIS_DIR = _Path(__file__).resolve().parent
if str(_THIS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_THIS_DIR))

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_document
from embeddings import EmbeddingClient, EmbeddingError
from ingest import load_documents_structured
from vector_index import VectorIndex, VectorIndexError

# Default validated by the offline retrieval benchmark (evaluation/retrieval_benchmark.py):
# top_k=3 (balanced config) achieved recall=0.667 vs recall=0.250 for top_k=2 (small config)
# on the 18-question benchmark, with no latency penalty.
DEFAULT_TOP_K = 3
DEFAULT_INDEX_DIRNAME = ".rag_index"


class RetrievalError(RuntimeError):
    """Raised for retrieval-time problems: an empty question, an
    invalid top_k, or a vector-search failure."""


@dataclass
class RetrievalResult:
    """One ranked, source-aware retrieval hit, ready for context
    construction (see context.py)."""

    chunk_id: str
    source: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def default_index_dir(rag_dir: _Path | None = None) -> _Path:
    """Where the vector index lives by default: examples/rag/.rag_index/
    (git-ignored -- see .gitignore -- since it is a rebuildable, derived
    artifact, not source)."""
    rag_dir = rag_dir or _THIS_DIR
    return rag_dir / DEFAULT_INDEX_DIRNAME


def build_index(
    *,
    embedding_client: EmbeddingClient,
    data_dir: _Path | None = None,
    index_dir: _Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    on_progress: Callable[[str], None] | None = None,
) -> VectorIndex:
    """Load the corpus, chunk it, embed every chunk, and persist a
    searchable VectorIndex. This is the explicit "offline" indexing
    step; see the module docstring for why it never runs implicitly.
    """
    documents = load_documents_structured(data_dir)
    if not documents:
        raise RetrievalError(
            "No documents found to index. Add .md/.txt files under "
            f"{data_dir or 'examples/rag/data/documents'}."
        )

    all_chunks = []
    for document in documents:
        all_chunks.extend(
            chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    if not all_chunks:
        raise RetrievalError("Documents were found but produced zero chunks.")

    if on_progress:
        on_progress(f"Embedding {len(all_chunks)} chunk(s) from {len(documents)} document(s)...")

    try:
        vectors = embedding_client.embed([chunk.text for chunk in all_chunks])
    except EmbeddingError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any unexpected embedding failure
        raise EmbeddingError(
            f"Failed to embed corpus chunks: {type(exc).__name__}: {exc}"
        ) from exc

    index = VectorIndex()
    index.build(all_chunks, vectors)

    target_dir = index_dir or default_index_dir()
    index.save(target_dir)
    if on_progress:
        on_progress(f"Saved vector index ({index.size} chunk(s)) to {target_dir}")
    return index


def load_index(index_dir: _Path | None = None) -> VectorIndex | None:
    """Load a previously built index, or return None if none exists.

    Never raises for a simply-missing index -- that is an expected
    state (e.g. a fresh checkout before build_index has been run, or an
    environment without the ``rag`` extra installed), not an error.
    """
    target_dir = index_dir or default_index_dir()
    if not VectorIndex.exists(target_dir):
        return None
    try:
        return VectorIndex.load(target_dir)
    except VectorIndexError:
        return None


def retrieve(
    question: str,
    *,
    index: VectorIndex,
    embedding_client: EmbeddingClient,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    """Embed ``question`` and return its top-k most similar chunks from
    ``index``, most relevant first."""
    if not question or not question.strip():
        raise RetrievalError("Cannot retrieve context for an empty question.")
    if top_k <= 0:
        raise RetrievalError("top_k must be a positive integer.")

    try:
        query_vector = embedding_client.embed_one(question)
    except EmbeddingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(
            f"Failed to embed the question: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        hits = index.search(query_vector, top_k=top_k)
    except VectorIndexError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"Vector search failed: {type(exc).__name__}: {exc}") from exc

    return [
        RetrievalResult(
            chunk_id=hit.chunk_id, source=hit.source, score=hit.score, text=hit.text, metadata=hit.metadata
        )
        for hit in hits
    ]
