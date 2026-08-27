"""A small local vector index for the RAG pipeline, backed by FAISS.

FAISS and numpy are optional dependencies (see the ``[project.
optional-dependencies] rag`` extra in pyproject.toml, and
docs/getting-started or examples/rag/README.md for install
instructions). Importing this module never fails even if they are not
installed -- only the methods that actually need them do, raising a
clear VectorIndexError instead of a raw ImportError traceback. This
keeps the rest of the pipeline (chunking, document loading, the
full-corpus fallback in query.py) usable even in an environment where
the ``rag`` extra was not installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore

    _HAS_VECTOR_DEPS = True
except ImportError:  # pragma: no cover - exercised only without the extra installed
    _HAS_VECTOR_DEPS = False


class VectorIndexError(RuntimeError):
    """Raised for vector index configuration, availability, or usage
    problems (missing dependency, empty index, malformed inputs)."""


def _require_vector_deps() -> None:
    if not _HAS_VECTOR_DEPS:
        raise VectorIndexError(
            "faiss-cpu and numpy are required for vector search but are not "
            "installed. Install them with: pip install -e '.[rag]'"
        )


@dataclass
class SearchResult:
    """One ranked vector-search hit, with its stored payload."""

    chunk_id: str
    score: float
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorIndex:
    """Cosine-similarity search over a small set of chunk embeddings.

    Vectors are L2-normalized and compared with inner product (FAISS
    ``IndexFlatIP``), which is equivalent to cosine similarity and
    exact (no approximation) -- appropriate for the small,
    single-machine corpora this repository ships with. A larger corpus
    would swap in an approximate FAISS index type; the rest of the
    pipeline would not need to change.
    """

    def __init__(self) -> None:
        self._index = None
        self._payloads: list[dict[str, Any]] = []
        self._dimensions: int | None = None

    @property
    def size(self) -> int:
        return len(self._payloads)

    def build(self, chunks: Sequence[Any], vectors: Sequence[Sequence[float]]) -> None:
        """Build the index from parallel ``chunks`` and ``vectors``.

        ``chunks`` may be any objects with ``.chunk_id``, ``.text``,
        ``.source`` and ``.metadata`` attributes (e.g. chunking.Chunk).
        """
        _require_vector_deps()
        if len(chunks) != len(vectors):
            raise VectorIndexError("chunks and vectors must have the same length.")
        if not chunks:
            raise VectorIndexError("Cannot build a vector index from zero chunks.")

        matrix = np.array(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise VectorIndexError("vectors must be a list of equal-length numeric lists.")
        faiss.normalize_L2(matrix)
        self._dimensions = int(matrix.shape[1])
        self._index = faiss.IndexFlatIP(self._dimensions)
        self._index.add(matrix)
        self._payloads = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "metadata": dict(chunk.metadata),
            }
            for chunk in chunks
        ]

    def search(self, query_vector: Sequence[float], top_k: int = 3) -> list[SearchResult]:
        """Return the top_k most similar chunks to ``query_vector``,
        ordered from most to least relevant."""
        _require_vector_deps()
        if self._index is None or not self._payloads:
            raise VectorIndexError("Vector index is empty; build() or load() it first.")
        if top_k <= 0:
            raise VectorIndexError("top_k must be a positive integer.")

        query = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(query)
        k = min(top_k, len(self._payloads))
        scores, indices = self._index.search(query, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            payload = self._payloads[int(idx)]
            results.append(
                SearchResult(
                    chunk_id=payload["chunk_id"],
                    score=float(score),
                    source=payload["source"],
                    text=payload["text"],
                    metadata=payload["metadata"],
                )
            )
        return results

    def save(self, index_dir: Path) -> None:
        """Persist this index to ``index_dir`` (a FAISS binary file plus
        a JSON sidecar with chunk payloads and dimensionality)."""
        _require_vector_deps()
        if self._index is None:
            raise VectorIndexError("Nothing to save; build() the index first.")
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / "index.faiss"))
        (index_dir / "payloads.json").write_text(
            json.dumps({"dimensions": self._dimensions, "payloads": self._payloads}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        """Load a previously saved index. Raises VectorIndexError if
        ``index_dir`` does not contain one -- use ``VectorIndex.exists()``
        to check first without needing faiss/numpy installed."""
        _require_vector_deps()
        index_dir = Path(index_dir)
        index_path = index_dir / "index.faiss"
        payloads_path = index_dir / "payloads.json"
        if not index_path.exists() or not payloads_path.exists():
            raise VectorIndexError(f"No vector index found at {index_dir}.")

        instance = cls()
        instance._index = faiss.read_index(str(index_path))
        stored = json.loads(payloads_path.read_text(encoding="utf-8"))
        instance._dimensions = stored.get("dimensions")
        instance._payloads = stored.get("payloads", [])
        return instance

    @staticmethod
    def exists(index_dir: Path) -> bool:
        """Pure filesystem check for whether a built index is present --
        deliberately does not require faiss/numpy to be installed, so
        callers can decide whether to attempt real retrieval or fall
        back without paying any import cost either way."""
        index_dir = Path(index_dir)
        return (index_dir / "index.faiss").exists() and (index_dir / "payloads.json").exists()
