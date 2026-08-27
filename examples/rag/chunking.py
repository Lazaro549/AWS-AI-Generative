"""Deterministic, configurable text chunking for the RAG pipeline.

This module has no external dependencies (standard library only) and no
side effects, which makes it trivial to unit test in isolation -- see
tests/test_rag_pipeline.py. It also has no sibling imports (it does not
import ingest.Document), so it can always be imported regardless of how
or in what order the rest of the pipeline's modules are loaded.

Chunking is character-based (not token-based) to avoid adding a
tokenizer dependency; chunk_size/chunk_overlap are counted in
characters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# A trailing fragment shorter than this (in characters) is merged into
# the previous chunk instead of becoming its own tiny chunk -- see
# _split_into_windows().
MIN_CHUNK_CHARS = 80


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text, with enough metadata to trace it
    back to its source document and position within it."""

    chunk_id: str
    text: str
    source: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_into_windows(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Return a deterministic list of (start, end) character offsets
    covering ``text``.

    Windows advance by ``chunk_size - chunk_overlap`` characters each
    step. A short trailing window (< MIN_CHUNK_CHARS) is merged into the
    previous one, so chunking never produces near-empty fragments.
    """
    length = len(text)
    if length == 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        windows.append((start, end))
        if end == length:
            break
        start += step

    if len(windows) > 1:
        last_start, last_end = windows[-1]
        if (last_end - last_start) < MIN_CHUNK_CHARS:
            prev_start, _prev_end = windows[-2]
            windows[-2] = (prev_start, last_end)
            windows.pop()

    return windows


def chunk_text(
    text: str,
    *,
    document_id: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping, deterministic Chunk objects.

    Splitting uses plain character windows -- no randomness, no
    external calls -- so identical input always produces identical
    chunks (same ids, same boundaries), which is what makes this
    testable and safe to re-run when rebuilding the index.
    """
    if text is None:
        return []
    normalized = text.strip()
    if not normalized:
        return []

    base_metadata = dict(metadata or {})
    windows = _split_into_windows(normalized, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(windows):
        chunk_metadata = dict(base_metadata)
        chunk_metadata.update(
            {
                "chunk_index": index,
                "char_start": start,
                "char_end": end,
                "chunk_count": len(windows),
            }
        )
        chunk_text_value = normalized[start:end].strip()
        if not chunk_text_value:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}::chunk-{index}",
                text=chunk_text_value,
                source=source,
                document_id=document_id,
                metadata=chunk_metadata,
            )
        )
    return chunks


def chunk_document(
    document: Any,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a document-like object.

    Accepts anything with ``.id``, ``.source`` and ``.text`` attributes
    (e.g. ``ingest.Document``) rather than importing that class
    directly, so this module never needs a sibling import.
    """
    metadata = dict(getattr(document, "metadata", None) or {})
    title = getattr(document, "title", None)
    if title:
        metadata.setdefault("document_title", title)
    return chunk_text(
        document.text,
        document_id=document.id,
        source=document.source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=metadata,
    )
