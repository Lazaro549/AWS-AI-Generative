"""Regression tests for short trailing windows in RAG chunking."""

import pytest

from chunking import chunk_text


@pytest.mark.parametrize(
    ("text_length", "chunk_size", "chunk_overlap"),
    [
        (960, 500, 50),  # Previously merged a 60-char tail into a 510-char chunk.
        (960, 50, 10),   # Minimum tail must adapt when chunks are smaller than 80 chars.
        (105, 20, 0),    # No overlap and a short final window.
    ],
)
def test_short_trailing_window_never_exceeds_configured_chunk_size(
    text_length, chunk_size, chunk_overlap
):
    chunks = chunk_text(
        "x" * text_length,
        document_id="regression",
        source="regression.txt",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= chunk_size for chunk in chunks)
    assert len(chunks[-1].text) >= min(80, chunk_size)
