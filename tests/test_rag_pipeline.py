"""Tests for the new RAG retrieval pipeline: chunking, embeddings, the
FAISS-backed vector index, retrieval orchestration, context/prompt
construction, and query.py's real-retrieval-with-fallback behaviour.

This is a NEW file only -- it does not modify tests/test_rag.py or any
other existing test. All AWS-dependent behaviour is exercised with the
deterministic MockEmbeddingClient (see examples/rag/embeddings.py), so
nothing here requires real AWS credentials.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RAG_DIR = Path(__file__).resolve().parents[1] / "examples" / "rag"

# Make the new pipeline modules importable the normal way. They are
# self-contained/defensive about sys.path themselves (see their module
# docstrings), but doing this once here keeps every test in this file
# simple and consistent.
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

import chunking  # noqa: E402
import context as context_module  # noqa: E402
import ingest  # noqa: E402
import retrieval  # noqa: E402
import vector_index  # noqa: E402
from embeddings import MockEmbeddingClient  # noqa: E402


def load_module(name, path):
    """Same dynamic-loading helper tests/test_rag.py uses, for the one
    test below that needs its own isolated copy of query.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- chunking.py -------------------------------------------------------


def test_chunk_text_respects_configured_size_and_overlap():
    text = "x" * 1000
    chunks = chunking.chunk_text(
        text, document_id="doc", source="doc.md", chunk_size=300, chunk_overlap=50
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 300
    # Consecutive windows should overlap by the configured amount.
    assert chunks[0].metadata["char_end"] - chunks[1].metadata["char_start"] == 50


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunking.chunk_text("", document_id="doc", source="doc.md") == []
    assert chunking.chunk_text("   \n  ", document_id="doc", source="doc.md") == []


def test_chunk_text_avoids_tiny_trailing_fragment():
    # chunk_size=500, overlap=50 => step=450; length=960 would otherwise
    # produce a final 60-character window, below MIN_CHUNK_CHARS (80).
    text = "y" * 960
    chunks = chunking.chunk_text(
        text, document_id="doc", source="doc.md", chunk_size=500, chunk_overlap=50
    )
    assert all(len(c.text) >= chunking.MIN_CHUNK_CHARS for c in chunks)


def test_chunk_text_is_deterministic():
    text = "AWS Bedrock provides access to foundation models. " * 20
    first = chunking.chunk_text(text, document_id="doc", source="doc.md")
    second = chunking.chunk_text(text, document_id="doc", source="doc.md")
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_chunk_text_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        chunking.chunk_text("hello world", document_id="doc", source="doc.md", chunk_size=10, chunk_overlap=10)


def test_chunk_document_uses_document_attributes():
    document = ingest.Document(
        id="aws_lambda", source="aws_lambda.md", path="/tmp/aws_lambda.md", title="Aws Lambda",
        text="AWS Lambda is a serverless compute service. " * 10,
    )
    chunks = chunking.chunk_document(document, chunk_size=200, chunk_overlap=20)
    assert chunks
    assert all(c.document_id == "aws_lambda" for c in chunks)
    assert all(c.source == "aws_lambda.md" for c in chunks)


# --- embeddings.py (mock client) ---------------------------------------


def test_mock_embedding_client_is_deterministic_and_normalized():
    client = MockEmbeddingClient(dimensions=32)
    v1 = client.embed_one("Amazon Bedrock")
    v2 = client.embed_one("Amazon Bedrock")
    v3 = client.embed_one("AWS Lambda")

    assert v1 == v2
    assert v1 != v3
    norm = sum(x * x for x in v1) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


# --- ingest.py: structured loading --------------------------------------


def test_load_documents_structured_reads_the_real_corpus():
    documents = ingest.load_documents_structured()
    ids = {d.id for d in documents}
    assert {"aws_bedrock", "aws_lambda", "aws_s3", "aws_iam", "aws_sam", "aws_rag"} <= ids
    for d in documents:
        assert d.text.strip()
        assert d.source.endswith((".md", ".txt"))


def test_load_documents_structured_ignores_missing_directory(tmp_path):
    assert ingest.load_documents_structured(tmp_path / "does-not-exist") == []


def test_load_documents_structured_skips_unsupported_extensions(tmp_path):
    (tmp_path / "notes.md").write_text("Real document content.", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    documents = ingest.load_documents_structured(tmp_path)
    assert [d.source for d in documents] == ["notes.md"]


def test_load_documents_structured_raises_on_bad_encoding(tmp_path):
    bad_file = tmp_path / "broken.md"
    bad_file.write_bytes(b"\xff\xfe\x00\x01 not valid utf-8")

    with pytest.raises(ingest.DocumentLoadError):
        ingest.load_documents_structured(tmp_path)


def test_load_documents_structured_does_not_affect_legacy_loader(tmp_path):
    # examples/rag/data/documents/ is a *subdirectory* of DATA_DIR; the
    # legacy, non-recursive load_documents() must never see it.
    original_data_dir = ingest.DATA_DIR
    try:
        (tmp_path / "sample.txt").write_text("legacy content", encoding="utf-8")
        (tmp_path / "documents").mkdir()
        (tmp_path / "documents" / "new.md").write_text("new content", encoding="utf-8")

        ingest.DATA_DIR = tmp_path
        result = ingest.load_documents()
        assert result == "legacy content"
        assert "new content" not in result
    finally:
        ingest.DATA_DIR = original_data_dir


# --- vector_index.py -----------------------------------------------------


def _tiny_index():
    mock = MockEmbeddingClient(dimensions=16)
    chunks = [
        chunking.Chunk(chunk_id="a::chunk-0", text="Amazon Bedrock text", source="a.md", document_id="a"),
        chunking.Chunk(chunk_id="b::chunk-0", text="AWS Lambda text", source="b.md", document_id="b"),
        chunking.Chunk(chunk_id="c::chunk-0", text="Amazon S3 text", source="c.md", document_id="c"),
    ]
    vectors = mock.embed([c.text for c in chunks])
    index = vector_index.VectorIndex()
    index.build(chunks, vectors)
    return index, mock, chunks


def test_vector_index_search_orders_results_by_descending_score():
    index, mock, chunks = _tiny_index()
    query_vector = mock.embed_one(chunks[0].text)  # identical text -> best match is itself
    results = index.search(query_vector, top_k=3)

    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk_id == chunks[0].chunk_id
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_vector_index_search_caps_top_k_at_available_size():
    index, mock, chunks = _tiny_index()
    results = index.search(mock.embed_one("anything"), top_k=100)
    assert len(results) == len(chunks)


def test_vector_index_build_rejects_empty_input():
    with pytest.raises(vector_index.VectorIndexError):
        vector_index.VectorIndex().build([], [])


def test_vector_index_load_missing_raises_clear_error(tmp_path):
    with pytest.raises(vector_index.VectorIndexError):
        vector_index.VectorIndex.load(tmp_path / "nope")


def test_vector_index_save_and_load_round_trip(tmp_path):
    index, mock, chunks = _tiny_index()
    index.save(tmp_path)

    assert vector_index.VectorIndex.exists(tmp_path)
    reloaded = vector_index.VectorIndex.load(tmp_path)
    results = reloaded.search(mock.embed_one(chunks[1].text), top_k=1)
    assert results[0].chunk_id == chunks[1].chunk_id


# --- retrieval.py ----------------------------------------------------------


def test_retrieve_rejects_empty_question():
    index, mock, _ = _tiny_index()
    with pytest.raises(retrieval.RetrievalError):
        retrieval.retrieve("   ", index=index, embedding_client=mock)


def test_retrieve_returns_ranked_source_aware_results():
    index, mock, chunks = _tiny_index()
    results = retrieval.retrieve(chunks[2].text, index=index, embedding_client=mock, top_k=2)
    assert len(results) == 2
    assert results[0].source == "c.md"
    assert all(r.text for r in results)


def test_build_index_and_load_end_to_end_with_mock_embeddings(tmp_path):
    mock = MockEmbeddingClient(dimensions=32)
    built = retrieval.build_index(
        embedding_client=mock,
        data_dir=ingest.CORPUS_DIR,
        index_dir=tmp_path,
        chunk_size=400,
        chunk_overlap=50,
    )
    assert built.size > 6  # more than one chunk per document

    loaded = retrieval.load_index(tmp_path)
    assert loaded is not None
    results = retrieval.retrieve("What does AWS Lambda do?", index=loaded, embedding_client=mock, top_k=3)
    assert 1 <= len(results) <= 3


def test_build_index_raises_clear_error_for_empty_corpus(tmp_path):
    with pytest.raises(retrieval.RetrievalError):
        retrieval.build_index(embedding_client=MockEmbeddingClient(), data_dir=tmp_path, index_dir=tmp_path / "idx")


def test_load_index_returns_none_when_absent(tmp_path):
    assert retrieval.load_index(tmp_path / "missing") is None


# --- context.py --------------------------------------------------------


def test_assemble_context_labels_each_chunk_with_its_source():
    results = [
        retrieval.RetrievalResult(chunk_id="a::chunk-0", source="aws_s3.md", score=0.9, text="S3 stores objects."),
        retrieval.RetrievalResult(chunk_id="b::chunk-0", source="aws_iam.md", score=0.7, text="IAM controls access."),
    ]
    assembled = context_module.assemble_context(results)
    assert "[Source: aws_s3.md]" in assembled
    assert "[Source: aws_iam.md]" in assembled
    assert "S3 stores objects." in assembled


def test_assemble_context_empty_results_is_empty_string():
    assert context_module.assemble_context([]) == ""


def test_build_prompt_contains_context_and_question():
    prompt = context_module.build_prompt("What is IAM?", "IAM controls access.")
    assert "Context:" in prompt
    assert "Question:" in prompt
    assert "What is IAM?" in prompt
    assert "IAM controls access." in prompt


def test_build_prompt_handles_curly_braces_in_content():
    prompt = context_module.build_prompt(
        "What does {config} mean?", "A dict literal like {\"a\": 1} is valid JSON."
    )
    assert "What does {config} mean?" in prompt
    assert 'A dict literal like {"a": 1} is valid JSON.' in prompt


# --- query.py: fallback and real-retrieval integration ----------------------


def test_query_ask_falls_back_to_full_corpus_when_no_index_present(monkeypatch, tmp_path):
    """No index on disk -> ask() must use the same full-corpus context
    load_documents() has always produced, with zero extra Bedrock calls
    (see tests/test_rag.py::test_rag_ask_calls_bedrock for that contract)."""

    class FakeClient:
        def __init__(self):
            self.calls = []

        def invoke_model(self, modelId, body):
            self.calls.append({"modelId": modelId, "body": json.loads(body)})

            class FakeBody:
                def read(self_inner):
                    return json.dumps({"content": [{"text": "answer"}]}).encode("utf-8")

            return {"body": FakeBody()}

    fake_client = FakeClient()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake_client)

    ingest_copy = load_module("rag_ingest_pipeline_test", RAG_DIR / "ingest.py")
    ingest_copy.DATA_DIR = RAG_DIR / "data"
    sys.modules["ingest"] = ingest_copy

    query_module = load_module("rag_query_pipeline_test", RAG_DIR / "query.py")
    query_module.INDEX_DIR = tmp_path / "no-index-here"

    result = query_module.ask("What is Amazon Bedrock?")

    assert result == "answer"
    assert len(fake_client.calls) == 1
    prompt = fake_client.calls[0]["body"]["messages"][0]["content"]
    assert "Amazon Bedrock" in prompt  # from the real sample.txt content


def test_query_ask_uses_real_retrieval_when_index_is_present(monkeypatch, tmp_path):
    """With a pre-built index, ask() must send only the retrieved
    chunks as context -- not the whole corpus -- while still making
    exactly one Bedrock call (query embedding is served by the mock
    client here, generation by the fake Bedrock client)."""

    class FakeClient:
        def __init__(self):
            self.calls = []

        def invoke_model(self, modelId, body):
            self.calls.append({"modelId": modelId, "body": json.loads(body)})

            class FakeBody:
                def read(self_inner):
                    return json.dumps({"content": [{"text": "answer"}]}).encode("utf-8")

            return {"body": FakeBody()}

    fake_client = FakeClient()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake_client)

    mock_embeddings = MockEmbeddingClient(dimensions=32)
    retrieval.build_index(
        embedding_client=mock_embeddings, data_dir=ingest.CORPUS_DIR, index_dir=tmp_path, chunk_size=400, chunk_overlap=50
    )

    ingest_copy = load_module("rag_ingest_pipeline_test2", RAG_DIR / "ingest.py")
    ingest_copy.DATA_DIR = RAG_DIR / "data"
    sys.modules["ingest"] = ingest_copy

    query_module = load_module("rag_query_pipeline_test2", RAG_DIR / "query.py")
    query_module.INDEX_DIR = tmp_path
    # Swap in the mock embedding client so this test never needs a real
    # Bedrock embeddings response shape from FakeClient.
    monkeypatch.setattr(query_module, "get_embedding_client", lambda *a, **k: mock_embeddings)

    result = query_module.ask("What does AWS S3 store?")

    assert result == "answer"
    assert len(fake_client.calls) == 1  # generation only; embedding used the mock client directly
    prompt = fake_client.calls[0]["body"]["messages"][0]["content"]
    assert "[Source:" in prompt  # source-labeled retrieved context, not the raw sample.txt dump
    full_corpus = ingest_copy.load_documents()
    assert prompt.count(full_corpus) == 0  # the whole corpus was not dumped into the prompt
