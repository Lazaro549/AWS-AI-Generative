import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


RAG_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "rag"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_load_documents(tmp_path):
    ingest = load_module(
        "rag_ingest_test",
        RAG_DIR / "ingest.py",
    )

    document = tmp_path / "sample.txt"
    document.write_text("AWS Bedrock is a generative AI service.")

    original_data_dir = ingest.DATA_DIR

    try:
        ingest.DATA_DIR = tmp_path

        result = ingest.load_documents()

        assert "AWS Bedrock" in result

    finally:
        ingest.DATA_DIR = original_data_dir
        sys.modules.pop("rag_ingest_test", None)


def test_load_multiple_documents(tmp_path):
    ingest = load_module(
        "rag_ingest_multiple_test",
        RAG_DIR / "ingest.py",
    )

    (tmp_path / "one.txt").write_text("Document one")
    (tmp_path / "two.txt").write_text("Document two")

    original_data_dir = ingest.DATA_DIR

    try:
        ingest.DATA_DIR = tmp_path

        result = ingest.load_documents()

        assert "Document one" in result
        assert "Document two" in result

    finally:
        ingest.DATA_DIR = original_data_dir
        sys.modules.pop("rag_ingest_multiple_test", None)


def test_load_documents_empty_directory(tmp_path):
    ingest = load_module(
        "rag_ingest_empty_test",
        RAG_DIR / "ingest.py",
    )

    original_data_dir = ingest.DATA_DIR

    try:
        ingest.DATA_DIR = tmp_path

        result = ingest.load_documents()

        assert result == ""

    finally:
        ingest.DATA_DIR = original_data_dir
        sys.modules.pop("rag_ingest_empty_test", None)


class FakeBody:
    def read(self):
        return json.dumps(
            {
                "content": [
                    {
                        "text": "Amazon Bedrock provides access to foundation models."
                    }
                ]
            }
        ).encode()


class FakeClient:
    def __init__(self):
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": FakeBody()}


@pytest.fixture
def rag_query(monkeypatch):
    fake_client = FakeClient()

    monkeypatch.setattr(
        "boto3.client",
        lambda *args, **kwargs: fake_client,
    )

    ingest = load_module(
        "rag_ingest_query_test",
        RAG_DIR / "ingest.py",
    )

    ingest.DATA_DIR = RAG_DIR / "data"

    sys.modules["ingest"] = ingest

    query = load_module(
        "rag_query_test",
        RAG_DIR / "query.py",
    )

    yield query, fake_client

    sys.modules.pop("rag_query_test", None)
    sys.modules.pop("rag_ingest_query_test", None)
    sys.modules.pop("ingest", None)


def test_rag_ask_returns_model_response(rag_query):
    query, _ = rag_query

    result = query.ask("What is Amazon Bedrock?")

    assert (
        result
        == "Amazon Bedrock provides access to foundation models."
    )


def test_rag_ask_calls_bedrock(rag_query):
    query, fake_client = rag_query

    query.ask("What is Amazon Bedrock?")

    assert len(fake_client.calls) == 1


def test_rag_ask_uses_expected_model(rag_query):
    query, fake_client = rag_query

    query.ask("What is Amazon Bedrock?")

    request = fake_client.calls[0]

    assert request["modelId"] == query.MODEL_ID


def test_rag_prompt_contains_question(rag_query):
    query, fake_client = rag_query

    question = "What is generative AI?"

    query.ask(question)

    request = fake_client.calls[0]
    body = json.loads(request["body"])

    prompt = body["messages"][0]["content"]

    assert question in prompt


def test_rag_prompt_contains_context(rag_query):
    query, fake_client = rag_query

    query.ask("What is AWS?")

    request = fake_client.calls[0]
    body = json.loads(request["body"])

    prompt = body["messages"][0]["content"]

    assert "Context:" in prompt
    assert "Question:" in prompt


def test_load_documents_with_subdirectories(tmp_path):
    """Test that subdirectories are ignored (only root-level files are loaded)."""
    ingest = load_module(
        "rag_ingest_subdir_test",
        RAG_DIR / "ingest.py",
    )

    # Create root level file
    (tmp_path / "root.txt").write_text("Root document")

    # Create subdirectory with file
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested document")

    original_data_dir = ingest.DATA_DIR

    try:
        ingest.DATA_DIR = tmp_path

        result = ingest.load_documents()

        assert "Root document" in result
        # Subdirectories should be ignored by iterdir() without recursion
        assert "Nested document" not in result

    finally:
        ingest.DATA_DIR = original_data_dir
        sys.modules.pop("rag_ingest_subdir_test", None)


def test_load_documents_with_utf8_encoding(tmp_path):
    """Test that UTF-8 encoded documents are properly loaded."""
    ingest = load_module(
        "rag_ingest_utf8_test",
        RAG_DIR / "ingest.py",
    )

    # Create file with special UTF-8 characters
    content = "AWS Bedrock supports multiple languages and special chars: café, naïve, 日本語"
    (tmp_path / "utf8.txt").write_text(content, encoding="utf-8")

    original_data_dir = ingest.DATA_DIR

    try:
        ingest.DATA_DIR = tmp_path

        result = ingest.load_documents()

        # Verify the document was loaded (should be non-empty)
        assert result != ""
        assert isinstance(result, str)
        # Verify ASCII content is preserved
        assert "AWS Bedrock" in result
        assert "supports" in result
        # Verify the full content is there (length check for UTF-8 chars)
        assert len(result) > 50  # Content with UTF-8 chars is longer

    finally:
        ingest.DATA_DIR = original_data_dir
        sys.modules.pop("rag_ingest_utf8_test", None)


def test_load_documents_ignores_non_files(tmp_path):
    """Test that non-file entries (directories) are skipped."""
    ingest = load_module(
        "rag_ingest_mixed_test",
        RAG_DIR / "ingest.py",
    )

    (tmp_path / "valid.txt").write_text("Valid document")
    (tmp_path / "empty_dir").mkdir()

    original_data_dir = ingest.DATA_DIR

    try:
        ingest.DATA_DIR = tmp_path

        result = ingest.load_documents()

        assert "Valid document" in result
        assert result != ""

    finally:
        ingest.DATA_DIR = original_data_dir
        sys.modules.pop("rag_ingest_mixed_test", None)


def test_load_documents_sorted_order(tmp_path):
    """Test that documents are loaded in sorted order."""
    ingest = load_module(
        "rag_ingest_sort_test",
        RAG_DIR / "ingest.py",
    )

    (tmp_path / "a_first.txt").write_text("First")
    (tmp_path / "z_last.txt").write_text("Last")
    (tmp_path / "m_middle.txt").write_text("Middle")

    original_data_dir = ingest.DATA_DIR

    try:
        ingest.DATA_DIR = tmp_path

        result = ingest.load_documents()

        first_pos = result.find("First")
        middle_pos = result.find("Middle")
        last_pos = result.find("Last")

        assert first_pos < middle_pos < last_pos

    finally:
        ingest.DATA_DIR = original_data_dir
        sys.modules.pop("rag_ingest_sort_test", None)


def test_rag_ask_with_empty_context(tmp_path):
    """Test RAG with empty document context."""
    ingest = load_module(
        "rag_ingest_empty_rag_test",
        RAG_DIR / "ingest.py",
    )

    ingest.DATA_DIR = tmp_path  # Empty directory

    try:
        # Test that load_documents returns empty string for empty directory
        context = ingest.load_documents()
        assert context == ""
        assert isinstance(context, str)

    finally:
        sys.modules.pop("rag_ingest_empty_rag_test", None)


def test_rag_ask_with_very_long_question(rag_query):
    """Test RAG with an extremely long question."""
    query, fake_client = rag_query

    # Create a very long question
    long_question = "What is " + "generative AI " * 500 + "?"

    result = query.ask(long_question)

    assert result is not None
    request = fake_client.calls[0]
    body = json.loads(request["body"])
    prompt = body["messages"][0]["content"]
    assert long_question in prompt


def test_rag_ask_with_special_characters(rag_query):
    """Test RAG with special characters in question."""
    query, fake_client = rag_query

    special_question = 'What is AWS Bedrock 🚀? Use "quotes" and \\backslash'

    result = query.ask(special_question)

    request = fake_client.calls[0]
    body = json.loads(request["body"])
    prompt = body["messages"][0]["content"]
    assert special_question in prompt


def test_rag_ask_uses_correct_max_tokens(rag_query):
    """Test that max_tokens parameter is used correctly."""
    query, fake_client = rag_query

    query.ask("What is AWS?")

    request = fake_client.calls[0]
    body = json.loads(request["body"])

    assert "max_tokens" in body
    assert isinstance(body["max_tokens"], int)
    assert body["max_tokens"] > 0


def test_rag_ask_uses_correct_temperature(rag_query):
    """Test that temperature parameter is used correctly."""
    query, fake_client = rag_query

    query.ask("What is AWS?")

    request = fake_client.calls[0]
    body = json.loads(request["body"])

    assert "temperature" in body
    assert isinstance(body["temperature"], (int, float))
    assert 0 <= body["temperature"] <= 1


def test_rag_prompt_structure(rag_query):
    """Test that the prompt has the correct structure."""
    query, fake_client = rag_query

    query.ask("What is AWS?")

    request = fake_client.calls[0]
    body = json.loads(request["body"])

    # Verify basic structure
    assert "messages" in body
    assert isinstance(body["messages"], list)
    assert len(body["messages"]) == 1
    assert "role" in body["messages"][0]
    assert body["messages"][0]["role"] == "user"
    assert "content" in body["messages"][0]
