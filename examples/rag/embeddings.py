"""Embedding clients for the RAG pipeline.

Two implementations share one small interface:

- ``BedrockEmbeddingClient`` -- calls Amazon Bedrock (this repository's
  existing, AWS-native provider) via the same boto3 bedrock-runtime
  client shape already used by examples/rag/query.py for generation.
  This is the real, semantic implementation and the intended provider
  for actual use. The embedding model id is configurable (see
  DEFAULT_BEDROCK_EMBEDDING_MODEL_ID and query.py's EMBEDDING_MODEL_ID)
  -- it is never hard-coded as the only option.
- ``MockEmbeddingClient`` -- a deterministic, dependency-free stand-in
  used by tests and local development so unit tests never need real AWS
  credentials (see tests/test_rag_pipeline.py). It is isolated here,
  clearly documented as non-semantic, and never selected automatically
  -- callers choose it explicitly via get_embedding_client("mock").

Both raise EmbeddingError on failure, with a message that is always
safe to print or log (never a credential or full request body).
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Protocol, Sequence

DEFAULT_BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
MOCK_EMBEDDING_DIMENSIONS = 256


class EmbeddingError(RuntimeError):
    """Raised when an embedding request cannot be completed (missing
    AWS credentials, a Bedrock/network failure, an empty input, or an
    unexpected response shape). Messages never include credentials.
    """


class EmbeddingClient(Protocol):
    """Minimal interface both embedding clients implement."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


class MockEmbeddingClient:
    """Deterministic, offline embedding client for tests and local
    development -- never used in production by default.

    Each text is hashed (SHA-256) into a fixed-size vector, then
    L2-normalized. Identical input always yields an identical vector;
    different inputs yield different (pseudo-random, unit-length)
    vectors. That is enough to exercise chunking, indexing, and top-k
    search end-to-end with no network access and no AWS credentials.

    This is explicitly NOT a semantic embedding: it has no notion of
    meaning or similarity between related texts, so it must never be
    used to judge retrieval *quality* -- only pipeline *mechanics*
    (ordering, scoring, persistence). See BedrockEmbeddingClient for
    real, semantic embeddings.
    """

    def __init__(self, dimensions: int = MOCK_EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def _vector_for(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimensions:
            digest = hashlib.sha256(f"{text}::{counter}".encode("utf-8")).digest()
            for word in struct.unpack(">8I", digest):
                values.append((word / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        values = values[: self.dimensions]
        norm = sum(v * v for v in values) ** 0.5
        if norm == 0:
            return values
        return [v / norm for v in values]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector_for(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vector_for(text)


class BedrockEmbeddingClient:
    """Real semantic embeddings via Amazon Bedrock.

    Uses the same boto3 bedrock-runtime client shape as
    examples/rag/query.py's generation client -- callers are expected
    to construct that client the same way (``boto3.client(
    "bedrock-runtime", region_name=...)``) and pass it in, so this
    class never reads AWS credentials or region configuration itself.
    """

    def __init__(self, client, model_id: str = DEFAULT_BEDROCK_EMBEDDING_MODEL_ID) -> None:
        self._client = client
        self.model_id = model_id

    def _invoke(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed an empty string.")
        body = json.dumps({"inputText": text})
        try:
            response = self._client.invoke_model(modelId=self.model_id, body=body)
            payload = json.loads(response["body"].read())
        except Exception as exc:  # noqa: BLE001 - normalize every AWS/network failure mode
            raise EmbeddingError(
                f"Bedrock embedding request failed for model '{self.model_id}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        embedding = payload.get("embedding")
        if not embedding:
            raise EmbeddingError(
                f"Bedrock embedding response for model '{self.model_id}' did not "
                "include an 'embedding' field."
            )
        return embedding

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._invoke(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._invoke(text)


def get_embedding_client(
    provider: str, *, client=None, model_id: str | None = None
) -> EmbeddingClient:
    """Factory for embedding clients.

    ``provider`` is explicit ("bedrock" or "mock") -- there is no
    silent fallback from Bedrock to the mock client, so a misconfigured
    production environment fails clearly (EmbeddingError) instead of
    quietly returning meaningless vectors.
    """
    if provider == "mock":
        return MockEmbeddingClient()
    if provider == "bedrock":
        if client is None:
            raise EmbeddingError(
                "A boto3 bedrock-runtime client is required for the 'bedrock' provider."
            )
        return BedrockEmbeddingClient(client, model_id=model_id or DEFAULT_BEDROCK_EMBEDDING_MODEL_ID)
    raise EmbeddingError(f"Unknown embedding provider: '{provider}' (expected 'bedrock' or 'mock').")
