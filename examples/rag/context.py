"""Context and prompt construction for the RAG pipeline.

Turns ranked retrieval results into a single, source-annotated context
block, and combines that with the repository's existing RAG prompt
template (prompts/rag.txt) to build the exact string sent to Bedrock.

prompts/rag.txt already instructs the model to answer only from the
supplied context and to say so explicitly when the context is
insufficient ("I don't have enough information to answer this
question."), which is exactly what this pipeline needs -- so it is
reused here rather than duplicated. If that file is ever missing, a
fallback string with identical wording is used instead, so context
construction never hard-fails on a packaging/deployment issue.
"""

import sys as _sys
from pathlib import Path as _Path

_THIS_DIR = _Path(__file__).resolve().parent
if str(_THIS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_THIS_DIR))

from typing import Sequence

from retrieval import RetrievalResult

REPO_ROOT = _THIS_DIR.parents[1]
PROMPT_TEMPLATE_PATH = REPO_ROOT / "prompts" / "rag.txt"

# Mirrors prompts/rag.txt exactly; used only if that file cannot be read.
_FALLBACK_TEMPLATE = (
    "You are an AI assistant that answers questions using the provided context.\n\n"
    "Use ONLY the information in the context below.\n"
    "If the answer is not contained in the context, say:\n"
    '"I don\'t have enough information to answer this question."\n\n'
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
)


def _load_template() -> str:
    try:
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_TEMPLATE


def assemble_context(results: Sequence[RetrievalResult]) -> str:
    """Join ranked retrieval results into one context block, most
    relevant first, each chunk labeled with the document it came from
    so the model (and a reader) can see where each piece of context
    was sourced."""
    if not results:
        return ""
    blocks = [f"[Source: {result.source}]\n{result.text}" for result in results]
    return "\n\n".join(blocks)


def build_prompt(question: str, context_text: str) -> str:
    """Fill the RAG prompt template with ``context_text`` and
    ``question``.

    Uses plain substring replacement rather than str.format(), so that
    a curly brace appearing in real document content or in a user's
    question is never misread as a template placeholder.
    """
    template = _load_template()
    return template.replace("{context}", context_text).replace("{question}", question)
