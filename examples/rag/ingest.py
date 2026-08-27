from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Module-level variable that can be overridden for testing
DATA_DIR = Path(__file__).parent / "data"


def load_documents() -> str:
    """Load text documents from the data/ folder next to this file and return a single
    concatenated string suitable for context injection.

    If no documents are found, return an empty string.
    """
    data_dir = DATA_DIR
    texts: List[str] = []

    if not data_dir.exists() or not data_dir.is_dir():
        return ""

    for p in sorted(data_dir.iterdir()):
        if p.is_file():
            try:
                texts.append(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: could not read {p}: {e}")
                continue

    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Structured document loading (new retrieval pipeline)
# ---------------------------------------------------------------------------
#
# Everything below is ADDITIVE: load_documents() and DATA_DIR above are
# completely unchanged and keep powering the existing evaluation/ scripts
# and tests/test_rag.py exactly as before.
#
# This section adds a second, metadata-preserving loader used by the new
# chunking / embeddings / vector-index / retrieval pipeline (see
# retrieval.py). It reads from CORPUS_DIR, a *subdirectory* of DATA_DIR
# (examples/rag/data/documents/) rather than DATA_DIR itself, so it never
# changes what load_documents() returns: load_documents() iterates
# DATA_DIR non-recursively (see test_load_documents_with_subdirectories),
# so a nested "documents/" folder is invisible to it.

SUPPORTED_DOCUMENT_EXTENSIONS = (".md", ".txt")

CORPUS_DIR = DATA_DIR / "documents"


class DocumentLoadError(Exception):
    """Raised when a supported document file exists but cannot be read
    (e.g. it is not valid UTF-8 text). A file with an unsupported
    extension is silently skipped rather than treated as an error --
    see load_documents_structured().
    """


@dataclass(frozen=True)
class Document:
    """A single loaded source document, with metadata preserved for
    chunking, indexing, and source-aware retrieval results."""

    id: str
    source: str
    path: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_documents_structured(data_dir: Path | None = None) -> List[Document]:
    """Discover and load documents as structured Document objects,
    preserving source/path/id metadata for downstream chunking.

    Defaults to CORPUS_DIR (examples/rag/data/documents/). Unlike
    load_documents(), this:

    - only considers files with a supported extension
      (SUPPORTED_DOCUMENT_EXTENSIONS); other files are skipped, not
      treated as errors -- so a stray README or hidden file sitting in
      the corpus folder does not break loading;
    - raises DocumentLoadError for a matched file that cannot be
      decoded as UTF-8, instead of printing a warning and silently
      continuing, so a genuinely broken document is never silently
      dropped from the corpus;
    - returns [] for a missing or empty directory rather than raising,
      so "no documents yet" is a normal state -- see
      retrieval.build_index() for how that is surfaced to the caller.
    """
    directory = data_dir if data_dir is not None else CORPUS_DIR
    if not directory.exists() or not directory.is_dir():
        return []

    documents: List[Document] = []
    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(
                f"Could not read '{file_path.name}' as UTF-8 text."
            ) from exc
        except OSError as exc:
            raise DocumentLoadError(f"Could not read '{file_path.name}': {exc}") from exc

        if not text.strip():
            continue

        document_id = file_path.stem
        title = document_id.replace("_", " ").replace("-", " ").strip().title()
        documents.append(
            Document(
                id=document_id,
                source=file_path.name,
                path=str(file_path),
                title=title,
                text=text,
                metadata={"extension": file_path.suffix.lower()},
            )
        )
    return documents
