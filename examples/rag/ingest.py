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
