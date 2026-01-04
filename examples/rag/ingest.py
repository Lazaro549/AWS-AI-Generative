from pathlib import Path

DATA_DIR = Path("data")

def load_documents() -> str:
    documents = []
    for file in DATA_DIR.glob("*.txt"):
        documents.append(file.read_text())
    return "\n".join(documents)

if __name__ == "__main__":
    docs = load_documents()
    print("Documents loaded:\n")
    print(docs)
