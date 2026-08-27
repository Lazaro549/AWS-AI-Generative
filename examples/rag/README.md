# RAG with Amazon Bedrock

A Retrieval-Augmented Generation (RAG) example with a real retrieval
pipeline: documents are chunked, embedded, indexed in a local FAISS
vector index, and searched for the top-k most relevant chunks before
those (and only those) are sent to Amazon Bedrock as context.

## Architecture

```
Documents (examples/rag/data/documents/*.md)
        |
        v
Document loading  ---------------------------  ingest.load_documents_structured()
        |
        v
Chunking  ------------------------------------  chunking.chunk_document()
        |
        v
Embeddings  -----------------------------------  embeddings.BedrockEmbeddingClient
        |                                        (Amazon Titan Embeddings)
        v
Vector index (FAISS)  --------------------------  vector_index.VectorIndex
        |
        v
Similarity search / top-k retrieval  -----------  retrieval.retrieve()
        |
        v
Context construction (source-labeled)  ---------  context.assemble_context()
        |
        v
Prompt (prompts/rag.txt) + question  -----------  context.build_prompt()
        |
        v
Amazon Bedrock (generation)  -------------------  query.ask()
        |
        v
Answer
```

Indexing (embedding the corpus) and querying (embedding one question
and searching) are two separate steps, the way a real RAG deployment
separates offline ingestion from online serving:

- **`python build_index.py`** builds/rebuilds the index from
  `data/documents/*.md`. Re-run it after adding, editing, or removing
  documents there.
- **`python query.py`** answers a question. If no index exists yet, it
  builds one automatically the first time (a one-time cost); if AWS
  credentials or Bedrock access are not available, it falls back to the
  original, simpler behaviour of this example -- concatenating every
  document under `data/` -- so the example still works out of the box.

## Two document locations, on purpose

- `data/sample.txt` -- the original, single-document corpus.
  `ingest.load_documents()` still reads only this file (unchanged), and
  the `evaluation/` framework and this repo's existing tests are
  written against exactly that behaviour, so it is left as-is.
- `data/documents/*.md` -- the new, six-document corpus (Bedrock,
  Lambda, S3, IAM, SAM, and RAG itself) that the chunking / embeddings
  / vector-index / retrieval pipeline above actually indexes and
  searches. It lives in a subdirectory of `data/`, which
  `load_documents()` never sees (it only reads `data/`'s top-level
  files, non-recursively) -- see `ingest.py` for details.

## Run

```bash
pip install -e ".[rag]"       # adds faiss-cpu + numpy for vector search
cp ../../.env.example ../../.env   # then fill in your own values
python query.py
```

```
No vector index found -- building one now (one-time step)...
Embedding 21 chunk(s) from 6 document(s)...
Saved vector index (21 chunk(s)) to .../.rag_index
Ask a question: What does AWS Lambda do?

Retrieved context:
1. aws_lambda.md — score: 0.83
2. aws_sam.md — score: 0.41
3. aws_bedrock.md — score: 0.22

Answer:
 AWS Lambda is a serverless, event-driven compute service...
```

To force a rebuild (e.g. after editing a document), run
`python build_index.py` directly, or delete `.rag_index/`.

## Configuration

Same reuse pattern as the rest of this repository: an environment
variable, if set, overrides `config.json`, which overrides a
hard-coded default.

| Setting               | Env var                     | config.json key      | Default                          |
|------------------------|------------------------------|-----------------------|------------------------------------|
| AWS region              | `AWS_REGION`                 | `region`               | `us-east-1`                        |
| Generation model         | `BEDROCK_MODEL_ID`           | `model_id`             | `anthropic.claude-3-sonnet-20240229-v1:0` |
| Max tokens                | `BEDROCK_MAX_TOKENS`         | `max_tokens`           | `512`                              |
| Temperature                 | `BEDROCK_TEMPERATURE`        | `temperature`          | `0.3`                              |
| Embedding model              | `BEDROCK_EMBEDDING_MODEL_ID` | `embedding_model_id`   | `amazon.titan-embed-text-v2:0`     |
| Chunk size (characters)        | `RAG_CHUNK_SIZE`             | `chunk_size`           | `800`                              |
| Chunk overlap (characters)       | `RAG_CHUNK_OVERLAP`          | `chunk_overlap`        | `120`                              |
| Top-k                              | `RAG_TOP_K`                  | `top_k`                | `3`                                 |
| Index directory                      | --                            | `index_dir`             | `.rag_index` (git-ignored, rebuildable) |

## Error handling

- **Missing AWS credentials / Bedrock access** -- embedding and
  generation calls raise a clear `EmbeddingError` (embeddings.py) or
  print a one-line Bedrock error (query.py) instead of a raw
  traceback; the CLI falls back to full-corpus context rather than
  crashing.
- **Missing vector index** -- treated as a normal, expected state
  (`retrieval.load_index()` returns `None`), not an error.
- **Empty corpus** -- `build_index()` raises a clear `RetrievalError`
  naming the expected document directory.
- **Invalid document file** -- a matched (`.md`/`.txt`) file that is
  not valid UTF-8 raises `DocumentLoadError`; an unsupported file type
  is simply skipped, not treated as an error.
- **Empty question** -- rejected clearly by both `query.ask()` and
  `retrieval.retrieve()`, instead of embedding an empty string.

## Tests

```bash
pytest tests/test_rag.py tests/test_rag_pipeline.py -v
```

`tests/test_rag_pipeline.py` covers chunking, the mock embedding
client, the vector index, retrieval, and context/prompt construction
-- all with the deterministic `MockEmbeddingClient`, so no AWS
credentials are required.
