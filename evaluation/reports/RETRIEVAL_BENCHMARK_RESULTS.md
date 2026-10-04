# Retrieval Benchmark Results

**Run ID:** `20261004T044141Z`  
**Timestamp (UTC):** `2026-10-04T04:41:41.005056+00:00`  
**Questions:** `18`  
**Corpus documents:** `6`  
**Embedding:** `MockEmbeddingClient (deterministic, hash-based, non-semantic, 256-dim)`  

## Method

Offline benchmark using the repository's real chunking → FAISS → top-k retrieval pipeline. Embeddings use `MockEmbeddingClient` (deterministic, hash-based, non-semantic) so the experiment requires no AWS credentials and is fully reproducible. Retrieval quality reflects pipeline mechanics (chunk granularity, top-k coverage) under controlled conditions, not semantic similarity.

## Results

| Config | Chunk size | Overlap | Top-k | Chunks | Precision | Recall | Avg latency (ms) | P95 latency (ms) | Build (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 400 | 60 | 2 | 39 | 0.139 | 0.250 | 0.16 | 0.47 | 15.7 |
| balanced | 800 | 120 | 3 | 21 | 0.278 | 0.667 | 0.14 | 0.24 | 12.2 |
| wide | 1200 | 180 | 4 | 14 | 0.171 | 0.556 | 0.25 | 0.45 | 10.8 |

## Finding

Best recall: `balanced` (recall=0.667). Recall change vs baseline `small`: +0.417 (+166.7%). Mean latency change: -13.1%.

## Reproduce

```bash
python evaluation/retrieval_benchmark.py
```

No AWS credentials required. Results are deterministic across runs.
