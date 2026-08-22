# Simple RAG with Amazon Bedrock

This example demonstrates a **basic Retrieval-Augmented Generation (RAG)** flow
without external vector databases.

## Flow
1. Load documents
2. Inject context into the prompt
3. Generate answer using Bedrock

## Run
```bash
pip install boto3
python query.py
