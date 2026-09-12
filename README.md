![AWS AI Generative](logo.png)

# AWS AI Generative

![Tests](https://github.com/Lazaro549/AWS-AI-Generative/actions/workflows/test.yml/badge.svg)
![Deploy](https://github.com/Lazaro549/AWS-AI-Generative/actions/workflows/deploy.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

Simple and practical examples of Generative AI on AWS using Amazon Bedrock —
from a minimal chatbot and a from-scratch RAG pipeline to a serverless Lambda
API and a production-quality evaluation framework for measuring RAG quality
objectively.

## 📁 Repository structure

```
AWS-AI-Generative/
├── examples/
│   ├── chatbot/        # Minimal Bedrock chatbot (app.py)
│   └── rag/             # RAG: chunking, Bedrock embeddings, local FAISS vector index
├── evaluation/           # Objective evaluation framework for the RAG example
│   ├── metrics.py        # Context precision/recall, faithfulness, relevancy...
│   ├── evaluate_retrieval.py
│   ├── evaluate_generation.py
│   ├── dataset/          # Evaluation question/answer dataset
│   └── reports/          # Generated JSON evaluation reports (git-ignored)
├── src/lambda/            # AWS Lambda handler exposing Bedrock via an API
├── infrastructure/        # AWS SAM template (serverless deployment)
├── notebooks/              # Jupyter notebooks: text generation, chatbot
├── prompts/                 # Reusable prompt templates
├── scripts/                  # Setup / validation helper scripts
├── docs/                      # Getting started + best practices guides
├── tests/                      # Unit + integration tests (pytest)
└── certifications/               # Supporting AWS AI certification materials
```

## 🚀 Getting started

### Prerequisites
- Python 3.10+
- An AWS account with access to [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- AWS credentials configured locally (`aws configure`, `~/.aws/credentials`, or `AWS_PROFILE`)

### Install

```bash
git clone https://github.com/Lazaro549/AWS-AI-Generative.git
cd AWS-AI-Generative
pip install -e ".[dev]"
cp .env.example .env   # then fill in your own values
```

### Run an example

```bash
python examples/chatbot/app.py
python examples/rag/query.py
```

Each example also falls back to its own `config.json` when an environment
variable isn't set — see [`docs/getting-started.md`](docs/getting-started.md).

## 🧪 Tests

```bash
pytest -v
```

The suite covers the chatbot, the RAG pipeline, the Lambda handler,
environment configuration, and the evaluation framework — all AWS calls are
mocked, so no credentials are required to run it.

## 📊 Evaluation framework

[`evaluation/`](evaluation/) is an independent, objective evaluation
framework for the RAG example: context precision, context recall,
faithfulness, answer relevancy, retrieval and end-to-end latency, and cost
per query. See [`evaluation/README.md`](evaluation/README.md) for
methodology, dataset format, and how to run it.

```bash
python evaluation/evaluate_retrieval.py    # no AWS access needed
python evaluation/evaluate_generation.py   # needs Bedrock access
python evaluation/rag_benchmark.py          # live multi-document benchmark
```

## 🏁 Live RAG benchmark

`evaluation/rag_benchmark.py` is the canonical benchmark for the multi-document RAG pipeline. It compares three retrieval configurations (chunk size, overlap, and top-k) across two Amazon Bedrock models, using the same 18-question dataset. Each run measures context precision/recall, faithfulness, answer F1, end-to-end latency, success rate, token usage, and optional cost/query.

The benchmark only publishes measured results from live Bedrock calls. It never turns failed calls into zeroes or invents pricing. The reproducible workflow in `.github/benchmark.yml` can run the benchmark with an AWS OIDC role stored as `AWS_BENCHMARK_ROLE_ARN` and commit the resulting JSON/Markdown report to `evaluation/reports/`.

```bash
python evaluation/rag_benchmark.py
```

## ☁️ Deployment

[`infrastructure/template.yaml`](infrastructure/template.yaml) is an AWS SAM
template that deploys the Lambda handler as a serverless API backed by
Bedrock. The `deploy` workflow in
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) builds and
deploys it on every push to `main`, after the test suite passes.

## 📄 License

[MIT](LICENSE)

## 💸 Donations

If you'd like to support this project:

- 🇦🇷 ARS (Argentina)  
  Alias: `lazaro.503.alaba.mp`

- 🌎 USD (Argentina only, local transfers)  
  Alias: `ahogada.duras.foca`
