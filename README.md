# AWS AI Generative
![](assets/logo.png)


## 📌 Overview

This repository contains **simple and practical examples of Generative AI on AWS**
using **Amazon Bedrock**.

It is designed for:
- Learning and experimentation
- Demos and workshops
- Proofs of concept (PoCs)

The repository keeps everything **minimal, readable, and easy to extend**, focusing on
core Generative AI patterns such as chatbots and Retrieval-Augmented Generation (RAG).

---

## 🧠 Generative AI on AWS

**Generative AI** enables applications to generate new content such as text,
summaries, and conversational responses.

AWS provides managed services to build Generative AI solutions securely and at scale,
including:

- **Amazon Bedrock** – Serverless access to foundation models
- **Amazon SageMaker** – ML experimentation and training
- **AWS Lambda & API Gateway** – Serverless APIs
- **Amazon S3** – Document storage
- **AWS IAM & KMS** – Security and encryption

This repository focuses primarily on **Amazon Bedrock**.

---

## 🏗️ Reference Architecture

A typical Generative AI flow on AWS includes:

1. User input (CLI / Notebook / Application)
2. Prompt orchestration
3. Optional context injection (RAG)
4. Model inference via Amazon Bedrock
5. Response generation
6. Security, monitoring, and governance

---

## ✅ Prerequisites

- Python 3.10+
- An AWS account with access to [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- Bedrock model access enabled in your region ([request access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html))
- AWS credentials configured (`aws configure` or `~/.aws/credentials`)

---

## 🚀 Getting Started

**1. Clone and install dependencies**
```bash
git clone https://github.com/Lazaro549/aws-ai-generative.git
cd aws-ai-generative
pip install .
```

**2. Configure your environment**
```bash
cp .env.example .env
```
Edit `.env` with your values:
```env
AWS_REGION=us-east-1
AWS_PROFILE=default
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

**3. Verify Bedrock access**
```bash
python scripts/check_bedrock_access.py
```

**4. Run an example**
```bash
# Chatbot
python examples/chatbot/app.py

# RAG
python examples/rag/query.py
```

---

## 📂 Repository Structure

```text
.
├── examples/          # Simple Generative AI examples
│   ├── chatbot/       # Amazon Bedrock chatbot
│   └── rag/           # Basic RAG implementation
│
├── notebooks/         # Jupyter notebooks for experimentation
│
├── prompts/           # Reusable prompt templates
│
├── scripts/           # Helper scripts (setup, validation)
│
├── docs/              # Lightweight documentation
│
├── .env.example       # Environment variable template
├── pyproject.toml     # Project dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---
## 💸 Donations

If you'd like to support this project:

- 🇦🇷 ARS (Argentina)  
  Alias: `lazaro.503.alaba.mp`

- 🌎 USD (Argentina only, local transfers)  
  Alias: `ahogada.duras.foca`

