# AWS Generative AI

![AWS Generative AI Architecture](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2024/09/05/ML-17360-logic-flow-1.png)

## 📌 Overview

This repository provides **simple, practical examples of Generative AI on AWS**
using **Amazon Bedrock**.  
It is designed for **learning, demos, and proof-of-concepts**, keeping the codebase
minimal, clear, and easy to extend.

The focus is on:
- Chatbots
- Retrieval-Augmented Generation (RAG)
- Prompt engineering
- Hands-on experimentation with Bedrock foundation models

---

## 🧠 What is Generative AI on AWS?

**Generative AI** enables applications to create new content such as text, summaries,
and conversational responses.

AWS supports Generative AI workloads through services like:
- **Amazon Bedrock** – Serverless access to foundation models
- **Amazon SageMaker** – ML experimentation and training
- **AWS Lambda & API Gateway** – Serverless APIs
- **Amazon S3** – Document storage
- **AWS IAM & KMS** – Security and encryption

This repository focuses primarily on **Amazon Bedrock**.

---

## 🏗️ Reference Architecture

The diagram above represents a typical Generative AI flow on AWS:

1. User input (CLI / Notebook / App)
2. Prompt orchestration
3. Optional context injection (RAG)
4. Foundation model inference via Amazon Bedrock
5. Response generation
6. Security, monitoring, and governance

---

## 📂 Repository Structure

```text
.
├── examples/          # Simple Generative AI examples
│   ├── chatbot/       # Bedrock chatbot
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
├── .gitignore
├── LICENSE
└── README.md

