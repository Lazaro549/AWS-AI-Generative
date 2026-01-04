# AWS Generative AI

![AWS Generative AI Architecture](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2024/09/05/ML-17360-logic-flow-1.png)

## 📌 Overview

This repository focuses on **Generative AI on AWS**, providing practical examples, reference architectures, and best practices to design, build, and scale **generative AI applications** using **Amazon Web Services (AWS)**.

It is intended for cloud architects, developers, and data teams looking to integrate **Generative AI** in a secure, scalable, and cost-efficient way.

---

## 🧠 What is Generative AI on AWS?

**Generative AI** enables the creation of new content such as text, images, code, audio, and more. AWS provides a comprehensive ecosystem for building generative AI solutions, including:

- **Amazon Bedrock** – Access to foundation models (FMs) from multiple providers
- **Amazon SageMaker** – Model training, fine-tuning, and deployment
- **AWS Lambda** – Serverless inference
- **Amazon API Gateway** – API exposure and management
- **Amazon OpenSearch** – Semantic search and RAG architectures
- **Amazon S3** – Data and prompt storage
- **AWS IAM & KMS** – Security, identity, and encryption

---

## 🏗️ Reference Architecture

The diagram above illustrates a typical logical flow for Generative AI applications on AWS, including:

1. User input (Web / App / API)
2. Request orchestration and validation
3. Context retrieval (RAG)
4. Foundation model inference
5. Post-processing and response delivery
6. Observability, security, and governance

---

## 📂 Repository Structure

```text
.
├── architectures/       # Architecture diagrams and patterns
├── examples/            # Hands-on Generative AI examples
├── notebooks/           # Jupyter Notebooks (SageMaker)
├── prompts/             # Reusable prompt templates
├── scripts/             # Automation and deployment scripts
├── terraform/           # Infrastructure as Code (IaC)
└── README.md
