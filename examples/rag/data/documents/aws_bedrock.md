# Amazon Bedrock

Amazon Bedrock is a fully managed AWS service that provides access to
foundation models (FMs) from multiple leading AI providers — including
Anthropic, Meta, Mistral, Cohere, Stability AI, and Amazon's own Titan
models — through a single, unified API. Instead of hosting and managing
model infrastructure, developers call Bedrock's API to run inference
against the model that best fits their use case.

## Key characteristics

- **Serverless inference.** There are no clusters or instances to
  provision; Bedrock scales automatically with request volume.
- **Model choice.** Applications can switch between providers/models
  through configuration rather than re-architecting, which makes it
  easy to compare quality, latency, and cost across models.
- **AWS-native security.** Requests run inside your AWS account and can
  be governed with IAM policies, encrypted with KMS, monitored with
  CloudWatch, and kept within your existing VPC boundary.
- **Knowledge Bases for Amazon Bedrock.** A managed capability that
  handles document ingestion, chunking, embedding, and vector storage
  for Retrieval-Augmented Generation, so teams can build RAG
  applications without operating a separate vector database — though,
  as this repository demonstrates, it is also possible to build the
  same pipeline explicitly for full control over each stage.
- **Guardrails and Agents.** Additional Bedrock features let teams
  apply content filters and policy checks, and orchestrate multi-step,
  tool-using agents on top of the same foundation models.

## Common use cases

Bedrock is commonly used for chatbots and virtual assistants, text
generation and summarization, question answering, and — combined with
retrieval — Retrieval-Augmented Generation (RAG), where external
documents are supplied as context so the model can answer accurately
about information it was not trained on.

## How this repository uses it

The `examples/chatbot/` example calls Bedrock directly for single-turn
generation. The `examples/rag/` example goes further: it retrieves
relevant chunks from a local document corpus and passes only that
retrieved context to Bedrock, rather than sending everything Bedrock
might need to know in every request.
