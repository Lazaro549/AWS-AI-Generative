# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines a retrieval system with a
foundation model: instead of relying only on what the model learned
during training, the application first retrieves relevant information
from an external source and includes it in the prompt, so the model
generates its answer grounded in that retrieved evidence.

## Why RAG

- **Reduces hallucination.** Answers can be grounded in retrieved text
  instead of relying purely on the model's internal, imperfect
  memory of facts.
- **Keeps knowledge current without retraining.** Updating the
  document corpus updates what the system can answer about,
  without any model fine-tuning.
- **Works with private or proprietary data.** Documents that were
  never part of a model's training data — internal wikis, product
  docs, this repository's own `.md` files — can still ground answers.
- **Source attribution.** Because retrieval returns specific chunks
  with their originating document, an application can show users
  which sources an answer relied on.

## A realistic RAG pipeline

1. **Document loading** — read source documents and preserve metadata
   (source file, document ID) needed later for attribution.
2. **Chunking** — split documents into overlapping, manageably sized
   pieces. Chunks that are too large dilute similarity search; chunks
   that are too small lose context, so chunk size and overlap are
   tuned per corpus.
3. **Embeddings** — convert each chunk (and, later, each query) into a
   numeric vector that captures its semantic meaning, using an
   embedding model such as Amazon Titan Embeddings on Bedrock.
4. **Vector index** — store chunk embeddings in a structure that
   supports fast similarity search, such as a FAISS index for a local,
   single-machine corpus.
5. **Similarity search / top-k retrieval** — embed the incoming
   question, compare it against every indexed chunk (commonly with
   cosine similarity), and keep only the top-k most similar chunks —
   not the entire corpus.
6. **Context construction** — assemble the retrieved chunks (often
   labeled with their source) into a context block for the prompt,
   with instructions telling the model to answer from that context and
   to say so explicitly when the context is insufficient.
7. **Generation** — send the question plus the constructed context to
   a foundation model (Amazon Bedrock, in this repository) to produce
   the final answer.

## Evaluating RAG quality

Retrieval and generation are usually evaluated separately: **context
precision** and **context recall** measure whether retrieval found the
right (and only the right) evidence, while **faithfulness** and
**answer relevancy** measure whether the generated answer is actually
grounded in that evidence and addresses the question. This repository's
`evaluation/` framework implements exactly these metrics.
