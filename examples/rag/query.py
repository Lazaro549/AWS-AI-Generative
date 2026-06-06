import json
import os
import boto3
from ingest import load_documents

with open("config.json") as f:
    config = json.load(f)

REGION = os.getenv("AWS_REGION", config["region"])
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", config["model_id"])

client = boto3.client(
    "bedrock-runtime",
    region_name=REGION
)

def ask(question: str) -> str:
    context = load_documents()

    prompt = f"""
You are a helpful assistant.
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = ask(question)
    print("\nAnswer:\n", answer)
