import json
import boto3
from ingest import load_documents

REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

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
        "max_tokens": 512,
        "temperature": 0.3,
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
