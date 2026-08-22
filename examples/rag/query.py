import json
import os
from functools import lru_cache
from pathlib import Path

import boto3
from dotenv import load_dotenv
from ingest import load_documents

load_dotenv()

_config_path = Path(__file__).parent / "config.json"
try:
    config = json.loads(_config_path.read_text())
except FileNotFoundError:
    config = {}

REGION = os.getenv("AWS_REGION", config.get("region", "us-east-1"))
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0"))
MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", config.get("max_tokens", "512")))
TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", config.get("temperature", "0.3")))

client = boto3.client("bedrock-runtime", region_name=REGION)


@lru_cache(maxsize=1)
def _cached_documents() -> str:
    return load_documents()


def ask(question: str, _docs: str | None = None) -> str:
    context = _docs if _docs is not None else _cached_documents()

    prompt = f"""You are a helpful assistant.
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        print(f"Error calling Bedrock: {e}")
        raise


if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = ask(question)
    print("\nAnswer:\n", answer)
