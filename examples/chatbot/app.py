import json
import os
from pathlib import Path
from typing import List, Dict

import boto3
from dotenv import load_dotenv

load_dotenv()

_config_path = Path(__file__).parent / "config.json"
try:
    config = json.loads(_config_path.read_text())
except FileNotFoundError:
    config = {
        "region": os.getenv("AWS_REGION", "us-east-1"),
        "model_id": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
        "max_tokens": int(os.getenv("BEDROCK_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.3")),
    }

client = boto3.client("bedrock-runtime", region_name=config["region"])


def chat(prompt: str, history: List[Dict] | None = None) -> str:
    messages = list(history) if history else []
    messages.append({"role": "user", "content": prompt})

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "messages": messages,
    }

    try:
        response = client.invoke_model(modelId=config["model_id"], body=json.dumps(body))
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        print(f"Error calling Bedrock: {e}")
        raise


if __name__ == "__main__":
    print("🤖 Bedrock Chatbot (type 'exit' to quit)")
    history: List[Dict] = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        try:
            reply = chat(user_input, history)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            print(f"Bot: {reply}\n")
        except Exception:
            print("Bot: Sorry, something went wrong. Please try again.\n")
