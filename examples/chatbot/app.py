import json
import boto3

# Load config
with open("config.json") as f:
    config = json.load(f)

client = boto3.client(
    "bedrock-runtime",
    region_name=config["region"]
)

def chat(prompt: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = client.invoke_model(
        modelId=config["model_id"],
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

if __name__ == "__main__":
    print("🤖 Bedrock Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        reply = chat(user_input)
        print(f"Bot: {reply}\n")
