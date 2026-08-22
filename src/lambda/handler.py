import json
import os

import boto3


MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-sonnet-20240229-v1:0",
)
MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("BEDROCK_TEMPERATURE", "0.3"))

bedrock = boto3.client("bedrock-runtime")


def response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    try:
        raw_body = event.get("body") or "{}"

        if isinstance(raw_body, str):
            body = json.loads(raw_body)
        else:
            body = raw_body

        prompt = body.get("prompt", "").strip()

        if not prompt:
            return response(
                400,
                {
                    "error": "The 'prompt' field is required."
                },
            )

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        result = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_body),
        )

        model_response = json.loads(
            result["body"].read()
        )

        generated_text = model_response["content"][0]["text"]

        return response(
            200,
            {
                "response": generated_text,
                "model": MODEL_ID,
            },
        )

    except json.JSONDecodeError:
        return response(
            400,
            {
                "error": "Invalid JSON body."
            },
        )

    except Exception as exc:
        print(f"ERROR: {exc}")

        return response(
            500,
            {
                "error": "Internal server error."
            },
        )
