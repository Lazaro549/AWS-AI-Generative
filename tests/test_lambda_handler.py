import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


LAMBDA_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lambda"
)


class FakeBody:
    def __init__(self, response_text="Hello from Bedrock"):
        self.response_text = response_text

    def read(self):
        return json.dumps(
            {"content": [{"text": self.response_text}]}
        ).encode()


class FakeBedrocClient:
    def __init__(self, response_text="Hello from Bedrock"):
        self.invoke_model_calls = []
        self.response_text = response_text

    def invoke_model(self, **kwargs):
        self.invoke_model_calls.append(kwargs)
        return {"body": FakeBody(self.response_text)}


@pytest.fixture
def lambda_handler_module(monkeypatch):
    """Load the Lambda handler module with mocked Bedrock client."""
    fake_bedrock = FakeBedrocClient()

    monkeypatch.setattr(
        "boto3.client",
        lambda service, **kwargs: fake_bedrock if service == "bedrock-runtime" else MagicMock(),
    )

    spec = importlib.util.spec_from_file_location(
        "lambda_handler",
        LAMBDA_DIR / "handler.py",
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    yield module, fake_bedrock

    sys.modules.pop("lambda_handler", None)


def test_lambda_handler_with_valid_prompt(lambda_handler_module):
    """Test Lambda handler with a valid prompt."""
    module, fake_bedrock = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "Hello, how are you?"})
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "response" in body
    assert body["response"] == "Hello from Bedrock"
    assert len(fake_bedrock.invoke_model_calls) == 1


def test_lambda_handler_with_missing_prompt(lambda_handler_module):
    """Test Lambda handler when prompt field is missing."""
    module, _ = lambda_handler_module

    event = {
        "body": json.dumps({})
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "error" in body


def test_lambda_handler_with_empty_prompt(lambda_handler_module):
    """Test Lambda handler with an empty prompt."""
    module, _ = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": ""})
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "error" in body


def test_lambda_handler_with_whitespace_only_prompt(lambda_handler_module):
    """Test Lambda handler with whitespace-only prompt."""
    module, _ = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "   \n\t  "})
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "error" in body


def test_lambda_handler_with_invalid_json(lambda_handler_module):
    """Test Lambda handler with malformed JSON in body."""
    module, _ = lambda_handler_module

    event = {
        "body": "Not valid JSON {]"
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "error" in body


def test_lambda_handler_with_null_body(lambda_handler_module):
    """Test Lambda handler when body is null."""
    module, _ = lambda_handler_module

    event = {
        "body": None
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "error" in body


def test_lambda_handler_response_structure(lambda_handler_module):
    """Test that Lambda handler response has correct structure."""
    module, _ = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "Test"})
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert "statusCode" in response
    assert "headers" in response
    assert "body" in response
    assert response["headers"]["Content-Type"] == "application/json"
    assert response["headers"]["Access-Control-Allow-Origin"] == "*"


def test_lambda_handler_includes_model_id(lambda_handler_module):
    """Test that Lambda handler returns the model ID used."""
    module, _ = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "Test"})
    }
    context = {}

    response = module.lambda_handler(event, context)

    body = json.loads(response["body"])
    assert "model" in body
    assert body["model"] == module.MODEL_ID


def test_lambda_handler_sends_correct_model_id(lambda_handler_module):
    """Test that Lambda handler sends correct model ID to Bedrock."""
    module, fake_bedrock = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "Test prompt"})
    }
    context = {}

    module.lambda_handler(event, context)

    request = fake_bedrock.invoke_model_calls[0]
    assert request["modelId"] == module.MODEL_ID


def test_lambda_handler_sends_correct_prompt(lambda_handler_module):
    """Test that Lambda handler sends prompt to Bedrock correctly."""
    module, fake_bedrock = lambda_handler_module

    prompt_text = "What is AWS?"
    event = {
        "body": json.dumps({"prompt": prompt_text})
    }
    context = {}

    module.lambda_handler(event, context)

    request = fake_bedrock.invoke_model_calls[0]
    body = json.loads(request["body"])
    assert body["messages"][0]["content"] == prompt_text


def test_lambda_handler_uses_max_tokens(lambda_handler_module):
    """Test that Lambda handler uses MAX_TOKENS configuration."""
    module, fake_bedrock = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "Test"})
    }
    context = {}

    module.lambda_handler(event, context)

    request = fake_bedrock.invoke_model_calls[0]
    body = json.loads(request["body"])
    assert body["max_tokens"] == module.MAX_TOKENS


def test_lambda_handler_uses_temperature(lambda_handler_module):
    """Test that Lambda handler uses TEMPERATURE configuration."""
    module, fake_bedrock = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "Test"})
    }
    context = {}

    module.lambda_handler(event, context)

    request = fake_bedrock.invoke_model_calls[0]
    body = json.loads(request["body"])
    assert body["temperature"] == module.TEMPERATURE


def test_lambda_handler_with_dict_body(lambda_handler_module):
    """Test Lambda handler when body is already a dict (not string)."""
    module, fake_bedrock = lambda_handler_module

    event = {
        "body": {"prompt": "Test"}  # Already a dict, not JSON string
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "response" in body


def test_lambda_handler_with_long_prompt(lambda_handler_module):
    """Test Lambda handler with a very long prompt."""
    module, fake_bedrock = lambda_handler_module

    long_prompt = "A" * 10000

    event = {
        "body": json.dumps({"prompt": long_prompt})
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["statusCode"] == 200
    request = fake_bedrock.invoke_model_calls[0]
    body = json.loads(request["body"])
    assert body["messages"][0]["content"] == long_prompt


def test_lambda_handler_with_special_characters(lambda_handler_module):
    """Test Lambda handler with special characters in prompt."""
    module, fake_bedrock = lambda_handler_module

    prompt = 'Test "quotes" and \\n newlines and émojis 🚀'

    event = {
        "body": json.dumps({"prompt": prompt})
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["statusCode"] == 200
    request = fake_bedrock.invoke_model_calls[0]
    body = json.loads(request["body"])
    assert prompt in body["messages"][0]["content"]


def test_lambda_handler_cors_headers(lambda_handler_module):
    """Test that Lambda handler includes CORS headers."""
    module, _ = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "Test"})
    }
    context = {}

    response = module.lambda_handler(event, context)

    assert response["headers"]["Access-Control-Allow-Origin"] == "*"
    assert response["headers"]["Content-Type"] == "application/json"


def test_lambda_handler_uses_correct_anthropic_version(lambda_handler_module):
    """Test that Lambda handler uses correct Anthropic version."""
    module, fake_bedrock = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "Test"})
    }
    context = {}

    module.lambda_handler(event, context)

    request = fake_bedrock.invoke_model_calls[0]
    body = json.loads(request["body"])
    assert body["anthropic_version"] == "bedrock-2023-05-31"


def test_lambda_handler_response_body_is_json_string(lambda_handler_module):
    """Test that Lambda handler response body is a valid JSON string."""
    module, _ = lambda_handler_module

    event = {
        "body": json.dumps({"prompt": "Test"})
    }
    context = {}

    response = module.lambda_handler(event, context)

    # Should not raise an exception
    body = json.loads(response["body"])
    assert isinstance(body, dict)


def test_lambda_handler_multiple_requests(lambda_handler_module):
    """Test Lambda handler with multiple sequential requests."""
    module, fake_bedrock = lambda_handler_module

    for i in range(3):
        event = {
            "body": json.dumps({"prompt": f"Test prompt {i}"})
        }
        context = {}

        response = module.lambda_handler(event, context)

        assert response["statusCode"] == 200

    assert len(fake_bedrock.invoke_model_calls) == 3
