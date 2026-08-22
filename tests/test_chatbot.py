import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


CHATBOT_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "chatbot"
)


class FakeBody:
    def read(self):
        return json.dumps(
            {"content": [{"text": "Hello from Bedrock"}]}
        ).encode()


class FakeClient:
    def __init__(self):
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": FakeBody()}


@pytest.fixture
def chatbot_module(monkeypatch):
    fake_client = FakeClient()

    monkeypatch.setattr(
        "boto3.client",
        lambda *args, **kwargs: fake_client,
    )

    original_cwd = Path.cwd()

    try:
        os.chdir(CHATBOT_DIR)

        spec = importlib.util.spec_from_file_location(
            "chatbot_app",
            CHATBOT_DIR / "app.py",
        )

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        yield module, fake_client

    finally:
        os.chdir(original_cwd)
        sys.modules.pop("chatbot_app", None)


def test_chat_returns_model_text(chatbot_module):
    module, _ = chatbot_module

    result = module.chat("Hello")

    assert result == "Hello from Bedrock"


def test_chat_invokes_bedrock(chatbot_module):
    module, fake_client = chatbot_module

    module.chat("Test prompt")

    assert len(fake_client.calls) == 1


def test_chat_sends_correct_model(chatbot_module):
    module, fake_client = chatbot_module

    module.chat("Test prompt")

    request = fake_client.calls[0]

    assert request["modelId"] == module.config["model_id"]


def test_chat_sends_correct_prompt(chatbot_module):
    module, fake_client = chatbot_module

    module.chat("Test prompt")

    request = fake_client.calls[0]
    body = json.loads(request["body"])

    assert body["messages"] == [
        {
            "role": "user",
            "content": "Test prompt",
        }
    ]


def test_chat_uses_config_values(chatbot_module):
    module, fake_client = chatbot_module

    module.chat("Test prompt")

    request = fake_client.calls[0]
    body = json.loads(request["body"])

    assert body["max_tokens"] == module.config["max_tokens"]
    assert body["temperature"] == module.config["temperature"]


def test_chat_with_empty_prompt(chatbot_module):
    """Test that empty prompts are handled."""
    module, fake_client = chatbot_module

    result = module.chat("")

    # Should still invoke Bedrock (empty message handling is Bedrock's responsibility)
    assert len(fake_client.calls) == 1


def test_chat_with_long_prompt(chatbot_module):
    """Test that long prompts are properly handled."""
    module, fake_client = chatbot_module

    long_prompt = "A" * 5000

    result = module.chat(long_prompt)

    assert result == "Hello from Bedrock"
    request = fake_client.calls[0]
    body = json.loads(request["body"])
    assert body["messages"][0]["content"] == long_prompt


def test_chat_with_special_characters(chatbot_module):
    """Test that special characters in prompts are properly escaped."""
    module, fake_client = chatbot_module

    special_prompt = 'Hello "world" with \\n newline and émojis 🚀'

    result = module.chat(special_prompt)

    assert len(fake_client.calls) == 1
    request = fake_client.calls[0]
    body = json.loads(request["body"])
    assert special_prompt in body["messages"][0]["content"]


def test_chat_multiple_calls(chatbot_module):
    """Test that multiple chat calls work independently."""
    module, fake_client = chatbot_module

    module.chat("First prompt")
    module.chat("Second prompt")
    module.chat("Third prompt")

    assert len(fake_client.calls) == 3
    assert fake_client.calls[0]["body"] != fake_client.calls[1]["body"]


def test_chat_response_format(chatbot_module):
    """Test that response is a string."""
    module, fake_client = chatbot_module

    result = module.chat("Test")

    assert isinstance(result, str)
    assert len(result) > 0


class FakeBodyError(FakeBody):
    """Mock body that returns malformed JSON."""
    def read(self):
        return b"Not valid JSON"


@pytest.fixture
def chatbot_module_with_error(monkeypatch):
    """Fixture that returns an error response from Bedrock."""
    class FakeClientError(FakeClient):
        def invoke_model(self, **kwargs):
            self.calls.append(kwargs)
            return {"body": FakeBodyError()}

    fake_client = FakeClientError()

    monkeypatch.setattr(
        "boto3.client",
        lambda *args, **kwargs: fake_client,
    )

    original_cwd = Path.cwd()

    try:
        os.chdir(CHATBOT_DIR)

        spec = importlib.util.spec_from_file_location(
            "chatbot_app_error",
            CHATBOT_DIR / "app.py",
        )

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        yield module, fake_client

    finally:
        os.chdir(original_cwd)
        sys.modules.pop("chatbot_app_error", None)


def test_chat_with_malformed_response(chatbot_module_with_error):
    """Test behavior when Bedrock returns malformed JSON."""
    module, fake_client = chatbot_module_with_error

    with pytest.raises(Exception):
        module.chat("Test prompt")


def test_chat_uses_correct_anthropic_version(chatbot_module):
    """Test that the correct Anthropic version is used."""
    module, fake_client = chatbot_module

    module.chat("Test")

    request = fake_client.calls[0]
    body = json.loads(request["body"])

    assert body["anthropic_version"] == "bedrock-2023-05-31"


def test_config_has_required_fields(chatbot_module):
    """Test that config has all required fields."""
    module, _ = chatbot_module

    assert "region" in module.config
    assert "model_id" in module.config
    assert "max_tokens" in module.config
    assert "temperature" in module.config
