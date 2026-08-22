"""Tests for environment configuration and .env support."""
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_env_example_file_exists():
    """Test that .env.example file exists."""
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    assert env_example.exists(), ".env.example file should exist"


def test_env_example_file_format():
    """Test that .env.example has valid format."""
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    
    with open(env_example, "r") as f:
        content = f.read()
    
    # Check for required environment variables
    required_vars = [
        "AWS_REGION",
        "AWS_PROFILE",
        "BEDROCK_MODEL_ID",
        "BEDROCK_MAX_TOKENS",
        "BEDROCK_TEMPERATURE",
    ]
    
    for var in required_vars:
        assert var in content, f"{var} should be documented in .env.example"


def test_env_example_has_defaults():
    """Test that .env.example provides default values."""
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    
    with open(env_example, "r") as f:
        content = f.read()
    
    # Check for some default values
    assert "us-east-1" in content, "Should have a default AWS region"
    assert "bedrock" in content.lower(), "Should reference Bedrock"


def test_gitignore_excludes_env_file():
    """Test that .gitignore excludes .env files."""
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    
    assert gitignore.exists(), ".gitignore file should exist"
    
    with open(gitignore, "r") as f:
        content = f.read()
    
    assert ".env" in content, ".gitignore should exclude .env files"


def test_gitignore_excludes_aws_credentials():
    """Test that .gitignore excludes AWS credentials."""
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    
    with open(gitignore, "r") as f:
        content = f.read()
    
    assert ".aws" in content, ".gitignore should exclude .aws directory"
    assert "credentials" in content, ".gitignore should exclude credentials"


def test_project_toml_includes_python_dotenv():
    """Test that pyproject.toml includes python-dotenv dependency."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    
    with open(pyproject, "r") as f:
        content = f.read()
    
    assert "python-dotenv" in content, "python-dotenv should be in dependencies"


def test_env_variable_parsing():
    """Test that environment variables can be parsed correctly."""
    test_env_vars = {
        "AWS_REGION": "us-west-2",
        "BEDROCK_MAX_TOKENS": "1024",
        "BEDROCK_TEMPERATURE": "0.5",
    }
    
    with patch.dict(os.environ, test_env_vars):
        # Simulate environment loading
        region = os.getenv("AWS_REGION", "us-east-1")
        max_tokens = int(os.getenv("BEDROCK_MAX_TOKENS", "512"))
        temperature = float(os.getenv("BEDROCK_TEMPERATURE", "0.3"))
        
        assert region == "us-west-2"
        assert max_tokens == 1024
        assert temperature == 0.5


def test_env_fallback_to_defaults():
    """Test that environment variables fall back to defaults."""
    # Clear environment variables
    env_backup = {
        "AWS_REGION": os.environ.get("AWS_REGION"),
        "BEDROCK_MAX_TOKENS": os.environ.get("BEDROCK_MAX_TOKENS"),
    }
    
    with patch.dict(os.environ, {}, clear=False):
        for key in ["AWS_REGION", "BEDROCK_MAX_TOKENS"]:
            os.environ.pop(key, None)
        
        region = os.getenv("AWS_REGION", "us-east-1")
        max_tokens = int(os.getenv("BEDROCK_MAX_TOKENS", "512"))
        
        assert region == "us-east-1"
        assert max_tokens == 512
    
    # Restore
    for key, value in env_backup.items():
        if value:
            os.environ[key] = value


def test_integer_env_variable_conversion():
    """Test that string environment variables are converted to integers."""
    with patch.dict(os.environ, {"BEDROCK_MAX_TOKENS": "2048"}):
        value = int(os.getenv("BEDROCK_MAX_TOKENS", "512"))
        assert value == 2048
        assert isinstance(value, int)


def test_float_env_variable_conversion():
    """Test that string environment variables are converted to floats."""
    with patch.dict(os.environ, {"BEDROCK_TEMPERATURE": "0.7"}):
        value = float(os.getenv("BEDROCK_TEMPERATURE", "0.3"))
        assert value == 0.7
        assert isinstance(value, float)


def test_env_variable_with_invalid_integer():
    """Test handling of invalid integer environment variables."""
    with patch.dict(os.environ, {"BEDROCK_MAX_TOKENS": "not_a_number"}):
        with pytest.raises(ValueError):
            int(os.getenv("BEDROCK_MAX_TOKENS", "512"))


def test_env_variable_with_invalid_float():
    """Test handling of invalid float environment variables."""
    with patch.dict(os.environ, {"BEDROCK_TEMPERATURE": "not_a_number"}):
        with pytest.raises(ValueError):
            float(os.getenv("BEDROCK_TEMPERATURE", "0.3"))


def test_aws_profile_env_variable():
    """Test AWS_PROFILE environment variable."""
    with patch.dict(os.environ, {"AWS_PROFILE": "production"}):
        profile = os.getenv("AWS_PROFILE", "default")
        assert profile == "production"


def test_bedrock_model_id_env_variable():
    """Test BEDROCK_MODEL_ID environment variable."""
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    with patch.dict(os.environ, {"BEDROCK_MODEL_ID": model_id}):
        loaded_model = os.getenv("BEDROCK_MODEL_ID")
        assert loaded_model == model_id


def test_log_level_env_variable():
    """Test LOG_LEVEL environment variable."""
    with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
        log_level = os.getenv("LOG_LEVEL", "INFO")
        assert log_level == "DEBUG"


def test_debug_flag_env_variable():
    """Test DEBUG environment variable."""
    with patch.dict(os.environ, {"DEBUG": "True"}):
        debug = os.getenv("DEBUG", "False")
        assert debug == "True"
