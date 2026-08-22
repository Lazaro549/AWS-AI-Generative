# Tests

This directory contains comprehensive test suites for the AWS AI Generative project.

## Test Structure

```
tests/
├── test_chatbot.py           # Chatbot example tests
├── test_rag.py               # RAG (Retrieval-Augmented Generation) tests
├── test_lambda_handler.py    # AWS Lambda handler tests
├── test_env_config.py        # Environment configuration tests
└── README.md                 # This file
```

## Running Tests

### Install test dependencies

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest
```

### Run with verbose output

```bash
pytest -v
```

### Run specific test file

```bash
pytest tests/test_chatbot.py -v
```

### Run specific test

```bash
pytest tests/test_chatbot.py::test_chat_returns_model_text -v
```

### Run with coverage report

```bash
pip install pytest-cov
pytest --cov=. --cov-report=html
```

### Run tests by marker

```bash
# Run only unit tests
pytest -m "not slow"

# Run specific marker tests
pytest -m "env"
```

## Test Coverage

### test_chatbot.py
- **5 basic tests** (original)
- **+8 new tests** for comprehensive coverage:
  - Empty and long prompts
  - Special characters handling
  - Multiple consecutive calls
  - Response format validation
  - Error handling (malformed responses)
  - Configuration validation
  - Anthropic API version verification

**Total: 13 tests**

### test_rag.py
- **8 basic tests** (original)
- **+10 new tests** for comprehensive coverage:
  - Document loading with different encodings (UTF-8)
  - Subdirectory handling
  - Sorted document ordering
  - Very long questions
  - Special characters in prompts
  - Parameter validation (max_tokens, temperature)
  - Prompt structure verification

**Total: 18 tests**

### test_lambda_handler.py (NEW)
- **25 comprehensive tests** for Lambda handler:
  - Valid prompt handling
  - Missing/empty/whitespace prompts
  - Invalid JSON body
  - Null body handling
  - Response structure validation
  - Model ID and prompt verification
  - Configuration usage (max_tokens, temperature)
  - Dict vs string body handling
  - Long prompts and special characters
  - CORS headers
  - Anthropic API version
  - Multiple sequential requests

**Total: 25 tests**

### test_env_config.py (NEW)
- **20+ tests** for environment configuration:
  - .env.example file existence and format
  - .gitignore security checks
  - pyproject.toml dependency verification
  - Environment variable parsing
  - Type conversions (int, float)
  - Default value fallbacks
  - Invalid value handling
  - AWS profile configuration
  - Model ID and logging configuration

**Total: 20+ tests**

## Total Test Count

- **Original tests**: 13 tests (5 + 8)
- **New tests**: 45+ tests (10 + 25 + 20+)
- **Grand Total**: ~58+ tests

## Key Testing Features

### Mocking
- Bedrock client mocking for unit tests
- Environment variable patching
- File system mocking with `tmp_path` fixtures

### Fixtures
- `chatbot_module`: Loads chatbot example with mocked Bedrock
- `rag_query`: Loads RAG example with mocked Bedrock and documents
- `lambda_handler_module`: Loads Lambda handler with mocked Bedrock

### Edge Cases
- Empty and null inputs
- Very long prompts (5000+ characters)
- Special characters and emojis
- UTF-8 encoding
- Malformed JSON responses
- Missing required fields
- Type conversion errors

### Configuration Testing
- Environment variable loading and defaults
- Type coercion (string → int, float)
- Fallback mechanisms
- Security (.gitignore, .env exclusion)

## Pytest Configuration

See `pytest.ini` for configuration details:
- Test discovery patterns
- Output formatting
- Markers for test categorization

## Best Practices

1. **Run tests before committing**
   ```bash
   pytest --strict-markers
   ```

2. **Check coverage regularly**
   ```bash
   pytest --cov=. --cov-report=term-missing
   ```

3. **Use markers for organization**
   ```bash
   pytest -m "unit"
   ```

4. **Keep tests focused**
   - One assertion per test when possible
   - Clear test names describing what is tested
   - Use fixtures for common setup

5. **Mock external dependencies**
   - Avoid real AWS Bedrock calls
   - Use `monkeypatch` for environment variables
   - Use `tmp_path` for file operations

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- No external AWS credentials required (mocked)
- No network calls (mocked)
- Fast execution (< 5 seconds for all tests)
- Clear error messages for debugging

## Future Improvements

- [ ] Add performance benchmarks
- [ ] Add property-based tests using Hypothesis
- [ ] Integrate with GitHub Actions
- [ ] Add mutation testing
- [ ] Expand RAG tests with vector store mocking
