# Getting Started

This guide helps you run your first Generative AI example on AWS.

## Prerequisites
- AWS account
- Access to Amazon Bedrock
- Python 3.10+
- AWS CLI configured

## Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Lazaro549/AWS-AI-Generative.git
   cd AWS-AI-Generative
   ```
2. Install the project (in a virtualenv if you use one):
   ```bash
   pip install -e ".[dev]"
   ```
3. Set up your environment:
   ```bash
   cp .env.example .env
   # then edit .env with your AWS region, profile, and Bedrock model ID
   ```
4. (Optional) Confirm your AWS credentials can reach Bedrock:
   ```bash
   python scripts/check_bedrock_access.py
   ```
5. Run an example:
   ```bash
   python examples/chatbot/app.py
   python examples/rag/query.py
   ```

See the main [README](../README.md) for the full repository layout, running
the test suite, and the [`evaluation/`](../evaluation/) framework.
