#!/usr/bin/env bash

echo "🔧 Setting up local environment..."

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install boto3

echo "✅ Environment ready"
echo "👉 Run: source .venv/bin/activate"
