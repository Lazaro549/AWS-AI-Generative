import boto3

REGION = "us-east-1"

def check_access():
    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=REGION
        )

        # Lightweight call to validate permissions
        client.list_foundation_models()
        print("✅ Amazon Bedrock access confirmed")

    except Exception as e:
        print("❌ Unable to access Amazon Bedrock")
        print(str(e))

if __name__ == "__main__":
    check_access()
