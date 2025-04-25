# test_sagemaker.py
import os, json, boto3
# test_sagemaker.py
from dotenv import load_dotenv

# 1️⃣ Load .env into os.environ
load_dotenv()

ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT_NAME")
REGION   = os.getenv("AWS_REGION", "us-east-1")
client   = boto3.client("sagemaker-runtime", region_name=REGION)

def ask(prompt):
    payload = {"inputs": prompt}
    resp = client.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(resp["Body"].read().decode())["generated_text"]

if __name__ == "__main__":
    print(ask("Hello, world! How are you?"))


