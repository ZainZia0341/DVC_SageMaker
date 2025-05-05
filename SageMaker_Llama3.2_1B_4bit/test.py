# test.py
import boto3, json

client = boto3.client("sagemaker-runtime", region_name="us-east-1")
payload = {"inputs": "Tell me a joke on sales person for software house"}

resp = client.invoke_endpoint(
    EndpointName="my-endpoint",
    Body=json.dumps(payload),
    ContentType="application/json",
    Accept="application/json"
)
print(json.loads(resp["Body"].read()))
