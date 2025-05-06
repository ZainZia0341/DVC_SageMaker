# test.py
import base64, json, boto3

def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = boto3.client("sagemaker-runtime", region_name="us-east-1")

payload = {
  "inputs": [
    {"type": "image", "image": load_image_b64("download.jpg")},
    {"type": "text",  "text":  "Describe what you see here."}
  ]
}

resp = client.invoke_endpoint(
    EndpointName="llama-3-vision-instruct",
    Body=json.dumps(payload),
    ContentType="application/json",
    Accept="application/json"
)
print(json.loads(resp["Body"].read()))
