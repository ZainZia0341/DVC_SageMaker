# deploy_sagemaker.py

from dotenv import load_dotenv
import os
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# 1️⃣ Load env vars from .env
load_dotenv()
role     = os.getenv("SAGEMAKER_ROLE_ARN")
region   = os.getenv("AWS_REGION", "us-east-1")
endpoint = os.getenv("SAGEMAKER_ENDPOINT_NAME")

if not role or not endpoint:
    raise ValueError("Make sure SAGEMAKER_ROLE_ARN and SAGEMAKER_ENDPOINT_NAME are set in your .env")

# 2️⃣ Resolve an absolute path to your code folder
root_dir = os.path.dirname(__file__)
code_dir = os.path.join(root_dir, "code")
if not os.path.isdir(code_dir):
    raise ValueError(f"Cannot find code directory at {code_dir}")

# 3️⃣ Initialize SageMaker/Boto clients
sm = boto3.client("sagemaker", region_name=region)

# 4️⃣ Clean up any previous endpoint & endpoint‐config
try:
    sm.delete_endpoint(EndpointName=endpoint)
    print(f"Deleted old endpoint {endpoint}")
    sm.get_waiter("endpoint_deleted").wait(EndpointName=endpoint)
    print(f"Confirmed endpoint deletion")
except sm.exceptions.ClientError:
    pass

try:
    sm.delete_endpoint_config(EndpointConfigName=endpoint)
    print(f"Deleted old endpoint‐config {endpoint}")
except sm.exceptions.ClientError:
    pass

# 5️⃣ Define the HuggingFaceModel with your custom inference code
hf_model = HuggingFaceModel(
    entry_point="inference.py",
    source_dir=code_dir,
    role=role,
    transformers_version="4.48.0",   # supported by your SDK
    pytorch_version="2.3.0",         # supported by your SDK
    py_version="py311",
    env={"AWS_REGION": region},
)

# 6️⃣ Deploy the model as a real SageMaker endpoint
predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",                                  # 16 GB T4 GPU
    endpoint_name=endpoint,
    container_startup_health_check_timeout_in_seconds=600,            # up to 10 min for large model download
    wait=True,                                                       # block until InService
    logs=True,                                                       # stream CloudWatch logs locally
)

print("✅ Deployed endpoint:", predictor.endpoint_name)
