from sagemaker import image_uris, Session
# import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# 1) Retrieve the official PyTorch 2.6 inference container URI directly
image_uri = image_uris.retrieve(
    framework="pytorch",               # framework name
    region="us-east-1",                 # your AWS region
    version="2.6.0",                    # container’s PyTorch version
    py_version="py312",                 # Python 3.10
    instance_type="ml.g5.xlarge",       # target instance type
    image_scope="inference"             # inference-only image
)  # :contentReference[oaicite:0]{index=0}

sess = Session()
# role = sagemaker.get_execution_role()   # ensure this is the same-account role
# arn:aws:iam::879961718398:role/service-role/AmazonSageMaker-ExecutionRole-20250416T155074
# print("role arn returned from sagemaker ", role)

# 2) Create the Hugging Face Model, overriding the default image lookup
hf_model = HuggingFaceModel(
    model_data="s3://dvc-sagemaker-model/llama-3.2-1B-Instruct-bnb-4bit-flat.tar.gz",
    role="arn:aws:iam::879961718398:role/service-role/AmazonSageMaker-ExecutionRole-20250416T155074",                          # "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    entry_point="inference.py",
    source_dir="code/",
    image_uri=image_uri,                # override default framework/version resolution :contentReference[oaicite:1]{index=1}
    env={"SM_NUM_GPUS": "1"},
    sagemaker_session=Session(),
)

# 3) Deploy (or update) your endpoint
predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="my-endpoint",
)  # :contentReference[oaicite:2]{index=2}
