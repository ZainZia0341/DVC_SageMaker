# deploy.py

from sagemaker import image_uris, Session
from sagemaker.huggingface import HuggingFaceModel


from sagemaker.session import Session
sess = Session()


# 1) Choose the inference container
image_uri = image_uris.retrieve(
    framework="pytorch",
    region="us-east-1",
    version="2.6.0",
    py_version="py312",
    instance_type="ml.g5.xlarge",
    image_scope="inference",
)

# sess = Session()

# 2) Create the Hugging Face Model, pointing to your multimodal code
hf_model = HuggingFaceModel(
    model_data="s3://dvc-sagemaker-model/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit.tar.gz",
    role="arn:aws:iam::879961718398:role/service-role/AmazonSageMaker-ExecutionRole-20250416T155074",
    entry_point="inference.py",
    source_dir="code/",
    image_uri=image_uri,
    env={
      "SM_NUM_GPUS":        "1",
      "TRANSFORMERS_CACHE": "/tmp/model_cache",
      "HF_HOME":            "/tmp/model_cache"
    },
    sagemaker_session=sess,
)

# 3) Deploy to an endpoint
predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.12xlarge",
    endpoint_name="llama-3-2-vision-from-sagemaker-16th-try",
)
print("Multimodal endpoint:", predictor.endpoint_name)