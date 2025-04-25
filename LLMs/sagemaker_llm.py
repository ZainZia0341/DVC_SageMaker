# app/LLMs/sagemaker_llm.py
import os, json, boto3
from langchain_core.llms.base import LLM
from typing import Optional, List, Mapping

class SageMakerLLM(LLM):
    def __init__(self, endpoint_name:str, region:str=None, model_kwargs:dict=None):
        self.endpoint = endpoint_name
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("sagemaker-runtime", region_name=self.region)
        self.model_kwargs = model_kwargs or {}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {"inputs": prompt, "parameters": self.model_kwargs}
        resp = self.client.invoke_endpoint(
            EndpointName=self.endpoint,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(resp["Body"].read().decode())
        return result.get("generated_text", "")

    @property
    def _identifying_params(self) -> Mapping[str, str]:
        return {"endpoint_name": self.endpoint}

    @property
    def lc_serializable(self) -> bool:
        return True
