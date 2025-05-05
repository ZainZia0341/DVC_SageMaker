# code/inference.py

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

def model_fn(model_dir, context=None):
    # 4-bit NF4 + half precision compute
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )

    # wrap in HF pipeline
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

def input_fn(request_body, request_content_type):
    """
    SageMaker will call this instead of the default_input_fn.
    We read JSON and return a plain Python dict, not a numpy.object_ array.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Expect {"inputs": "your text"} or {"inputs": ["text1","text2",...]}
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(data, pipe):
    """
    data is exactly what input_fn returned.
    We extract the 'inputs' key and pass it to the HF pipeline.
    """
    return pipe(data["inputs"])

def output_fn(prediction, response_content_type):
    """
    Serialize the HF pipeline output (usually a list of dicts) back to JSON.
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported response content type: {response_content_type}")
