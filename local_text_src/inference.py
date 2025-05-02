# src/inference.py
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def model_fn(model_dir, context=None):
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return {"tokenizer": tokenizer, "model": model}

def predict_fn(input_data, model_bundle, context=None):
    """
    Expects JSON dict:
      { "inputs": "Hello world" }
    """
    text = input_data.get("inputs") or input_data.get("text") or ""
    inputs = model_bundle["tokenizer"](
        text, return_tensors="pt", padding=True
    ).to(model_bundle["model"].device)

    # generate
    output_ids = model_bundle["model"].generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9
    )
    # decode and return list of strings
    return model_bundle["tokenizer"].batch_decode(output_ids, skip_special_tokens=True)

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported content type: {content_type}")
