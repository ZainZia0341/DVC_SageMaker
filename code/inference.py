import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def model_fn(model_dir):
    """Load the quantized model and tokenizer once at container startup."""
    model_id = "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit"
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True,
    )
    return model, tokenizer

def input_fn(body, content_type):
    """Parse JSON request into (prompt, generation_params)."""
    data = json.loads(body)
    prompt = data.get("inputs") or data.get("prompt") or ""
    params = data.get("parameters", {})
    return prompt, params

def predict_fn(input_data, model_and_tok):
    """Run model.generate() and return raw text."""
    prompt, params = input_data
    model, tok = model_and_tok
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    defaults = {"max_new_tokens":200, "do_sample":True, "temperature":0.7}
    gen = model.generate(**inputs, **{**defaults, **params})
    return tok.decode(gen[0], skip_special_tokens=True)

def output_fn(prediction, accept):
    """Return JSON response."""
    return json.dumps({"generated_text": prediction}), "application/json"
