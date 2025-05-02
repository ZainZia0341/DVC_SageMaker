# local_vision_src/inference.py
import io
import json
import base64
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

def model_fn(model_dir, context=None):
    # 1) load the processor
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True
    )

    # 2) load the multimodal model in float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float16  # everything will be float16
    )
    model.to(device)

    return {"processor": processor, "model": model, "device": device}

def predict_fn(request, model_bundle, context=None):
    # decode or blank image
    if "image" in request:
        img = Image.open(
            io.BytesIO(base64.b64decode(request["image"]))
        ).convert("RGB")
    else:
        img = Image.new("RGB", (384, 384), color=(0, 0, 0))

    # pick up your prompt (either "text" or "inputs")
    prompt = request.get("text") or request.get("inputs") or ""

    # tokenize image+prompt
    inputs = model_bundle["processor"](
        images=img,
        text=prompt,
        return_tensors="pt"
    )
    inputs = {k: v.to(model_bundle["device"]) for k, v in inputs.items()}

    # generate
    output_ids = model_bundle["model"].generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9
    )

    # decode
    return model_bundle["processor"].batch_decode(
        output_ids, skip_special_tokens=True
    )

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported content type: {content_type}")
