# src/inference.py
import os, io, json, base64, torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

def model_fn(model_dir, context=None):
    # model_dir should *itself* be the path that has all the HF files+    
    # # and local_files_only=True prevents any HF-hub lookup
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model     = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return {"processor": processor, "model": model, "device": device}

def _decode_image(b64_string):
    data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(data)).convert("RGB")

# Remove this entirely to fallback on the toolkit's default JSON handler:
# def input_fn(request_body, request_content_type):
#     ...

def predict_fn(input_data, model_bundle, context=None):
    """
    Handle both image+text and text-only cases.
    Accepts three args so we swallow the toolkit's context.
    """
    # If they passed "inputs", treat it as text-only
    if "inputs" in input_data:
        text = input_data["inputs"]
        inputs = model_bundle["processor"](
            text, return_tensors="pt"
        ).to(model_bundle["device"])
        generated_ids = model_bundle["model"].generate(**inputs)
        return model_bundle["processor"].batch_decode(generated_ids, skip_special_tokens=True)

    # Otherwise image+text
    img = _decode_image(input_data["image"])
    text = input_data.get("text", "")
    inputs = model_bundle["processor"](images=img, text=text, return_tensors="pt")
    inputs = {k: v.to(model_bundle["device"]) for k, v in inputs.items()}
    generated_ids = model_bundle["model"].generate(**inputs)
    return model_bundle["processor"].batch_decode(generated_ids, skip_special_tokens=True)

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported content type: {response_content_type}")