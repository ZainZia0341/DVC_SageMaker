import base64, io, os, json
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

# Load model and processor at module import (so it happens once at container startup)
model_id = os.getenv("HF_MODEL_ID", "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit")
# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
# Load the multimodal model in 4-bit mode with device mapping
model = AutoModelForCausalLM.from_pretrained(
     model_id,
     device_map="auto",
     quantization_config=bnb_config,
     torch_dtype=torch.bfloat16,       # optional but recommended
     trust_remote_code=True            # if your model repo requires it
)
processor = AutoProcessor.from_pretrained(model_id)  # This provides image processor + tokenizer

def model_fn(model_dir):
    # Return the model and processor (the inference toolkit will call this to get the model)
    return {"model": model, "processor": processor}

def input_fn(request_body, content_type="application/json"):
    # Parse the incoming request JSON to a Python dict
    if content_type == "application/json":
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(data, model_data):
    """Perform prediction on the input data using the model."""
    model = model_data["model"]
    processor = model_data["processor"]
    # The input JSON is expected to have an "inputs" field containing a list of items
    # (images and text segments).
    # Example expected format:
    # {
    #   "inputs": [
    #       {"type": "image", "content": "<base64-image-1>"},
    #       {"type": "image", "content": "<base64-image-2>"},
    #       {"type": "text",  "content": "Question about the images?"}
    #   ]
    # }
    inputs = data.get("inputs", data)  # 'data' may itself be the inputs list
    # Construct the messages list for the processor (multimodal chat format)
    messages = []
    for item in inputs:
        if item.get("type") == "image":
            # For image input: decode base64 content to an image file
            if "content" in item:
                image_bytes = base64.b64decode(item["content"])
                img = Image.open(io.BytesIO(image_bytes))
                img_path = f"/tmp/image_{len(messages)}.png"
                img.save(img_path)
                # Use a file URL for the processor to read this image
                messages.append({"type": "image", "url": f"file://{img_path}"})
            elif "url" in item:
                # If an image URL is provided, pass it through
                messages.append({"type": "image", "url": item["url"]})
        elif item.get("type") == "text":
            # For text input: take the content as user query
            text = item.get("text") or item.get("content") or ""
            messages.append({"type": "text", "text": text})
    # Use the processor to generate model inputs from the messages.
    # We'll patch the requests.get function temporarily to allow local file URLs.
    import requests
    original_get = requests.get
    def _get_file_or_http(url, *args, **kwargs):
        if url.startswith("file://"):
            # If file URL, read the file from local disk
            path = url[len("file://"):]
            with open(path, "rb") as f:
                content = f.read()
            class DummyResponse:
                def __init__(self, content):
                    self.content = content
                    self.status_code = 200
            return DummyResponse(content)
        return original_get(url, *args, **kwargs)
    requests.get = _get_file_or_http  # Monkey-patch requests.get
    # Apply the chat template to construct model-ready inputs (with image tokens, etc.)
    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    # Restore the original requests.get
    requests.get = original_get
    # Move model inputs to the same device as the model (GPU)
    model_inputs = model_inputs.to(model.device)
    # Generate the model's response (limit to reasonable new tokens for safety)
    output_ids = model.generate(**model_inputs, max_new_tokens=256)
    # Decode the generated tokens to text, skipping the input prompt part
    generated_text = processor.batch_decode(
        output_ids[:, model_inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )[0]
    # Return the response in dict format
    return {"generated_text": generated_text}

def output_fn(prediction, accept="application/json"):
    # Serialize the prediction dict to JSON
    return json.dumps(prediction), accept
