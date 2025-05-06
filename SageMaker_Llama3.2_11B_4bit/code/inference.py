# code/inference.py

import io, json, base64, torch
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    AutoProcessor,
    pipeline,
)

def model_fn(model_dir):
    # 4-bit NF4 + CPU offload
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    # processor & model
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_dir,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    # create HF pipeline for image+text → text
    return pipeline(
        task="image-text-to-text",
        model=model,
        processor=processor,
        device_map="auto",
        return_full_text=False,             # strip out the prompt
        max_new_tokens=512,                 # give it enough budget to emit full JSON
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

def _decode_image(b64string):
    data = base64.b64decode(b64string)
    return Image.open(io.BytesIO(data)).convert("RGB")

def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    payload = json.loads(request_body)
    messages = payload["inputs"]  # list of {"type":"image"/"text", ...}
    # decode any image messages
    for msg in messages:
        if msg["type"] == "image":
            msg["image"] = _decode_image(msg["image"])
    return {"inputs": messages}

def predict_fn(data, pipe):
    return pipe(
        text=data["inputs"],
        max_new_tokens=data.get("max_new_tokens", 64),
        return_full_text=data.get("return_full_text", False),
    )

def output_fn(prediction, response_content_type):
    if response_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {response_content_type}")
    return json.dumps(prediction)
