import os, io, json, base64
from fastapi import FastAPI, Request, Response
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

app = FastAPI()

# ─── SageMaker health check ─────────────────────────────────────────
@app.get("/ping")
async def ping():
    return Response(status_code=200)

# ─── load & cache model on first invocation ────────────────────────
_model_store = None

def model_fn(model_dir):
    global _model_store
    if _model_store is None:
        model_id = os.getenv("HF_MODEL_ID", "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        _processor = AutoProcessor.from_pretrained(model_id)
        _model_store = {"model": _model, "processor": _processor}
    return _model_store

# ─── request parsing ────────────────────────────────────────────────
def input_fn(request_body, content_type="application/json"):
    if content_type.lower().startswith("application/json"):
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")

# ─── core prediction logic ──────────────────────────────────────────
def predict_fn(data, model_data):
    model, processor = model_data["model"], model_data["processor"]
    inputs = data.get("inputs", data)

    messages = []
    for item in inputs:
        if item.get("type") == "image":
            img_bytes = base64.b64decode(item["content"])
            img = Image.open(io.BytesIO(img_bytes))
            tmp = f"/tmp/img{len(messages)}.png"
            img.save(tmp)
            messages.append({"type": "image", "url": f"file://{tmp}"})
        else:
            text = item.get("text") or item.get("content") or ""
            messages.append({"type": "text", "text": text})

    # patch requests to read file:// URLs
    import requests
    real_get = requests.get
    def file_or_http(url, *a, **kw):
        if url.startswith("file://"):
            with open(url[7:], "rb") as f: return type("R",(object,),{"content":f.read(),"status_code":200})()
        return real_get(url, *a, **kw)
    requests.get = file_or_http

    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    requests.get = real_get

    model_inputs = model_inputs.to(model.device)
    out_ids = model.generate(**model_inputs, max_new_tokens=256)
    gen = processor.batch_decode(
        out_ids[:, model_inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )[0]
    return {"generated_text": gen}

# ─── response formatting ────────────────────────────────────────────
def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), accept

# ─── SageMaker invocation endpoint ─────────────────────────────────
@app.post("/invocations")
async def invoke(request: Request):
    body = await request.body()
    store = model_fn(None)
    data = input_fn(body, content_type=request.headers.get("content-type", ""))
    pred = predict_fn(data, store)
    out, mime = output_fn(pred)
    return Response(content=out, media_type=mime)
