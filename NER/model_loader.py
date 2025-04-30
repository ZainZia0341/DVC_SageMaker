import os, torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

_model = None
_processor = None

def model_fn():
    global _model, _processor
    if _model is None:
        model_id = os.getenv("HF_MODEL_ID",
                             "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("⏳ Loading quantized model…", flush=True)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        _processor = AutoProcessor.from_pretrained(model_id)
        print("✅ Model loaded", flush=True)
    return _model, _processor

def predict_fn(data):
    model, processor = model_fn()
    # -- build messages list, decode images/text as in your snippet --
    inputs = data.get("inputs", data)
    messages = []
    for item in inputs:
        if item.get("type") == "image":
            import base64, io
            from PIL import Image
            img_bytes = base64.b64decode(item["content"])
            img = Image.open(io.BytesIO(img_bytes))
            path = f"/tmp/img_{len(messages)}.png"
            img.save(path)
            messages.append({"type":"image","url":f"file://{path}"})
        elif item.get("type") == "text":
            text = item.get("text") or item.get("content","")
            messages.append({"type":"text","text":text})

    # Monkey-patch requests.get for file://
    import requests
    orig_get = requests.get
    def _get(url,*a,**k):
        if url.startswith("file://"):
            p=url[7:]
            return type("R",(object,),{"content":open(p,"rb").read(),"status_code":200})()
        return orig_get(url,*a,**k)
    requests.get = _get

    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    requests.get = orig_get

    output_ids = model.generate(**model_inputs, max_new_tokens=256)
    text = processor.batch_decode(
        output_ids[:, model_inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )[0]
    return {"generated_text": text}
