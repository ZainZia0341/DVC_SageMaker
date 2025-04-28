from flask import Flask, request, Response
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

# Load quantized model and tokenizer
MODEL_ID = "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit"
quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    # Return 200 if model loaded, else 404
    status = 200 if model is not None else 404
    return Response(status=status)

@app.route("/invocations", methods=["POST"])
def invocations():
    # Parse input JSON
    data = request.get_json(force=True)
    prompt = data.get("input", "")
    # Tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return JSON with the generated text
    return Response(json.dumps({"output": text}),
                    mimetype="application/json", status=200)
