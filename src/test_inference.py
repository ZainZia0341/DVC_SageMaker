# test_inference.py

import os, tempfile, json, base64
from src.inference import model_fn, predict_fn, output_fn

# 1. Point to your downloaded local model folder
MODEL_DIR = "/path/to/Llama-3.2-11B-Vision-Instruct-bnb-4bit"

# 2. Load the model bundle
print("Loading model_fn…")
bundle = model_fn(MODEL_DIR, context=None)
print("✔ model_fn OK")

# 3. Test text-only
print("\nRunning text-only prediction…")
text_input = {"inputs": "Hello, how are you today?"}
text_out   = predict_fn(text_input, bundle, context=None)
print("→", text_out)

# 4. Test image+text
print("\nRunning image+text prediction…")
# encode a small PNG or JPEG into base64
with open("test_small.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
vision_input = {"image": b64, "text": "Describe this image."}
vision_out   = predict_fn(vision_input, bundle, context=None)
print("→", vision_out)

# 5. Make sure output_fn serializes it cleanly
print("\nSerializing…")
body, content_type = output_fn(vision_out, "application/json")
print("→ JSON body length:", len(body), "content_type:", content_type)
