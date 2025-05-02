# run_local.py
import json
from local_src.inference import model_fn, predict_fn

if __name__ == "__main__":
    model_dir = r"D:\zain\gemma-2-2b-bnb-4bit"
    bundle = model_fn(model_dir)

    # Text-only test
    resp = predict_fn({"inputs": "write a song for me on PSL"}, bundle)
    print("▶️ Response:", resp)
