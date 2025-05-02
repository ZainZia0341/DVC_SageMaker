# local_run_vision.py
import base64
from local_vision_src.inference import model_fn, predict_fn

if __name__ == "__main__":
    model_dir = r"D:\zain\granite-vision-3.2-2b-unsloth-bnb-4bit"
    bundle = model_fn(model_dir)

    # text-only
    resp_text = predict_fn({"inputs": "Hello, how are you?"}, bundle)
    print("🗨️ Text →", resp_text)

    # image + prompt
    with open("test.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp_img = predict_fn(
        {"image": img_b64, "text": "What’s happening in this picture?"},
        bundle
    )
    print("🖼️ Image →", resp_img)
