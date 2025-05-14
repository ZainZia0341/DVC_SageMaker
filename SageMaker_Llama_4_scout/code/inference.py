import json
import torch
from transformers import (
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    AutoProcessor,
    pipeline,
)

def model_fn(model_dir: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_dir,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    return pipeline(
        task="image-text-to-text",
        model=model,
        processor=processor,
        device_map="auto",
        return_full_text=False,
        max_new_tokens=512,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

def input_fn(request_body: str, request_content_type: str):
    print("------------- input in input function ---------------")
    print(request_body)
    # We expect a JSON array of {role, content:[{type, …}]} entries
    data = json.loads(request_body)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array of message dicts")
    return data

def predict_fn(messages: list, pipe):
    # Directly hand your array of turns to the pipeline,
    # exactly like your Colab test did.
    # The pipeline will accept both URL‐style and base64 images.
    print("------------- input in predict function ---------------")
    print(messages)    
    return pipe(text=messages)

def output_fn(prediction, response_content_type: str):
    print("------------- input in output function ---------------")
    print(prediction)    
    if response_content_type != "application/json":
        raise ValueError("Only application/json is supported")
    # Pull out just the generated_text
    if isinstance(prediction, list):
        texts = [item.get("generated_text", str(item)) for item in prediction]
        result = texts[0] if len(texts) == 1 else texts
    elif isinstance(prediction, dict) and "generated_text" in prediction:
        result = prediction["generated_text"]
    else:
        result = str(prediction)
    return json.dumps(result)
