# app.py
from fastapi import FastAPI, Request, Response
import json
import inference  # your inference.py

app = FastAPI()
_model_store = inference.model_fn(None)

@app.get("/ping")
async def ping():
    return Response(status_code=200)

@app.post("/invocations")
async def invoke(request: Request):
    # 1) read JSON
    body = await request.body()
    data = inference.input_fn(body, content_type="application/json")
    # 2) predict
    pred = inference.predict_fn(data, _model_store)
    # 3) format output
    out_body, content_type = inference.output_fn(pred, accept="application/json")
    return Response(content=out_body, media_type=content_type)
