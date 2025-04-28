# app.py
from fastapi import FastAPI, Request, Response
import inference, json

app = FastAPI()

@app.get("/ping")
async def ping():
    return Response(status_code=200)

_model_store = None

@app.post("/invocations")
async def invoke(request: Request):
    global _model_store
    if _model_store is None:
        _model_store = inference.model_fn(None)
    data = inference.input_fn(await request.body(), content_type="application/json")
    pred = inference.predict_fn(data, _model_store)
    body, ctype = inference.output_fn(pred)
    return Response(content=body, media_type=ctype)