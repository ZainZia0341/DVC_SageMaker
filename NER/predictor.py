from flask import Flask, request, Response
import json
from model_loader import model_fn, predict_fn

# Pre-load model so /ping is immediately healthy
model_fn()

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return Response(response="\n", status=200, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def invoke():
    data = request.get_json(force=True)
    result = predict_fn(data)
    return Response(response=json.dumps(result),
                    status=200,
                    mimetype="application/json")
