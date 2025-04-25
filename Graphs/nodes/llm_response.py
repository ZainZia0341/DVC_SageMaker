# app/nodes/llm_response_generation_node.py
import os, json, boto3
from langchain_core.messages import AIMessage
from ..Graph_State import State

SM = boto3.client("sagemaker-runtime", region_name=os.getenv("AWS_REGION"))
ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT_NAME")

def call_sagemaker(prompt, params=None):
    payload = {"inputs": prompt}
    if params:
        payload["parameters"] = params
    resp = SM.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )
    return json.loads(resp["Body"].read().decode()).get("generated_text","")

def llm_response_generation_node(state: State):
    user_q = state["messages"][0].content
    file_content = state["messages"][-1].content or "No file content"
    if isinstance(file_content, list):
        texts = [blk.get("text","") for blk in file_content]
    else:
        texts = [file_content]
    prompt = "User Question: " + user_q + "\n\n" + "\n\n".join(texts)

    try:
        result = call_sagemaker(prompt, {"temperature":0.7})
    except Exception as e:
        return {"messages":[AIMessage(content=f"Error: {e}")]}

    return {"messages":[AIMessage(content=result)]}
