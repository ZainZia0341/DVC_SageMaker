import boto3
import os
import json
import shutil
import base64
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from LLMs.llm import load_google_gemini_model, load_groq_model
from langgraph.graph.message import add_messages
from pypdf import PdfReader
import requests
from groq import Groq

# Optional: for Excel files processing.
try:
    import pandas as pd
except ImportError:
    pd = None

# Load your LLM – in this case Groq.
# (The downstream nodes remain unchanged.)
# llm = load_google_gemini_model()
llm = load_groq_model()

s3_client = boto3.client("s3")

class S3_State(TypedDict):
    messages: Annotated[list, add_messages]
    file_name: str

graph_builder_s3 = StateGraph(S3_State)


def process_file_with_groq(full_path: str) -> dict:
    """
    Process one file from local storage by examining its extension,
    extracting content or encoding an image as needed, then building
    a Groq completion prompt that includes one or more content blocks.
    Returns the analysis as a dict (parsed JSON) or an error dict.
    """
    ext = os.path.splitext(full_path)[1].lower()
    file_name = os.path.basename(full_path)
    
    # Base prompt common to all file types:
    base_prompt = f"""
You are a file analyzer which extracts all the information from the given file data.
Your response must be a valid JSON object with the following keys:
- tags: an array of exactly 10 lower-case tags,
- file_summary: a detailed summary of the file,
- file_content_category: the category you think the file belongs to,
- file_name: the file name.

Note:
In the tag array, each element will be a single word.
Do not add any extra text outside the JSON.
Do not add triple backticks, or anything else other than JSON.

Here is an example JSON response:
{{ 
    "tags": ["demo", "test", "sample", "json", "dummy", "file", "data", "info", "short", "example"], 
    "file_summary": "This is a dummy summary for a test file.", 
    "file_content_category": "demo", 
    "file_name": "test.txt" 
}}
"""
    content_blocks = []
    
    if ext in [".txt", ".csv"]:
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            return {"error": f"Error reading file {file_name}: {str(e)}"}
        prompt_text = base_prompt + f"\nHere is the file content: {file_content}\nFile Name: {file_name}"
        content_blocks.append({"type": "text", "text": prompt_text})
        
    elif ext == ".pdf":
        pdf_text = ""
        try:
            with open(full_path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    pdf_text += page.extract_text() or ""
        except Exception as e:
            return {"error": f"Error extracting text from PDF {file_name}: {str(e)}"}
        prompt_text = base_prompt + f"\nHere is the file content: {pdf_text}\nFile Name: {file_name}"
        content_blocks.append({"type": "text", "text": prompt_text})
        
    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            with open(full_path, "rb") as img_file:
                image_bytes = img_file.read()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{encoded_image}"
        except Exception as e:
            return {"error": f"Error reading image file {file_name}: {str(e)}"}
        prompt_text = base_prompt + f"\nPlease analyze the attached image.\nFile Name: {file_name}"
        content_blocks.append({"type": "text", "text": prompt_text})
        content_blocks.append({"type": "image_url", "image_url": {"url": image_data_url}})
        
    elif ext in [".xlsx", ".xls"]:
        if pd is None:
            return {"error": "pandas module required for processing Excel files."}
        try:
            df = pd.read_excel(full_path, engine="openpyxl")
            file_content = df.to_csv(index=False)
        except Exception as e:
            return {"error": f"Error reading Excel file {file_name}: {str(e)}"}
        prompt_text = base_prompt + f"\nHere is the file content extracted from Excel:\n{file_content}\nFile Name: {file_name}"
        content_blocks.append({"type": "text", "text": prompt_text})
    else:
        return {"error": f"Unsupported file type: {ext}"}
    
    message = {"role": "user", "content": content_blocks}
    
    # Initialize Groq client and call the API.
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[message],
            temperature=0.7,
            top_p=1,
            stream=False,
            stop=None
        )
    
        print("EEEEEEEEEEEEEEEEEEEEEEE ", completion)
    except Exception as e:
        return {"error": f"Groq API error: {str(e)}"}
    
    # Use this function after getting the response from the Groq API
    response_text = completion.choices[0].message.content
    print("Raw AI response:\n", response_text)  # For debugging

    # Clean the response text to remove any unwanted markdown formatting.
    clean_text = clean_response_text(response_text)
    print("Cleaned response:\n", clean_text)  # For debugging

    try:
        result = json.loads(clean_text)
    except Exception as e:
        result = {"error": f"Invalid JSON output: {response_text}"}
    return result


def clean_response_text(response_text: str) -> str:
    """
    Remove any markdown code fences (triple backticks) and language identifiers.
    """
    # Remove leading and trailing whitespace
    cleaned = response_text.strip()

    # Check if the response starts with triple backticks and possibly a language tag (like "```json")
    if cleaned.startswith("```"):
        # Split into lines
        lines = cleaned.splitlines()
        # If the first line starts with ```json or similar, remove it
        if lines[0].startswith("```"):
            lines = lines[1:]
        # If the last line is the closing triple backticks, remove it
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned

# --- Node for processing a downloaded file via Groq ---
def run_s3_download_graph(state: S3_State):
    # Instead of using a PromptTemplate chain, we now call process_file_with_groq.
    # We assume that state["file_name"] contains the file name and that the file is on disk
    # in a known target folder (which we will provide in the S3 processing loop).
    file_name = state["file_name"]
    # We assume that the file was downloaded to /tmp/s3_files/<vault>/.../<file_name>
    # For simplicity, here we assume the file is directly under /tmp/s3_files.
    full_path = os.path.join("/tmp/s3_files", file_name)
    
    print(f"------------------ Running file analysis for {full_path} ---------------------")
    analysis_result = process_file_with_groq(full_path)
    print("Analysis result from Groq:", analysis_result)
    
    # Create state messages:
    # We include the original file content (as a human message) and the analysis (as an AI message)
    try:
        with open(full_path, "rb") as f:
            content_bytes = f.read()
        try:
            # Try to decode as UTF-8; if not, leave it empty.
            original_content = content_bytes.decode("utf-8", errors="replace")
        except Exception:
            original_content = ""
    except Exception as e:
        original_content = f"Error reading file: {str(e)}"
        
    human_msg = HumanMessage(content=original_content)
    ai_msg = AIMessage(content=json.dumps(analysis_result))
    # Build the state with two messages: the first being the original input, the second the analysis.
    new_state = {"messages": [human_msg, ai_msg], "file_name": file_name}
    return new_state


# def dynamodb_node(state: S3_State, config: RunnableConfig):
#     print('---------------------------- DynamoDB Node -----------------------------')
#     # Get analysis from the AI message (assumed to be at index 1)
#     analysis = state["messages"][1].content
#     try:
#         newJson = json.loads(analysis)
#     except Exception as e:
#         print("Error parsing analysis JSON:", e)
#         newJson = {}

#     # Pull from AI's "file_name" field as fallback
#     file_name = newJson.get("file_name", "unknown")
#     original_file_name = state.get("file_name")
#     if original_file_name:
#         file_name = original_file_name

#     # REMOVE the vault prefix if it exists (e.g. "public/", "private/", or "community/").
#     # For instance, if file_name is "public/Hadith.txt", strip out "public/" so we store just "Hadith.txt".
#     allowed_vaults = ["public", "private", "community"]
#     normalized_file_name = file_name.replace("\\", "/")
#     parts = normalized_file_name.split("/", 1)  # Split into [vault, remainder]
#     if len(parts) > 1 and parts[0].lower() in allowed_vaults:
#         file_name = parts[1]
#     else:
#         file_name = normalized_file_name  # No vault prefix found
#     # model_name = config["configurable"] This is important for pasing user id data into graph during execution
#     table_name = "PublicFiles"
#     print(f"Saving analysis for file {file_name} into table {table_name}")

#     dynamodb = boto3.resource("dynamodb")
#     table = dynamodb.Table(table_name)
#     try:
#         table.put_item(Item={
#             "file_name": file_name,
#             "analysis": analysis
#         })
#         print("----------------------- Items saved in table ------------------------")
#     except Exception as e:
#         print("Error saving to DynamoDB:", str(e))

#     # You can return the updated messages here if desired
#     return {"messages": ""}


graph_builder_s3.add_node("run_s3_download_graph", run_s3_download_graph)
# graph_builder_s3.add_node("dynamodb_node", dynamodb_node)
graph_builder_s3.add_edge(START, "run_s3_download_graph")
graph_builder_s3.add_edge("run_s3_download_graph", END)
# graph_builder_s3.add_edge("dynamodb_node", END)

graph_s3 = graph_builder_s3.compile()

# URL of your presigned‑URL generator endpoint
PRESIGN_API = "https://p2vakzfxqb.execute-api.us-east-1.amazonaws.com/dev/generate-presigned-url"

def download_via_presigned_url(bucket: str, key: str, target_path: str):
    # 1) Ask your backend for a presigned URL
    resp = requests.post(
        PRESIGN_API,
        json={"bucketName": bucket, "key": key},
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    print("--------------- resp --------------- ", resp)
    resp.raise_for_status()
    print("--------------- resp again --------------- ", resp)
    presigned = resp.json().get("url")
    print("--------------- presigned url --------------- ", presigned)
    if not presigned:
        raise RuntimeError(f"No URL returned for {bucket}/{key}: {resp.text}")

    # 2) Download the file via HTTP GET
    r = requests.get(presigned, stream=True, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as f:
        for chunk in r.iter_content(1024 * 32):
            if not chunk:
                continue
            f.write(chunk)

def process_files(bucket: str, file_keys: List[str], config):
    allowed_vaults = ["public", "private", "community"]
    print("Bucket:", bucket)
    print("File keys:", file_keys)
    
    target_folder = "/tmp/s3_files"
    os.makedirs(target_folder, exist_ok=True)
    
    # Download each file via presigned URL
    for file_key in file_keys:
        file_path = os.path.join(target_folder, file_key)
        try:
            download_via_presigned_url(bucket, file_key, file_path)
            print(f"Downloaded {file_key} to {file_path}")
        except Exception as e:
            print(f"Error downloading {file_key}: {e}")
    
    results = []
    
    # Process each downloaded file (each file is processed one at a time)
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            full_path = os.path.join(root, file)
            print("Processing file:", full_path)
            # Instead of trying to decode the file as text here and using a chain,
            # we call our new multimodal Groq analysis function.
            state = {"file_name": os.path.relpath(full_path, target_folder)}
            try:
                # Call the graph which in turn calls run_s3_download_graph,
                # which now uses process_file_with_groq.
                # The new state will have two messages.
                for event in graph_s3.stream(state, config, stream_mode="values"):
                    state = event  # For our purposes, we use the final state.
            except Exception as e:
                print("Error processing events:", e)
                state = {"messages": [HumanMessage(content=""), AIMessage(content=json.dumps({"error": str(e)}))], "file_name": file}
            results.append({"file_name": file, "events": [state]})
    
    try:
        shutil.rmtree(target_folder)
        print("Target folder deleted.")
    except Exception as e:
        print(f"Error deleting temporary folder {target_folder}: {str(e)}")
    
    return results
