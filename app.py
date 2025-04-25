# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from typing import List, Union, Optional, Any
import os
from langgraph.types import Command
import json
from Graphs.Graph_Flow import run_graph, resume_graph
from S3_File_Download.Graph_s3_AI import process_files

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")


app = FastAPI()

#######################################################
# 1) CORS Middleware
#######################################################
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify a list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#######################################################
# 2) Request/Response Schemas
#######################################################
class ChatRequest(BaseModel):
    thread_id: str
    user_id: Optional[str] = None  # New field for user id
    user_message: Optional[str] = None
    resume_data: Union[str, List[str]] = None
    match_threshold: Optional[int] = None
    
class ChatResponse(BaseModel):
    paused: bool
    interrupt_data: Optional[Any] = None
    events: list

#######################################################
# 3) Single Endpoint: /chat
#######################################################
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id, "user_id": request.user_id}}
    paused = False
    interrupt_data = None

    try:
        if request.resume_data:
            # Resume from an interrupt
            cmd = Command(resume={"data": request.resume_data})
            events_list = resume_graph(cmd, config)
        else:
            # Initialize your graph state
            initial_state = {
                "messages": [],
                "tags": "",
                "files": "",
                
                # Pass the match_threshold from request (or default to 0 if None)
                "match_threshold": request.match_threshold or 0
            }

            if request.user_message:
                initial_state["messages"].append({
                    "role": "user",
                    "content": request.user_message
                })

            events_list, paused, interrupt_data = run_graph(initial_state, config)

        return ChatResponse(
            paused=paused,
            interrupt_data=interrupt_data,
            events=events_list
        )

    except Exception as e:
        traceback_str = "".join(traceback.format_exception(None, e, e.__traceback__))
        print("[ERROR] An exception occurred in /chat:\n", traceback_str)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------- s3 file downloading and tags summary generation ------------------------------------------------- #

class S3DownloadRequest(BaseModel):
    thread_id: Optional[str] = None
    bucket: str
    fileKeys: List[str]
    vault: Optional[str] = None  # e.g. "private", "public", "community"
    user_id: Optional[str] = None  # For private vault identification


@app.post("/s3_download")
def s3_download_endpoint(request: S3DownloadRequest):
    try:
        config = {"thread_id": "123"}
        results = process_files(request.bucket, request.fileKeys, config)
        
        # Get list of processed file names
        processed_files = [res.get("file_name", "unknown") for res in results]
        
        # Extract valid tags; since we now expect one event with one valid AI message per file,
        # we simply take the first message from the event.
        extracted_tags = []
        file_summaries = []
        file_content_category = []
        for result in results:
            # print("RRRRRRRRRRRRRRRRRRRRRRRRRRR ", result)
            if "events" in result and result["events"]:
                event = result["events"][0]  # one event per file
                if event["messages"]:
                    message = event["messages"][1]  # our deduplicated valid AI message
                    try:
                        content_data = json.loads(message.content)
                        print("---------------- content_Data --------------- ", content_data)
                        # Check if the expected "tags" key is present
                        if "tags" in content_data and isinstance(content_data["tags"], list):
                            extracted_tags.append(content_data["tags"])
                            file_summaries.append(content_data["file_summary"])
                            file_content_category.append(content_data["file_content_category"])
                        else:
                            print(f"Message from file {result['file_name']} does not contain expected tags.")
                    except Exception as e:
                        print(f"Error parsing JSON from file {result['file_name']}: {e}")
        
        return {
            "message": "Files processed successfully.",
            "processed_files": processed_files,
            "tags": extracted_tags,
            "file_summary": file_summaries,
            "file_content_category": file_content_category, 
        }
    
    except Exception as e:
        traceback_str = "".join(traceback.format_exception(None, e, e.__traceback__))
        print("[ERROR] Exception in /s3_download endpoint:\n", traceback_str)
        raise HTTPException(status_code=500, detail=str(e))
    