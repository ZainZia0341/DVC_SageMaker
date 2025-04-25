import os
import re
import json
import ast
import boto3
from pypdf import PdfReader
from langchain_core.messages import HumanMessage
import pandas as pd
from ..Graph_State import State
import base64
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.



s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

def get_file_from_s3_node(state: State):
    """
    Download files from S3 and build a list of content blocks for a multi-modal LLM.
    
    Supported file types:
      - Text-based: txt, csv, pdf, xls, xlsx (content is extracted as text)
      - Images: jpg, jpeg, png (content is provided as a base64-encoded data URL)
    
    For image files, two blocks are added:
      1. A text prompt with the file name.
      2. An image block with the image data.
      
    The function returns a state with a HumanMessage whose content is the list of blocks.
    """
    print("State received:", state)
    print("Files content:", state["files"].content)
    
    content = state["files"].content
    file_names = []
    
    # Extract file names from state, supporting different input formats.
    if isinstance(content, list):
        file_names = content
    elif isinstance(content, str):
        try:
            data_dict = ast.literal_eval(content)
            if isinstance(data_dict, dict) and "data" in data_dict:
                data_val = data_dict["data"]
                file_names = data_val if isinstance(data_val, list) else [data_val]
            else:
                file_names = [content]
        except Exception as e:
            print("Error parsing file content as dict:", e)
            file_names = [content]
    else:
        print("Unsupported format for state['files'].content")
        return {}
    
    print("User provided file names:", file_names)
    
    content_blocks = []
    for file_name in file_names:
        s3_key = f"public/{file_name}"
        local_path = f"/tmp/{file_name}"
        print("Downloading", s3_key, "to", local_path)
        try:
            s3_client.download_file(BUCKET_NAME, s3_key, local_path)
            print(f"Downloaded file from S3: {file_name}")
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            continue
        
        extension = file_name.lower().rsplit(".", 1)[-1]
        
        # Handle PDF files.
        if extension == "pdf":
            pdf_text = ""
            try:
                with open(local_path, "rb") as pdf_file:
                    reader = PdfReader(pdf_file)
                    for page in reader.pages:
                        pdf_text += (page.extract_text() or "") + "\n"
            except Exception as e:
                print(f"Error processing PDF {file_name}: {e}")
                pdf_text = "Error extracting PDF content."
            block = {
                "type": "text",
                "text": f"--- Content from {file_name} ---\n{pdf_text}"
            }
            content_blocks.append(block)
            
        # Handle plain text and CSV files.
        elif extension in ["txt", "csv"]:
            text_content = ""
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                text_content = "Error extracting text content."
            block = {
                "type": "text",
                "text": f"--- Content from {file_name} ---\n{text_content}"
            }
            content_blocks.append(block)
            
        # Handle Excel files.
        elif extension in ["xls", "xlsx"]:
            text_content = ""
            try:
                df = pd.read_excel(local_path)
                text_content = df.to_csv(index=False)
            except Exception as e:
                print(f"Error reading Excel file {file_name}: {e}")
                text_content = "Error extracting Excel content."
            block = {
                "type": "text",
                "text": f"--- Content from {file_name} ---\n{text_content}"
            }
            content_blocks.append(block)
            
        # Handle image files.
        elif extension in ["jpg", "jpeg", "png"]:
            try:
                with open(local_path, "rb") as img_file:
                    image_bytes = img_file.read()
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                # Determine MIME type based on extension.
                mime_type = "image/jpeg" if extension in ["jpg", "jpeg"] else "image/png"
                image_data_url = f"data:{mime_type};base64,{encoded_image}"
            except Exception as e:
                print(f"Error processing image {file_name}: {e}")
                continue
            # Create a text block with instructions.
            prompt_text = f"Please analyze the attached image.\nFile Name: {file_name}"
            content_blocks.append({"type": "text", "text": prompt_text})
            # Create an image block with the image URL.
            content_blocks.append({"type": "image_url", "image_url": {"url": image_data_url}})
            
        else:
            print(f"Unsupported file type for file {file_name}")
            content_blocks.append({
                "type": "text",
                "text": f"Unsupported file type for {file_name}"
            })
            
    if not content_blocks:
        content_blocks.append({
            "type": "text",
            "text": "No content could be extracted from the provided file(s)."
        })
    
    # Return a HumanMessage with the content blocks list.
    return {"messages": HumanMessage(content=content_blocks)}