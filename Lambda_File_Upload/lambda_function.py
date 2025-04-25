import os
import json
import io
import cgi
import base64
import boto3
import requests

s3 = boto3.client("s3")

# Adjust these for your environment
BUCKET_NAME = os.environ.get("BUCKET_NAME", "my-s3-bucket")
FASTAPI_URL = "http://127.0.0.1:8000/s3_download"  # Replace with your actual FastAPI domain

def lambda_handler(event, context):
    """
    Expects multipart/form-data with fields:
      - vault (optional, defaults to "public")
      - user_id (optional, defaults to "anonymous")
      - community_id (optional, required if vault == "community")
      - One or more file fields (with filename)

    Example: In Postman (form-data):
      KEY            VALUE
      vault          public
      user_id        test-user
      files          [Attach a PDF or any file]

    1. Parse multipart form data from event["body"].
    2. Write each file to /tmp and upload to S3 in the appropriate vault subfolder.
    3. If any upload fails, return an error.
    4. If all uploads succeed, call the FastAPI endpoint with the list of uploaded file keys.
    5. Return success or error.
    """

    # 1) Extract the body from the event, and decode if base64-encoded (API Gateway sets isBase64Encoded=True).
    raw_body = event.get("body", "")
    if event.get("isBase64Encoded", False):
        raw_body = base64.b64decode(raw_body)

    # 2) Parse the multipart form data
    # Get the content type from headers (could be 'content-type' or 'Content-Type')
    content_type = event["headers"].get("Content-Type") or event["headers"].get("content-type")
    environ = {
        "REQUEST_METHOD": "POST",
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(len(raw_body))
    }

    form = cgi.FieldStorage(
        fp=io.BytesIO(raw_body),
        environ=environ,
        keep_blank_values=True
    )

    # Extract form fields
    vault = form.getvalue("vault", "public").lower()
    user_id = form.getvalue("user_id", "anonymous")
    community_id = form.getvalue("community_id")

    # Validate if vault is community
    if vault == "community" and not community_id:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing community_id for community vault."})
        }

    # Identify all file fields (fields that have a filename)
    file_fields = [field for field in (form.list or []) if field.filename]

    if not file_fields:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No files provided."})
        }

    # 3) Upload each file to S3
    uploaded_files = []
    for field in file_fields:
        file_name = field.filename
        file_bytes = field.file.read()  # Read the file contents

        # Create a temporary file in /tmp
        temp_path = f"/tmp/{file_name}"
        with open(temp_path, "wb") as tmp:
            tmp.write(file_bytes)

        # Determine subfolder based on vault
        if vault == "private":
            subfolder = f"{vault}/{user_id}"
        elif vault == "community":
            subfolder = f"{vault}/{community_id}"
        else:
            subfolder = vault  # e.g. 'public'

        # Construct the S3 key with the subfolder
        s3_key = f"{subfolder}/{file_name}"
        try:
            s3.upload_file(temp_path, BUCKET_NAME, s3_key)
            uploaded_files.append(s3_key)
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": f"Failed to upload {file_name}: {str(e)}"})
            }

    # 4) Call the FastAPI endpoint (optional)
    data_for_fastapi = {
        "bucket": BUCKET_NAME,
        "fileKeys": uploaded_files,
        "vault": vault,
        "user_id": user_id,
        "community_id": community_id
    }

    try:
        resp = requests.post(FASTAPI_URL, json=data_for_fastapi, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "All files uploaded to S3, but calling FastAPI failed.",
                "error": str(e)
            })
        }

    # 5) Return success
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "All files uploaded successfully, FastAPI triggered.",
            "fastapiResponse": resp.text
        })
    }
