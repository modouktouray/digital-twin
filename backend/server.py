from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from context import prompt

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Bedrock model selection
# Available models:
# - amazon.nova-micro-v1:0  (fastest, cheapest)
# - amazon.nova-lite-v1:0   (balanced - default)
# - amazon.nova-pro-v1:0    (most capable, higher cost)
# Use us. or eu. prefix for cross-region inference (e.g., us.amazon.nova-lite-v1:0)
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")

# Cross-region inference configuration
# Multiple regions for failover when throttled
BEDROCK_REGIONS = os.getenv("BEDROCK_REGIONS", "us-west-2,us-east-1,us-east-2").split(",")

# Initialize Bedrock clients for all configured regions
bedrock_clients = {
    region: boto3.client(service_name="bedrock-runtime", region_name=region)
    for region in BEDROCK_REGIONS
}

# Track current region index for round-robin/failover
current_region_index = 0

# Memory storage configuration
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "../memory")

# Initialize S3 client if needed
if USE_S3:
    s3_client = boto3.client("s3")


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class Message(BaseModel):
    role: str
    content: str
    timestamp: str


# Memory management functions
def get_memory_path(session_id: str) -> str:
    return f"{session_id}.json"


def load_conversation(session_id: str) -> List[Dict]:
    """Load conversation history from storage"""
    if USE_S3:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=get_memory_path(session_id))
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        # Local file storage
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []


def save_conversation(session_id: str, messages: List[Dict]):
    """Save conversation history to storage"""
    if USE_S3:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=get_memory_path(session_id),
            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        # Local file storage
        os.makedirs(MEMORY_DIR, exist_ok=True)
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        with open(file_path, "w") as f:
            json.dump(messages, f, indent=2)


def call_bedrock(conversation: List[Dict], user_message: str) -> str:
    """Call AWS Bedrock with conversation history and cross-region failover"""
    global current_region_index
    
    # Build messages in Bedrock format
    messages = []
    
    # Add system prompt as first user message (Bedrock convention)
    messages.append({
        "role": "user", 
        "content": [{"text": f"System: {prompt()}"}]
    })
    
    # Add conversation history (limit to last 10 exchanges to manage context)
    for msg in conversation[-20:]:  # Last 10 back-and-forth exchanges
        messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": [{"text": user_message}]
    })
    
    # Try each region with failover on throttling
    last_error = None
    regions_tried = 0
    
    while regions_tried < len(BEDROCK_REGIONS):
        region = BEDROCK_REGIONS[current_region_index]
        client = bedrock_clients[region]
        
        try:
            print(f"Calling Bedrock in region: {region}")
            
            # Call Bedrock using the converse API
            response = client.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=messages,
                inferenceConfig={
                    "maxTokens": 2000,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            )
            
            # Extract the response text
            return response["output"]["message"]["content"][0]["text"]
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'ThrottlingException':
                # Throttled - try next region
                print(f"Throttled in region {region}, trying next region...")
                last_error = e
                current_region_index = (current_region_index + 1) % len(BEDROCK_REGIONS)
                regions_tried += 1
                continue
                
            elif error_code == 'ValidationException':
                # Handle message format issues - don't retry
                print(f"Bedrock validation error: {e}")
                raise HTTPException(status_code=400, detail="Invalid message format for Bedrock")
                
            elif error_code == 'AccessDeniedException':
                print(f"Bedrock access denied in region {region}: {e}")
                # Try next region in case access is regional
                last_error = e
                current_region_index = (current_region_index + 1) % len(BEDROCK_REGIONS)
                regions_tried += 1
                continue
                
            else:
                print(f"Bedrock error in region {region}: {e}")
                raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")
    
    # All regions exhausted
    print(f"All {len(BEDROCK_REGIONS)} regions exhausted. Last error: {last_error}")
    raise HTTPException(
        status_code=429, 
        detail=f"All Bedrock regions throttled. Please wait before trying again. Regions tried: {BEDROCK_REGIONS}"
    )


@app.get("/")
async def root():
    return {
        "message": "AI Digital Twin API (Powered by AWS Bedrock)",
        "memory_enabled": True,
        "storage": "S3" if USE_S3 else "local",
        "ai_model": BEDROCK_MODEL_ID
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "use_s3": USE_S3,
        "bedrock_model": BEDROCK_MODEL_ID,
        "bedrock_regions": BEDROCK_REGIONS,
        "current_region": BEDROCK_REGIONS[current_region_index]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Load conversation history
        conversation = load_conversation(session_id)

        # Call Bedrock for response
        assistant_response = call_bedrock(conversation, request.message)

        # Update conversation history
        conversation.append(
            {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()}
        )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save conversation
        save_conversation(session_id, conversation)

        return ChatResponse(response=assistant_response, session_id=session_id)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve conversation history"""
    try:
        conversation = load_conversation(session_id)
        return {"session_id": session_id, "messages": conversation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)