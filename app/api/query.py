from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.pdf.embedder import get_embedding
from app.pdf.uploader import client  # QdrantClient instance
from app.config import QDRANT_COLLECTION
from app.db.fetchers import write_chat_record
import uuid
from typing import Optional
from datetime import datetime
from app.db.mongo import db


app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 3
    sessionId: Optional[str] = None
    userId: Optional[str] = None

def get_dynamic_id(payload):
    """Get the appropriate ID field based on module type"""
    module_type = payload.get("module_type", "sop")  # Default to sop if not specified
    id_field = f"{module_type}_id"
    return payload.get(id_field)

@app.post("/query")
def query_vector_db(request: QueryRequest):
    # 1. Handle sessionId
    session_id = request.sessionId if request.sessionId else str(uuid.uuid4())

    # 2. Embed the prompt
    query_embedding = get_embedding(request.prompt)
    # 3. Search Qdrant
    results = client.search(
        collection_name=QDRANT_COLLECTION, 
        query_vector=query_embedding,
        limit=request.top_k
    )
   
    # 4. Prepare response data
    response_data = [
        {
            "text": hit.payload.get("text"),
            "module_type": hit.payload.get("module_type"),
            "document_id": get_dynamic_id(hit.payload),
            "createdAt": hit.payload.get("createdAt"),
            "updatedAt": hit.payload.get("updatedAt"),
            "entityId": hit.payload.get("entityId"),
            "score": hit.score
        }
        for hit in results
    ]

    # 5. Store chat in MongoDB
    chat_payload = {
        "sessionId": session_id,
        "query": request.prompt,
        "response": response_data,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    }
    write_chat_record(chat_payload)

    # 6. Return response with sessionId
    return {
        "sessionId": session_id,
        "results": response_data
    }

@app.get("/sessions")
def get_sessions():
    """
    Aggregate chat records grouped by sessionId.
    Returns a list of sessions with their queries and responses.
    """
    pipeline = [
        {
            "$group": {
                "_id": "$sessionId",
                "chats": {
                    "$push": {
                        "query": "$query",
                        "response": "$response",
                        "createdAt": "$createdAt",
                        "updatedAt": "$updatedAt"
                    }
                }
            }
        },
        {
            "$project": {
                "sessionId": "$_id",
                "chats": 1,
                "_id": 0
            }
        }
    ]
    sessions = list(db["chatHistorys"].aggregate(pipeline))
    return {"sessions": sessions}
