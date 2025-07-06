from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.pdf.embedder import get_embedding
from app.pdf.uploader import client  # QdrantClient instance
from app.config import QDRANT_COLLECTION

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

def get_dynamic_id(payload):
    """Get the appropriate ID field based on module type"""
    module_type = payload.get("module_type", "sop")  # Default to sop if not specified
    id_field = f"{module_type}_id"
    return payload.get(id_field)

@app.post("/query")
def query_vector_db(request: QueryRequest):
    # 1. Embed the prompt
    query_embedding = get_embedding(request.prompt)
    # 2. Search Qdrant
    results = client.search(
        collection_name=QDRANT_COLLECTION, 
        query_vector=query_embedding,
        limit=request.top_k
    )
    print("results", results)
   
    # 3. Return the most relevant chunks and their meta
    return [
        {
            "text": hit.payload.get("text"),
            "module_type": hit.payload.get("module_type"),
            "document_id": get_dynamic_id(hit.payload),  # Dynamic ID based on type
            "createdAt": hit.payload.get("createdAt"),
            "updatedAt": hit.payload.get("updatedAt"),
            "entityId": hit.payload.get("entityId"),
            "score": hit.score
        }
        for hit in results
    ]
