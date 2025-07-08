from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.pdf.embedder import get_embedding
from app.pdf.uploader import client  # QdrantClient instance
from app.config import QDRANT_COLLECTION, OPENAI_API_KEY
from openai import OpenAI
from app.db.fetchers import write_chat_record, encode_object_id, validate_sop
import uuid
from typing import Optional
from datetime import datetime
from app.db.mongo import db
from bson import ObjectId

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def ask_openai_with_context(prompt, context):
    system_prompt = (
        "You are a helpful assistant for Five Iron Golf staff. "
        "Answer the user's question based STRICTLY on the provided context. "
        "1. If asked about specific scheduled audits or events, check if the context contains that specific information. If not, explain what information IS available about audits (like audit types, setup process, or forms). "
        "2. ALWAYS include relevant links/references using angle brackets at the end of sentences where appropriate (e.g., 'You can find the morning checklist here <Morning Audit Checklist>'). "
        "3. If the context contains only forms/templates but not scheduled events, explain this distinction to the user. "
        "4. When the exact information isn't available, offer alternative helpful information from the context, such as: 'While I don't see a list of scheduled audits, I can tell you about the audit forms available and how to set up audits.' "
        "5. NEVER invent or hallucinate information not present in the context. "
        "6. If relevant, explain how the user might find the specific information they're looking for based on the process information in the context. "
        "7. Keep your answers conversational, helpful and reference the context appropriately.\n\n"
        f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


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

    userId = request.userId if request.userId else "anonymous"

    # 2. Embed the prompt
    query_embedding = get_embedding(request.prompt)
    # 3. Search Qdrant
    results = client.search(
        collection_name=QDRANT_COLLECTION, 
        query_vector=query_embedding,
        limit=request.top_k
    )

     # 4. Filter results based on SOP validation
    filtered_results = []
    unauthorized_sops = []
    for hit in results:
        module_type = hit.payload.get("module_type")
        document_id = get_dynamic_id(hit.payload)
        
        # Only validate SOPs and only if we have a real user
        if module_type == "sop" and document_id and userId != "anonymous":
            # Check if the user has access to this SOP
            if validate_sop(ObjectId(document_id), ObjectId(userId)):
                filtered_results.append(hit)
            else:
                unauthorized_sops.append(document_id)
        else:
            # For non-SOPs or anonymous users, include all results
            filtered_results.append(hit)

    if unauthorized_sops:
        # User tried to access something they don't have permission for
        response_text = "You are not authorized to view some of the requested content. Please contact your administrator if you need access."
        
        # Store chat in MongoDB
        chat_payload = {
            "sessionId": session_id,
            "query": request.prompt,
            "response": response_text,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
            "userId": userId,
        }
        write_chat_record(chat_payload)
        
        # Return unauthorized message
        return {
            "sessionId": session_id,
            "results": response_text,
        }
    # 4. Extract context
    context_chunks = [hit.payload.get("text") for hit in results if hit.payload.get("text")]
    context = "\n".join(context_chunks)
    response_text = ask_openai_with_context(request.prompt, context)
   

    # 5. Store chat in MongoDB
    chat_payload = {
        "sessionId": session_id,
        "query": request.prompt,
        "response": response_text,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
        "userId": userId,
    }
    write_chat_record(chat_payload)

    # 6. Return response with sessionId
    return {
        "sessionId": session_id,
        "results": response_text
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
                },
                "userId": {"$first": "$userId"}
            }
        },
        {
            "$project": {
                "sessionId": "$_id",
                "chats": 1,
                "userId": 1,
                "_id": 0
            }
        }
    ]
    sessions = list(db["chatHistorys"].aggregate(pipeline))
    return {"sessions": sessions}
