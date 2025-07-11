from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.pdf.embedder import get_embedding
from app.pdf.uploader import client  # QdrantClient instance
from app.config import QDRANT_COLLECTION, OPENAI_API_KEY
from openai import OpenAI
from app.db.fetchers import write_chat_record
import uuid
from typing import Optional
from datetime import datetime
from app.db.mongo import db
import networkx as nx
import os

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def ask_openai_with_context(prompt, context, chat_history=""):
    system_prompt = (
        "You are a helpful, accurate, and professional conversational assistant for Five Iron Golf staff.\n"
        "You MUST follow a strict process to answer questions based on the most reliable available information.\n\n"
        
        "=== 📌 CONTEXT-AWARE ANSWERING (STRICT) ===\n"
        "1. Your first priority is to answer the question using ONLY the provided context below.\n"
        "2. If the user asks about specific **scheduled audits, events, or checklists**, look for exact matches in the context.\n"
        "3. If not found, explain clearly what information **is available**, such as:\n"
        "   • Types of audits\n"
        "   • Setup processes\n"
        "   • Available forms or templates\n"
        "4. NEVER hallucinate, guess, or invent information not present in the context.\n"
        "5. If a direct match isn't found, explain what similar or related information IS available in the context.\n"
        "6. If no related information exists in the context at all, simply state: 'I don't see information about [topic] in the bussiness. Would you like me to search for information on a related topic instead?'\n\n"
        
        "=== 🔍 KNOWLEDGE GRAPH NAVIGATION & CONVERSATION CONTINUITY ===\n"
        "1. Connect relevant nodes from the knowledge graph when answering questions.\n"
        "2. If the initial context doesn't contain a complete answer, mention other related information available.\n"
        "3. Example: 'I don't see specifics about X, but our knowledge graph contains related information about Y and Z. Would you like me to share that instead?'\n"
        "4. Only suggest information that actually appears in the context or graph nodes provided to you.\n"
        "5. ALWAYS acknowledge previous exchanges in the conversation. If the user's question follows up on a previous topic, acknowledge this connection.\n"
        "6. If the user's question changes topic entirely from previous exchanges, acknowledge the topic shift.\n"
        "7. Example of acknowledging: 'Regarding your question about fire safety hazards, I don't see specific information about that in our knowledge base. However, I notice you're now asking about...' or 'To follow up on our discussion about X, the information about Y is...'\n"
        "8. If the user responds with an affirmative answer (yes, yeah, sure, okay, absolutely, please do, go ahead, etc.) to your offer to search for related topics, you MUST:\n"
        "   • Acknowledge their response (e.g., 'Great! Based on your interest in [original topic]...')\n"
        "   • Extract the most relevant categories or topics from the knowledge graph context\n"
        "   • Present these as clear options for the user to explore further\n"
        "   • Example: 'Great! Based on your interest in fire safety, here are related categories I found in our knowledge base: 1) Safety Protocols, 2) Emergency Procedures, 3) Facility Maintenance. Which of these would you like to explore?'\n"
        "   • Continue this process, prompting the user with relevant categories, until the user selects a topic that leads to information actually present in the knowledge base or graph.\n"
        "   • Do not provide general or global information unless the user explicitly asks for it. If the user requests general info, reconfirm politely and explain why specific info could not be found in the relevant graph node cluster.\n\n"

        "=== 🔗 MARKDOWN FORMATTING REQUIREMENTS ===\n"
        "1. Always generate responses in valid markdown format. This is essential for frontend rendering. Do not use the input prompt as heading in responses and generate human-like conversational responses.\n"
        "2. Include links using markdown syntax where applicable.\n"
        "3. Use these markdown formatting conventions consistently:\n"
        "   • Use `#`, `##`, `###` for headings and subheadings\n"
        "   • Use `*` or `-` for bullet points and lists\n"
        "   • Use `**text**` for bold/important terms\n"
        "   • Use `*text*` for italic/emphasis\n"
        "   • Use `> text` for quotes or highlighted information\n"
        "   • Use proper line breaks between paragraphs (double line break)\n"
        "   • Use code blocks with backticks when showing examples or steps\n\n"
        
        "=== ⚠️ HANDLING MISSING INFORMATION ===\n"
        "1. When information isn't available, be direct but helpful: 'I don't see that information in our knowledge base.'\n"
        "2. Then offer: 'Would you like me to share what IS available on related topics?'\n"
        "3. If the user responds affirmatively (yes, yeah, sure, etc.), DO NOT provide generic information. Instead, extract relevant categories or topics from the context and present them as options, and continue this process until the user reaches information that is actually present in the knowledge base/graph.\n"
        "4. If the user requests information beyond the knowledge graph, politely confirm: 'The specific information isn't in our knowledge graph. Are you looking for general information on this topic? I should note that I can only provide verified information from our internal resources.'\n"
        "5. NEVER provide generic information when the answer isn't in the context. Instead, suggest searching for related topics that ARE in the context.\n\n"
        
        "=== 🔗 REFERENCING & FORMATTING RULES ===\n"
        "1. Include links using angle brackets where applicable.\n"
        "   • e.g., 'You can find the checklist here <Morning Audit Checklist>'\n"
        "2. ALWAYS include relevant links/references using angle brackets at the end of sentences where appropriate (e.g., 'You can find the morning checklist here <Morning Audit Checklist>').\n"
        "3. Always format the response as **rich text** that can be rendered on the frontend:\n"
        "   • Use bullet points for structured lists\n"
        "   • Use **bold** for key terms\n"
        "   • Use *italics* for emphasis\n"
        "   • Use line breaks between sections\n"
        "   • Insert any links using angle brackets: <Link Name>\n\n"
        
        "=== 💬 TONE & STYLE ===\n"
        "1. Be friendly, helpful, and clear.\n"
        "2. If the answer isn't directly available, explain the gap but still offer what *is* known.\n"
        "3. NEVER guess or hallucinate information.\n"
        "4. If asked about specific scheduled audits or events, check if the context contains that specific information. If not, explain what information IS available about audits (like audit types, setup process, or forms).\n"
        "5. If the context contains only forms/templates but not scheduled events, explain this distinction to the user.\n"
        "6. When the exact information isn't available, offer alternative helpful information from the context, such as: 'While I don't see a list of scheduled audits, I can tell you about the audit forms available and how to set up audits.'\n"
        "7. If relevant, explain how the user might find the specific information they're looking for based on the process information in the context.\n"
        "8. Keep your answers conversational, helpful and reference the context appropriately.\n\n"

        f"Context:\n{context}\n\n"
        f"Conversation so far:\n{chat_history}\n\n"
        f"User Question:\n{prompt}\n\n"
        f"Answer (Rich Text Response):"
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        max_tokens=1024,
        temperature=0.2,
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


def build_chat_history(session_id: str) -> str:
    chats = list(
        db["chatHistorys"].find({"sessionId": session_id}).sort("createdAt", 1)
    )
    history = ""
    for c in chats:
        user_msg = c.get("query", "").strip()
        ai_msg = c.get("response", "").strip()
        if user_msg:
            history += f"User: {user_msg}\n"
        if ai_msg:
            history += f"Assistant: {ai_msg}\n"
    return history.strip()

def load_active_knowledge_graph(graph_path=None):
    """
    Loads the active knowledge graph from a GraphML file.
    By default, loads sop_graph.graphml from the app directory.
    """
    if graph_path is None:
        graph_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sop_graph.graphml"))
    if not os.path.exists(graph_path):
        print(f"⚠️ Knowledge graph file not found: {graph_path}")
        return None
    return nx.read_graphml(graph_path)

def search_graph_for_context(graph, query, top_k=3):
    """
    Search the knowledge graph nodes for the most relevant context to the query.
    Uses simple keyword matching; can be replaced with embedding-based similarity.
    Returns a list of node texts.
    """
    if graph is None:
        return []

    # Gather all node texts
    node_texts = []
    for node_id, data in graph.nodes(data=True):
        text = data.get("text", "") or data.get("title", "")
        if text:
            node_texts.append((node_id, text))

    # Simple keyword match scoring (replace with embedding similarity for better results)
    scored = []
    query_lower = query.lower()
    for node_id, text in node_texts:
        score = sum(1 for word in query_lower.split() if word in text.lower())
        if score > 0:
            scored.append((score, node_id, text))
    # Sort by score descending, take top_k
    scored.sort(reverse=True)
    top_contexts = [text for _, _, text in scored[:top_k]]

    return top_contexts

def get_graph_context_from_results(results, graph, query, top_k=3):
    """
    Given vector search results and a knowledge graph,
    return a string context that includes the chunk text and related nodes,
    and also searches the graph for additional relevant context.
    """
    context_parts = []

    # Add context from vector search results
    if results:
        for hit in results:
            chunk_text = hit.payload.get("text", "")
            if chunk_text:
                context_parts.append(f"Chunk: {chunk_text}")

    # Add context from graph search
    graph_contexts = search_graph_for_context(graph, query, top_k)
    for gctx in graph_contexts:
        context_parts.append(f"Graph: {gctx}")

    return "\n".join(context_parts)

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
        limit=request.top_k,
    )

    # 4. Load active knowledge graph
    graph = load_active_knowledge_graph()  # You can pass a different path if needed

    # 5. Extract context from graph and vector results (graph is the "brain")
    context = get_graph_context_from_results(results, graph, request.prompt, request.top_k)

    # 🧠 Build chat history
    chat_history = build_chat_history(session_id)
    response_text = ask_openai_with_context(request.prompt, context, chat_history)

    # 6. Store chat in MongoDB
    chat_payload = {
        "sessionId": session_id,
        "query": request.prompt,
        "response": response_text,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
        "userId": userId,
    }
    write_chat_record(chat_payload)

    # 7. Return response with sessionId
    return {"sessionId": session_id, "results": response_text, "contentType": 'markdown'}


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
                        "updatedAt": "$updatedAt",
                    }
                },
                "userId": {"$first": "$userId"},
            }
        },
        {"$project": {"sessionId": "$_id", "chats": 1, "userId": 1, "_id": 0}},
    ]
    sessions = list(db["chatHistorys"].aggregate(pipeline))
    return {"sessions": sessions}
