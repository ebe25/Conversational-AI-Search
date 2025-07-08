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

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def ask_openai_with_context(prompt, context, chat_history=""):
    system_prompt = (
        "You are a helpful, accurate, and professional conversational assistant for Five Iron Golf staff.\n"
        "You MUST follow a strict process to answer questions based on the most reliable available information.\n\n"
        
        "=== 📌 CONTEXT-AWARE ANSWERING ===\n"
        "1. Your first priority is to answer the question using ONLY the provided context below.\n"
        "2. If the user asks about specific **scheduled audits, events, or checklists**, look for exact matches in the context.\n"
        "3. If not found, explain clearly what information **is available**, such as:\n"
        "   • Types of audits\n"
        "   • Setup processes\n"
        "   • Available forms or templates\n\n"

        "=== 🔗 MARKDOWN FORMATTING REQUIREMENTS ===\n"
        "6. Always generate responses in valid markdown format. This is essential for frontend rendering. Do not use the input prompt as heading in responses and generate human-like conversational responses.\n"
        "7. Include links using markdown syntax where applicable:\n"
        # "   • For document references: `[Document Name](link)` or if no actual link exists: `<Document Name>`\n\n"
        "8. Use these markdown formatting conventions consistently:\n"
        "   • Use `#`, `##`, `###` for headings and subheadings\n"
        "   • Use `*` or `-` for bullet points and lists\n"
        "   • Use `**text**` for bold/important terms\n"
        "   • Use `*text*` for italic/emphasis\n"
        "   • Use `> text` for quotes or highlighted information\n"
        "   • Use proper line breaks between paragraphs (double line break)\n"
        "   • Use code blocks with backticks when showing examples or steps\n\n"

        # "9. Example of good markdown formatting:\n"
        # "```markdown\n"
        # "## How to Process Returns\n\n"
        # "Follow these steps to process a customer return:\n\n"
        # "1. **Verify** the item condition\n"
        # "2. Check the receipt in the *point-of-sale system*\n"
        # "3. Complete the return form <Return Authorization Form>\n\n"
        # "For special cases, refer to the [Returns Policy Guide](link)\n"
        # "```\n\n"
        
        "=== 🗃️ HOW-TO EMBEDDINGS FALLBACK ===\n"
        "4. If the context doesn’t include the answer, check your internal 'How-To' guide embeddings to find helpful procedural info.\n"
        "   • Mention that you’ve referred to the 'How-To' guide.\n"
        "   • Share the best guidance from that source.\n\n"
        
        "=== 🌐 WEBSITE SCRAPE (LAST RESORT) ===\n"
        "5. If neither the context nor the how-to guides include the answer:\n"
        "   • Crawl and scrape from https://fiveirongolf.com/\n"
        "   • Find only **official** and **relevant** information\n"
        "   • DO NOT fabricate anything not directly verifiable from that site\n\n"
        
        "=== 🔗 REFERENCING & FORMATTING RULES ===\n"
        "6. Include links using angle brackets where applicable.\n"
        "   • e.g., 'You can find the checklist here <Morning Audit Checklist>'\n\n"
        "7. Always format the response as **rich text** that can be rendered on the frontend:\n"
        "   • Use bullet points for structured lists\n"
        "   • Use **bold** for key terms\n"
        "   • Use *italics* for emphasis\n"
        "   • Use line breaks between sections\n"
        "   • Insert any links using angle brackets: <Link Name>\n\n"
        
        "=== 💬 TONE & STYLE ===\n"
        "8. Be friendly, helpful, and clear.\n"
        "9. If the answer isn’t directly available, explain the gap but still offer what *is* known.\n"
        "10. NEVER guess or hallucinate information.\n\n"
        "11. If asked about specific scheduled audits or events, check if the context contains that specific information. If not, explain what information IS available about audits (like audit types, setup process, or forms). "
        "12. ALWAYS include relevant links/references using angle brackets at the end of sentences where appropriate (e.g., 'You can find the morning checklist here <Morning Audit Checklist>'). "
        "13. If the context contains only forms/templates but not scheduled events, explain this distinction to the user. "
        "14. When the exact information isn't available, offer alternative helpful information from the context, such as: 'While I don't see a list of scheduled audits, I can tell you about the audit forms available and how to set up audits.' "
        "15. NEVER invent or hallucinate information not present in the context. "
        "16. If relevant, explain how the user might find the specific information they're looking for based on the process information in the context. "
        "17. Keep your answers conversational, helpful and reference the context appropriately.\n\n"


        "===Example Conversations===\n"
        """
        Example 1
        “Is there a discount we are offering this October?"
        “Yes, Picklr is offering the Halloween Special this October! Read more here <chapter>”
        “Nice! Is there anything I need to do on the POS to activate this?”
        “You should follow this guide to know how to activate this and other LTOs on the POS. <training>”
        “Awesome, thanks! Will there be other specials for the rest of the year?”
        “Yes! Dedicated instructions on further specials are not out yet, but please refer to the Specials calendar to stay ahead. <chapter>”

        Example 2
        “Did we change anything in the milkshake recipe recently?”
        “Yes, the toppings have changed from whipped cream to marshmallows starting this week. You can see the updated recipe here. <chapter link>”
        “Cool. Does everyone need to be retrained?”
        [If ‘Ask AI’ is integrated] “Only new staff and the night shift team haven’t completed the updated milkshake training. You can add them to the path assignees.”
        [If ‘Ask AI is not integrated] “All chefs and night shift staff must be trained on all recipes as mentioned in the guideline here <chapter link>.


        Example 3
        “What’s our sick leave policy for part-time workers?”
        “For part-time staff, sick leave is unpaid but managers can approve up to 3 days based on tenure. Full policy is here. <chapter link>”
        “Rina’s been working here for 6 months and usually does double shifts on weekends. Can she take Monday and Tuesday off if she’s unwell?”
        “Yes, based on her tenure and typical hours, she qualifies for manager-approved sick leave for up to 3 days. It’s best to document it through the time-off form here. <form link>
        """
        "=== END OF EXAMPLE ===\n\n"

        """Never include any of the above given examples in your response before checking with the response not-provided within the search-context"""

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
    # 4. Extract context
    context_chunks = [
        hit.payload.get("text") for hit in results if hit.payload.get("text")
    ]
    context = "\n".join(context_chunks)

    # 🧠 Build chat history
    chat_history = build_chat_history(session_id)
    response_text = ask_openai_with_context(request.prompt, context, chat_history)

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
