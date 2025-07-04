import os
from app.audits.fetch_audits import fetch_all_audits
from app.config import QDRANT_COLLECTION
from app.pdf.chunker import langchain_chunk
from app.pdf.embedder import get_embedding
from app.pdf.uploader import upload_to_qdrant


def process_all_audits():
    audits = fetch_all_audits()
    print(f"Found {len(audits)} audits in DB.")
    
    for audit in audits:
        audit_id = str(audit.get("_id"))
        
        try:
            # Extract text from audit content
            title =  audit.get("title", "")
            status = audit.get('status', '')
            repeatCycle= audit.get('repeatCycle', '')

            # Build content sections
            content_parts = []
            
            if title:
                content_parts.append(f"Training Title: {title}")
            if status:
                content_parts.append(f"Status: {status}")
            if repeatCycle:
                content_parts.append(f"Repeat Cycle: {repeatCycle}")

            # Join all parts with double newlines for better separation
            text = "\n\n".join(content_parts)
            
            chunks = langchain_chunk(text)
            print(f"📦 Chunked text into {len(chunks)} parts.")
            
            embedded = [(chunk, get_embedding(chunk)) for chunk in chunks]
            
            # Upload to Qdrant with updated signature
            upload_to_qdrant(
                chunks=embedded,
                meta=audit,  # Pass full audit object
                collection=QDRANT_COLLECTION,
                module_type="audit"
            )
            
            print(f"✅ Uploaded {len(chunks)} chunks for Audit {audit_id}.")
            
        except Exception as e:
            print(f"❌ Failed to process Audit {audit_id}: {e}")