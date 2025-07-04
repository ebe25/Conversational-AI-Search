import os
from app.forms.fetch_forms import fetch_all_forms
from app.pdf.chunker import langchain_chunk
from app.pdf.embedder import get_embedding
from app.pdf.uploader import upload_to_qdrant
from app.config import QDRANT_COLLECTION


def process_all_forms():
    forms = fetch_all_forms()
    print(f"Found {len(forms)} forms in DB.")
    
    for form in forms:
        form_id = str(form.get("_id"))
        
        try:
            # Extract text from form content
            title =  form.get("title", "")
            category = form.get("category", "")
            status = form.get("status", "")
            visibility = form.get("visibility", "")

            # Build content sections
            content_parts = []

            if title:
                content_parts.append(f"Form Title: {title}")
            if category:

                content_parts.append(f"Category: {category}")
            if status:
                content_parts.append(f"Status: {status}")
            if visibility:
                content_parts.append(f"Visibility: {visibility}")

            # Join all parts with double newlines for better separation
            text = "\n\n".join(content_parts)

            chunks = langchain_chunk(text)
            print(f"📦 Chunked text into {len(chunks)} parts.")
            
            embedded = [(chunk, get_embedding(chunk)) for chunk in chunks]
            
            # Upload to Qdrant with updated signature
            upload_to_qdrant(
                chunks=embedded,
                meta=form,  # Pass full form object
                collection=QDRANT_COLLECTION,
                module_type="form"
            )
            
            print(f"✅ Uploaded {len(chunks)} chunks for Form {form_id}.")
            
        except Exception as e:
            print(f"❌ Failed to process Form {form_id}: {e}")