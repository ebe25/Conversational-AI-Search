import os
from app.config import QDRANT_COLLECTION
from app.tasks.fetch_tasks import fetch_all_tasks
from app.pdf.chunker import langchain_chunk
from app.pdf.embedder import get_embedding
from app.pdf.uploader import upload_to_qdrant


def process_all_tasks():
    tasks = fetch_all_tasks()
    print(f"Found {len(tasks)} tasks in DB.")
    
    for task in tasks:
        task_id = str(task.get("_id"))
        
        try:
            # Extract text from task content
            title = task.get("title", "")
            description = task.get("description", "")
            status = task.get("status", "")
            repeatCycle = task.get("repeatCycle", "")
                # Build content sections
            content_parts = []
            
            if title:
                content_parts.append(f"Training Title: {title}")
            if status:
                content_parts.append(f"Status: {status}")
            if description:
                content_parts.append(f"Description: {description}")
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
                meta=task,  # Pass full task object
                collection=QDRANT_COLLECTION,
                module_type="task"
            )
            
            print(f"✅ Uploaded {len(chunks)} chunks for Task {task_id}.")
            
        except Exception as e:
            print(f"❌ Failed to process Task {task_id}: {e}")