import os
from app.trainings.fetch_tps import fetch_all_trainings
from app.pdf.chunker import langchain_chunk
from app.pdf.embedder import get_embedding
from app.pdf.uploader import upload_to_qdrant


def process_all_trainings():
    trainings = fetch_all_trainings()
    print(f"Found {len(trainings)} trainings in DB.")
    
    for training in trainings:
        training_id = str(training.get("_id"))
        
        try:
            # Extract text from training content
            title = training.get("title", "")
            status = training.get("status", "")
            description = training.get("description", "")
            repeatCycle = training.get("repeatCycle", "")

            # if not title:
            #     print(f"❌ No title found for Training: {training_id}")
                
            # if not status: 
            #     print(f"❌ No status found for Training: {training_id}")
                
            # if not description: 
            #     print(f"❌ No description found for Training: {training_id}")
                
            # if not repeatCycle: 
            #     print(f"❌ No repeatCycle found for Training: {training_id}")
                

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

            # if not content_parts:
            #     print(f"❌ No content found for Training: {training_id}")
                
            
            # Join all parts with double newlines for better separation
            text = "\n\n".join(content_parts)
            # Process the text
            chunks = langchain_chunk(text)
            print(f"📦 Chunked text into {len(chunks)} parts.")
            
            embedded = [(chunk, get_embedding(chunk)) for chunk in chunks]
            
            # Upload to Qdrant with updated signature
            upload_to_qdrant(
                chunks=embedded,
                meta=training,  # Pass full training object
                collection="delightree_prod",
                module_type="training"
            )
            
            print(f"✅ Uploaded {len(chunks)} chunks for Training {training_id}.")
            
        except Exception as e:
            print(f"❌ Failed to process Training {training_id}: {e}")