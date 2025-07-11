import os
import networkx as nx
from app.config import QDRANT_COLLECTION
from app.tasks.fetch_tasks import fetch_all_tasks
from app.pdf.chunker import langchain_chunk
from app.pdf.embedder import get_embedding
from app.pdf.uploader import upload_to_qdrant


def process_all_tasks():
    tasks = fetch_all_tasks()
    print(f"Found {len(tasks)} tasks in DB.")
    G = nx.DiGraph()
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../input"))
    os.makedirs(input_dir, exist_ok=True)
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
            previous_chunk_id = None
            for i, chunk in enumerate(chunks):
                chunk_node_id = f"task::{task_id}_chunk_{i}"
                G.add_node(chunk_node_id, label="Chunk", task_id=task_id, text=chunk, order=i)
                G.add_edge(f"task::{task_id}", chunk_node_id, relation="has_step")
                if previous_chunk_id:
                    G.add_edge(previous_chunk_id, chunk_node_id, relation="next_step")
                previous_chunk_id = chunk_node_id
                # Export chunk as .txt file
                txt_path = os.path.join(input_dir, f"task_{task_id}_chunk_{i}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(chunk)
            G.add_node(f"task::{task_id}", label="Task", title=task.get("title", ""), status=task.get("status", ""))
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
    graph_path = os.path.join(os.path.dirname(__file__), "../task_graph.graphml")
    nx.write_graphml(G, graph_path)
    print(f"\n🧠 Knowledge graph saved to: {graph_path}")