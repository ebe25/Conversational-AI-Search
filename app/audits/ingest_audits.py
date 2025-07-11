import os
import networkx as nx
from app.audits.fetch_audits import fetch_all_audits
from app.config import QDRANT_COLLECTION
from app.pdf.chunker import langchain_chunk
from app.pdf.embedder import get_embedding
from app.pdf.uploader import upload_to_qdrant


def process_all_audits():
    audits = fetch_all_audits()
    print(f"Found {len(audits)} audits in DB.")
    G = nx.DiGraph()
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../input"))
    os.makedirs(input_dir, exist_ok=True)
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
            previous_chunk_id = None
            for i, chunk in enumerate(chunks):
                chunk_node_id = f"audit::{audit_id}_chunk_{i}"
                G.add_node(chunk_node_id, label="Chunk", audit_id=audit_id, text=chunk, order=i)
                G.add_edge(f"audit::{audit_id}", chunk_node_id, relation="has_step")
                if previous_chunk_id:
                    G.add_edge(previous_chunk_id, chunk_node_id, relation="next_step")
                previous_chunk_id = chunk_node_id
                # Export chunk as .txt file
                txt_path = os.path.join(input_dir, f"audit_{audit_id}_chunk_{i}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(chunk)
            G.add_node(f"audit::{audit_id}", label="Audit", title=audit.get("title", ""), status=audit.get("status", ""))
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
    graph_path = os.path.join(os.path.dirname(__file__), "../audit_graph.graphml")
    nx.write_graphml(G, graph_path)
    print(f"\n🧠 Knowledge graph saved to: {graph_path}")