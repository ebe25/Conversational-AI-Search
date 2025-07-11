import os
import networkx as nx
from app.forms.fetch_forms import fetch_all_forms
from app.pdf.chunker import langchain_chunk
from app.pdf.embedder import get_embedding
from app.pdf.uploader import upload_to_qdrant
from app.config import QDRANT_COLLECTION


def process_all_forms():
    forms = fetch_all_forms()
    print(f"Found {len(forms)} forms in DB.")
    G = nx.DiGraph()
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../input"))
    os.makedirs(input_dir, exist_ok=True)
    for form in forms:
        form_id = str(form.get("_id"))
        
        try:
            # Extract text from form content
            title = form.get("title", "")
            category = form.get("category", "")
            status = form.get("status", "")
            visibility = form.get("visibility", "")
            questions = form.get("questions", [])

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

            # Add questions and options in a readable format
            if questions:
                question_lines = []
                for idx, q in enumerate(questions):
                    label = q.get("label", "")
                    qtype = q.get("qType", "")
                    question_lines.append(f"Q{idx+1}: {label} (Type: {qtype})")
                    options = q.get("options", [])
                    if options:
                        for opt_idx, opt in enumerate(options):
                            opt_label = opt.get("label", "")
                            sub_title = opt.get("subTitle", "")
                            question_lines.append(f"    - Option {opt_idx+1}: {opt_label}" + (f" ({sub_title})" if sub_title else ""))
                if question_lines:
                    content_parts.append("Questions:\n" + "\n".join(question_lines))

            # Join all parts with double newlines for better separation
            text = "\n\n".join(content_parts)

            chunks = langchain_chunk(text)
            print(f"📦 Chunked text into {len(chunks)} parts.")
            
            embedded = [(chunk, get_embedding(chunk)) for chunk in chunks]
            
            previous_chunk_id = None
            for i, chunk in enumerate(chunks):
                chunk_node_id = f"form::{form_id}_chunk_{i}"
                G.add_node(chunk_node_id, label="Chunk", form_id=form_id, text=chunk, order=i)
                G.add_edge(f"form::{form_id}", chunk_node_id, relation="has_step")
                if previous_chunk_id:
                    G.add_edge(previous_chunk_id, chunk_node_id, relation="next_step")
                previous_chunk_id = chunk_node_id
                # Export chunk as .txt file
                txt_path = os.path.join(input_dir, f"form_{form_id}_chunk_{i}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(chunk)
            G.add_node(f"form::{form_id}", label="Form", title=form.get("title", ""), category=form.get("category", ""))
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
    graph_path = os.path.join(os.path.dirname(__file__), "../../data.graphml")
    nx.write_graphml(G, graph_path)
    print(f"\n🧠 Knowledge graph saved to: {graph_path}")