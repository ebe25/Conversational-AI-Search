import os
import requests
import networkx as nx
from app.config import QDRANT_COLLECTION
from app.pdf.pdf_parser import parse_pdf
from app.pdf.chunker import langchain_chunk
from app.pdf.embedder import get_embedding
from app.pdf.uploader import upload_to_qdrant
from app.pdf.fetch_sops import fetch_all_sops

def download_pdf_from_url(url, local_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path

def process_all_sops():
    sops = fetch_all_sops()
    print(f"Found {len(sops)} SOPs in DB.")

    # 🌐 Create a global graph
    G = nx.DiGraph()

    for sop in sops:
        sop_id = str(sop.get("_id"))

        try:
            if sop.get("sopType") == "document":
                files = sop.get("files", [])
                if not files or not isinstance(files, list):
                    print(f"❌ No files array for SOP: {sop_id}")
                    continue
                s3_url = files[0].get("url") if "url" in files[0] else None
                if not s3_url:
                    print(f"❌ No S3 link for SOP: {sop_id}")
                    continue

                local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sops"))
                os.makedirs(local_dir, exist_ok=True)
                local_pdf = os.path.join(local_dir, f"{sop_id}.pdf")

                download_pdf_from_url(s3_url, local_pdf)
                text = parse_pdf(local_pdf)

            elif sop.get("sopType") == "text":
                text = sop.get("raw_content", "") or sop.get("content", "")
            else:
                print(f"❌ Unknown sopType for SOP: {sop_id}")
                continue

            if not text:
                print(f"❌ No content found for SOP: {sop_id}")
                continue

            chunks = langchain_chunk(text)
            print(f"📦 Chunked text into {len(chunks)} parts.")

            embedded = []
            previous_chunk_id = None

            for i, chunk in enumerate(chunks):
                chunk_id = f"{sop_id}_chunk_{i}"
                embedding = get_embedding(chunk)
                embedded.append((chunk, embedding))

                # 🧠 Graph node: each chunk
                G.add_node(chunk_id, label="Chunk", sop_id=sop_id, text=chunk, order=i)

                # 🔗 Edge from SOP to chunk
                G.add_edge(f"sop::{sop_id}", chunk_id, relation="has_step")

                # 🔗 Edge between consecutive chunks
                if previous_chunk_id:
                    G.add_edge(previous_chunk_id, chunk_id, relation="next_step")
                previous_chunk_id = chunk_id

            # 🧠 Graph node: the SOP itself
            G.add_node(f"sop::{sop_id}", label="SOP", title=sop.get("title", ""), category=sop.get("category", ""))

            # Upload to Qdrant
            upload_to_qdrant(
                chunks=embedded,
                meta=sop,
                collection=QDRANT_COLLECTION,
                module_type="sop"
            )

            print(f"✅ Uploaded {len(chunks)} chunks for SOP {sop_id}.")

        except Exception as e:
            print(f"❌ Failed to process SOP {sop_id}: {e}")
        finally:
            if sop.get("sopType") == "document" and 'local_pdf' in locals() and os.path.exists(local_pdf):
                os.remove(local_pdf)

    # 💾 Save the graph to disk
    graph_path = os.path.join(os.path.dirname(__file__), "../sop_graph.graphml")
    nx.write_graphml(G, graph_path)
    print(f"\n🧠 Knowledge graph saved to: {graph_path}")
