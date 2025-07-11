import os
import json
import datetime
from bson import ObjectId
from app.pdf.chunker import langchain_chunk
from app.pdf.embedder import get_embedding
from app.pdf.uploader import upload_to_qdrant
from app.config import QDRANT_COLLECTION
import networkx as nx

def load_json_files(directory_path):
    """
    Load all JSON files from the specified directory
    
    Args:
        directory_path: Path to the directory containing JSON files
        
    Returns:
        List of dictionaries containing file data
    """
    json_data = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    data['filename'] = filename  # Add filename for reference
                    json_data.append(data)
                    print(f"✅ Loaded: {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {str(e)}")
    
    return json_data

def merge_json_content(json_data):
    """
    Merge all JSON files into a single document
    
    Args:
        json_data: List of JSON objects
        
    Returns:
        Dictionary containing merged content and metadata
    """
    merged_content = []
    source_files = []
    all_urls = []
    
    for data in json_data:
        # Extract markdown content
        markdown = data.get('markdown', '')
        if markdown:
            # Add a header with the source file for context
            filename_without_ext = data['filename'].replace('.json', '')
            section_header = f"\n\n## Source: {filename_without_ext}\n\n"
            merged_content.append(section_header + markdown)
        
        # Collect metadata
        source_files.append(data['filename'])
        if 'metadata' in data and 'url' in data['metadata']:
            all_urls.append(data['metadata']['url'])
    
    # Create merged document
    merged_doc = {
        '_id': ObjectId(),
        'content': '\n'.join(merged_content),
        'source_files': source_files,
        'source_urls': all_urls,
        'content_type': 'how-to-guide',
        'title': 'Delightree How-To Guide - Merged Content',
        'description': 'Merged content from all how-to guide JSON files',
        'createdAt': datetime.datetime.now(),
        'updatedAt': datetime.datetime.now(),
        'entityId': 'how-to-guide-merged',
        'total_files': len(json_data),
        'total_urls': len(all_urls)
    }
    
    return merged_doc

def process_and_ingest_guides(directory_path=None):
    """
    Main function to process all JSON files in the how-to-guide directory
    
    Args:
        directory_path: Optional custom directory path. If None, uses current directory.
    """
    if directory_path is None:
        directory_path = os.path.dirname(os.path.abspath(__file__))
    
    print(f"🔍 Processing JSON files in: {directory_path}")
    
    # Step 1: Load all JSON files
    json_data = load_json_files(directory_path)
    if not json_data:
        print("❌ No JSON files found in directory")
        return
    
    print(f"📁 Found {len(json_data)} JSON files")
    
    # Step 2: Merge content
    print("🔄 Merging JSON content...")
    merged_doc = merge_json_content(json_data)
    
    print(f"📝 Merged content length: {len(merged_doc['content'])} characters")
    
    # Step 3: Chunk the content
    print("✂️ Chunking content...")
    chunks = langchain_chunk(
        merged_doc['content'], 
        chunk_size=1000,  # Larger chunks for guide content
        chunk_overlap=200
    )
    
    print(f"📊 Created {len(chunks)} chunks")
    
    # Step 4: Generate embeddings
    print("🧠 Generating embeddings...")
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            embedded_chunks.append((chunk, embedding))
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(chunks)} chunks")
        except Exception as e:
            print(f"❌ Error embedding chunk {i}: {str(e)}")
    
    print(f"✅ Generated {len(embedded_chunks)} embeddings")
    
    # Step 5: Upload to Qdrant
    print("☁️ Uploading to Qdrant...")
    try:
        upload_to_qdrant(
            chunks=embedded_chunks,
            meta=merged_doc,
            collection=QDRANT_COLLECTION,
            module_type="guide"
        )
        print("🎉 Successfully uploaded how-to guides to Qdrant!")
        
        # Print summary
        print("\n📋 Summary:")
        print(f"   • Files processed: {merged_doc['total_files']}")
        print(f"   • URLs included: {merged_doc['total_urls']}")
        print(f"   • Chunks created: {len(embedded_chunks)}")
        print(f"   • Collection: {QDRANT_COLLECTION}")
        print(f"   • Module type: guide")
        
    except Exception as e:
        print(f"❌ Error uploading to Qdrant: {str(e)}")
    
    # Build knowledge graph
    G = nx.DiGraph()
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../input"))
    os.makedirs(input_dir, exist_ok=True)
    previous_chunk_id = None
    for i, chunk in enumerate(chunks):
        chunk_node_id = f"guide::merged_chunk_{i}"
        G.add_node(chunk_node_id, label="Chunk", guide_id=merged_doc['entityId'], text=chunk, order=i)
        G.add_edge(f"guide::{merged_doc['entityId']}", chunk_node_id, relation="has_step")
        if previous_chunk_id:
            G.add_edge(previous_chunk_id, chunk_node_id, relation="next_step")
        previous_chunk_id = chunk_node_id
        # Export chunk as .txt file
        txt_path = os.path.join(input_dir, f"guide_{merged_doc['entityId']}_chunk_{i}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(chunk)
    G.add_node(f"guide::{merged_doc['entityId']}", label="Guide", title=merged_doc.get("title", ""))
    
    # Save graph to file
    graph_path = os.path.join(os.path.dirname(__file__), "../guide_graph.graphml")
    nx.write_graphml(G, graph_path)
    print(f"\n🧠 Knowledge graph saved to: {graph_path}")

def process_individual_guides(directory_path=None):
    """
    Alternative function to process each JSON file as a separate document
    """
    if directory_path is None:
        directory_path = os.path.dirname(os.path.abspath(__file__))
    
    print(f"🔍 Processing individual JSON files in: {directory_path}")
    
    json_data = load_json_files(directory_path)
    if not json_data:
        print("❌ No JSON files found in directory")
        return
    
    total_chunks = 0
    
    G = nx.DiGraph()
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../input"))
    os.makedirs(input_dir, exist_ok=True)
    
    for data in json_data:
        print(f"\n📄 Processing: {data['filename']}")
        
        # Create document metadata
        doc_meta = {
            '_id': ObjectId(),
            'content': data.get('markdown', ''),
            'filename': data['filename'],
            'source_url': data.get('metadata', {}).get('url', ''),
            'title': data.get('metadata', {}).get('title', data['filename']),
            'description': data.get('metadata', {}).get('description', ''),
            'content_type': 'how-to-guide',
            'createdAt': datetime.datetime.now(),
            'updatedAt': datetime.datetime.now(),
            'entityId': data['filename'].replace('.json', ''),
        }
        
        if not doc_meta['content']:
            print(f"   ⚠️ Skipping {data['filename']} - no markdown content")
            continue
        
        # Chunk content
        chunks = langchain_chunk(doc_meta['content'], chunk_size=800, chunk_overlap=150)
        print(f"   ✂️ Created {len(chunks)} chunks")
        
        # Generate embeddings
        embedded_chunks = []
        for chunk in chunks:
            try:
                embedding = get_embedding(chunk)
                embedded_chunks.append((chunk, embedding))
            except Exception as e:
                print(f"   ❌ Error embedding chunk: {str(e)}")
        
        if embedded_chunks:
            # Upload to Qdrant
            try:
                upload_to_qdrant(
                    chunks=embedded_chunks,
                    meta=doc_meta,
                    collection=QDRANT_COLLECTION,
                    module_type="guide"
                )
                total_chunks += len(embedded_chunks)
                print(f"   ✅ Uploaded {len(embedded_chunks)} chunks")
            except Exception as e:
                print(f"   ❌ Error uploading: {str(e)}")
        
        # Build knowledge graph
        previous_chunk_id = None
        for i, chunk in enumerate(chunks):
            chunk_node_id = f"guide::{doc_meta['entityId']}_chunk_{i}"
            G.add_node(chunk_node_id, label="Chunk", guide_id=doc_meta['entityId'], text=chunk, order=i)
            G.add_edge(f"guide::{doc_meta['entityId']}", chunk_node_id, relation="has_step")
            if previous_chunk_id:
                G.add_edge(previous_chunk_id, chunk_node_id, relation="next_step")
            previous_chunk_id = chunk_node_id
            # Export chunk as .txt file
            txt_path = os.path.join(input_dir, f"guide_{doc_meta['entityId']}_chunk_{i}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(chunk)
        G.add_node(f"guide::{doc_meta['entityId']}", label="Guide", title=doc_meta.get("title", ""))
    
    # Save graph to file
    graph_path = os.path.join(os.path.dirname(__file__), "../guide_graph.graphml")
    nx.write_graphml(G, graph_path)
    print(f"\n🧠 Knowledge graph saved to: {graph_path}")

if __name__ == "__main__":
    # You can choose which approach to use:
    
    # Option 1: Merge all files into one document
    print("=== MERGING ALL GUIDES INTO ONE DOCUMENT ===")
    process_and_ingest_guides()
    
    print("\n" + "="*50 + "\n")
    
    # # Option 2: Process each file individually
    # print("=== PROCESSING EACH GUIDE INDIVIDUALLY ===")
    # process_individual_guides()