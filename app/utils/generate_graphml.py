import pandas as pd
import networkx as nx
import os

def generate_graphml_from_csv(csv_path, output_path):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()

    # Add nodes
    for _, row in df.iterrows():
        node_id = str(row["id"])
        node_attrs = row.to_dict()
        # Remove duplicate 'text' key if present
        if "text" in node_attrs:
            node_attrs.pop("text")
        G.add_node(node_id, text=row.get("text", ""), **node_attrs)

    # Add edges for relationships
    for _, row in df.iterrows():
        node_id = str(row["id"])
        parent_id = str(row["parent_id"]) if pd.notnull(row.get("parent_id")) else None
        prev_chunk_id = str(row["prev_chunk_id"]) if pd.notnull(row.get("prev_chunk_id")) else None
        next_chunk_id = str(row["next_chunk_id"]) if pd.notnull(row.get("next_chunk_id")) else None

        if parent_id and parent_id != node_id:
            G.add_edge(parent_id, node_id, relation="has_step")
        if prev_chunk_id and prev_chunk_id != node_id:
            G.add_edge(prev_chunk_id, node_id, relation="next_step")
        if next_chunk_id and next_chunk_id != node_id:
            G.add_edge(node_id, next_chunk_id, relation="next_step")

    nx.write_graphml(G, output_path)
    print(f"✅ GraphML file generated at: {output_path}")

if __name__ == "__main__":
    # Update these paths as needed
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../delightree_prod_kg_form_chunks.csv"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data.graphml"))
    generate_graphml_from_csv(csv_path, output_path)