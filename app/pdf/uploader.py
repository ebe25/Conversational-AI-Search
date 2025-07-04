import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.config import QDRANT_HOST, QDRANT_API_KEY
from bson import ObjectId

client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

def serialize_meta(meta):
    """Convert ObjectId and other non-serializable types to strings"""
    serialized = {}
    for key, value in meta.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, dict):
            serialized[key] = serialize_meta(value)  # Recursive for nested dicts
        elif isinstance(value, list):
            serialized[key] = [str(item) if isinstance(item, ObjectId) else item for item in value]
        else:
            serialized[key] = value
    return serialized

def upload_to_qdrant(chunks, meta, collection=QDRANT_HOST, module_type="sop"):
    """
    Upload chunks to Qdrant with dynamic metadata
    
    Args:
        chunks: List of (chunk, embedding) tuples
        meta: Metadata dictionary containing document info
        collection: Qdrant collection name
        module_type: Type of module (sop, training, form, task, audit)
    """
    # Ensure collection exists with correct vector size
    try:
        client.get_collection(collection)
    except Exception:
        recreate_collection(collection=collection, vector_size=384)

    # Serialize metadata to handle ObjectId and other types
    serialized_meta = serialize_meta(meta)
    
    # Create dynamic ID field name based on module type
    id_field = f"{module_type}_id"
    
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                id_field: str(serialized_meta["_id"]),  # Dynamic ID field
                "createdAt": serialized_meta.get("createdAt"),
                "updatedAt": serialized_meta.get("updatedAt"),
                "entityId": str(serialized_meta.get("entityId")),
                "module_type": module_type,  # Add module type for filtering
            }
        )
        for i, (chunk, embedding) in enumerate(chunks)
    ]

    client.upsert(collection_name=collection, points=points)
    print(f"✅ Uploaded {len(points)} points to collection '{collection}' with {id_field}")

def recreate_collection(collection="delightree_prod_docs", vector_size=1536):
    # Delete the collection if it exists, then create it with the correct vector size
    try:
        client.delete_collection(collection_name=collection)
        print(f"🗑️ Deleted existing collection: {collection}")
    except Exception:
        print(f"ℹ️ Collection {collection} did not exist, creating new one.")

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"✅ Created collection '{collection}' with vector size {vector_size}.")