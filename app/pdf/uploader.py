import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.config import QDRANT_HOST, QDRANT_API_KEY
from bson import ObjectId
import regex

client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)


def sanitize_title(title: str) -> str:
    """Remove characters that can break markdown rendering."""
    return regex.sub(r"[\[\]\(\)`]", "", title).strip()


def serialize_meta(meta):
    """Recursively convert ObjectId and non-serializable types to safe strings."""
    serialized = {}
    for key, value in meta.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, dict):
            serialized[key] = serialize_meta(value)
        elif isinstance(value, list):
            serialized[key] = [
                str(item) if isinstance(item, ObjectId) else item for item in value
            ]
        else:
            serialized[key] = value

    # Optional sanitization and fallback defaults
    if "title" in serialized and serialized["title"]:
        serialized["title"] = sanitize_title(serialized["title"])
    if "url" in serialized and serialized["url"]:
        serialized["url"] = serialized["url"].strip()

    return serialized


def upload_to_qdrant(chunks, meta, collection=QDRANT_HOST, module_type="sop"):
    """
    Upload embedded chunks to Qdrant with full metadata, including link references.

    Args:
        chunks: List of (chunk_text, embedding) tuples
        meta: Dict containing document metadata (must include title + url for linking)
        collection: Qdrant collection name
        module_type: 'sop', 'training', 'form', 'task', 'audit', etc.
    """
    # Ensure the collection exists
    try:
        client.get_collection(collection)
    except Exception:
        recreate_collection(collection=collection, vector_size=384)

    serialized_meta = serialize_meta(meta)
    id_field = f"{module_type}_id"

    title = serialized_meta.get("title", "")
    url = serialized_meta.get("url", "")

    if not title or not url:
        print(
            "⚠️ Warning: Missing `title` or `url` in metadata. Links in markdown will not work properly."
        )
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                id_field: str(serialized_meta.get("_id")),
                "createdAt": serialized_meta.get("createdAt"),
                "updatedAt": serialized_meta.get("updatedAt"),
                "entityId": str(serialized_meta.get("entityId")),
                "module_type": module_type,
                "title": title,
                "url": url,
            },
        )
        for chunk, embedding in chunks
    ]

    client.upsert(collection_name=collection, points=points)
    print(
        f"✅ Uploaded {len(points)} points to collection '{collection}' with enriched metadata."
    )

def recreate_collection(collection="delightree_prod_docs", vector_size=1536):
    # Delete the collection if it exists, then create it with the correct vector size
    try:
        client.delete_collection(collection_name=collection)
        print(f"🗑️ Deleted existing collection: {collection}")
    except Exception:
        print(f"ℹ️ Collection {collection} did not exist, creating new one.")

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"✅ Created collection '{collection}' with vector size {vector_size}.")