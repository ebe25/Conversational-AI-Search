from app.db.mongo import db
from bson import ObjectId

ENTITY_ID = ObjectId("67e58254b40e27710ecc0ee3")

def fetch_documents_by_collection(collection_name, filters=None):
    """Generic function to fetch documents from any collection"""
    collection = db[collection_name]
    query = {"entityId": ENTITY_ID}
    
    if filters:
        query.update(filters)
    
    documents = list(collection.find(query))
    return documents

def fetch_all_trainings():
    return fetch_documents_by_collection("tps")

def fetch_all_forms():
    return fetch_documents_by_collection("forms")

def fetch_all_tasks():
    return fetch_documents_by_collection("tasks")

def fetch_all_audits():
    return fetch_documents_by_collection("audits")