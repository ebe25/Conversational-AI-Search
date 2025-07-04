from app.db.mongo import db
from bson import ObjectId


def fetch_all_audits():
    audits_collection = db["audits"]
    documents = list(audits_collection.find({"entityId": ObjectId("67e58254b40e27710ecc0ee3")}))
    return documents

def fetch_audits_by_status(status=None):
    audits_collection = db["audits"]
    query = {"entityId": ObjectId("67e58254b40e27710ecc0ee3")}
    
    if status:
        query["status"] = status
    
    documents = list(audits_collection.find(query))
    return documents

def fetch_audits_by_type(audit_type=None):
    audits_collection = db["audits"]
    query = {"entityId": ObjectId("67e58254b40e27710ecc0ee3")}
    
    if audit_type:
        query["auditType"] = audit_type
    
    documents = list(audits_collection.find(query))
    return documents