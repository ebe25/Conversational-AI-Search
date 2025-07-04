from app.db.mongo import db
from bson import ObjectId


def fetch_all_forms():
    forms_collection = db["forms"]
    documents = list(forms_collection.find({"entityId": ObjectId("67e58254b40e27710ecc0ee3")}))
    return documents

def fetch_forms_by_type(form_type=None):
    forms_collection = db["forms"]
    query = {"entityId": ObjectId("67e58254b40e27710ecc0ee3")}
    
    if form_type:
        query["formType"] = form_type
    
    documents = list(forms_collection.find(query))
    return documents