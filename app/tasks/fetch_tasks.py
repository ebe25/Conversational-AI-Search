from app.db.mongo import db
from bson import ObjectId


def fetch_all_tasks():
    tasks_collection = db["tasks"]
    documents = list(tasks_collection.find({"entityId": ObjectId("67e58254b40e27710ecc0ee3")}))
    return documents

def fetch_tasks_by_status(status=None):
    tasks_collection = db["tasks"]
    query = {"entityId": ObjectId("67e58254b40e27710ecc0ee3")}
    
    if status:
        query["status"] = status
    
    documents = list(tasks_collection.find(query))
    return documents

def fetch_tasks_by_type(task_type=None):
    tasks_collection = db["tasks"]
    query = {"entityId": ObjectId("67e58254b40e27710ecc0ee3")}
    
    if task_type:
        query["taskType"] = task_type
    
    documents = list(tasks_collection.find(query))
    return documents