from app.db.mongo import db
from bson import ObjectId


def fetch_all_trainings():
    training_collection = db["tps"]
    documents = list(training_collection.find({"entityId": ObjectId("67e58254b40e27710ecc0ee3")}))
    return documents