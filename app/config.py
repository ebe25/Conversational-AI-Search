import os
from dotenv import load_dotenv

load_dotenv() 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")