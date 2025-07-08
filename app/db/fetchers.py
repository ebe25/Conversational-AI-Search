from app.db.mongo import db
from bson import ObjectId
from typing import List
import os
from hashids import Hashids


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

def write_chat_record(chat_payload):
    """Insert a chat record into the chat collection"""
    collection = db["chatHistorys"]
    result = collection.insert_one(chat_payload)
    return str(result.inserted_id)

def decode_object_id(encoded_id: str) -> str:
    """Decode an object ID if needed - implement your specific decoding logic here"""
    # This is a placeholder - replace with your actual decoding logic
    return encoded_id

def get_user_locations(user_id: ObjectId) -> List[dict]:
    """Get locations associated with a user"""
    # This is a placeholder - replace with your actual location retrieval logic
    locations = list(db["locations"].find({"users": user_id}))
    return locations

def is_null(obj) -> bool:
    """Check if an object is None"""
    return obj is None

def validate_sop(sop_id: str, user_id: str) -> bool:
    """
    Validate if a user is authorized to access a specific SOP.
    
    Args:
        sop_id: The ID of the SOP to validate
        user_id: The ID of the user requesting access
    
    Returns:
        bool: True if the user is authorized, False otherwise
    """
    try:
        # Convert IDs to ObjectId
        decoded_sop_id = decode_object_id(sop_id)
        user_id_obj = ObjectId(user_id)
        
        # Find SOP and user documents
        sop = db["sops"].find_one({"_id": ObjectId(decoded_sop_id)})
        user = db["users"].find_one({"_id": user_id_obj})
        
        # Get user locations
        locations = get_user_locations(user_id_obj)
        parent_ids = [str(loc["_id"]) for loc in locations]
        
        # Check if SOP exists
        if is_null(sop):
            return False
            
        # Check entity ID match
        if str(sop.get("entityId")) != str(user.get("entityId")):
            return False
            
        # Check admin privileges, public visibility, or creator access
        if (user.get("authRole") in ["superadmin", "masteradmin", "admin"] or 
                sop.get("visibility") == "public" or 
                str(sop.get("createdBy")) == str(user_id_obj)):
            return True
            
        # Check private visibility rules
        if sop.get("visibility") == "private":
            # Check if user is explicitly listed
            visible_to_users = sop.get("visibleTo", {}).get("users", [])
            if str(user_id_obj) in [str(uid) for uid in visible_to_users]:
                return True
                
            # Check location and role permissions
            loc_visibility = [str(loc) for loc in sop.get("visibleTo", {}).get("locations", [])]
            loc_intersection = list(set(loc_visibility) & set(parent_ids))
            
            condition = sop.get("visibleTo", {}).get("condition")
            roles = sop.get("visibleTo", {}).get("roles", [])
            
            # OR condition: user has matching location OR matching role
            if (condition == "or" and 
                    (len(loc_intersection) > 0 or user.get("role") in roles)):
                return True
                
            # AND condition: user has matching location AND matching role
            elif (condition == "and" and 
                    len(loc_intersection) > 0 and 
                    user.get("role") in roles):
                return True
            else:
                return False
                
        return True
        
    except Exception as e:
        print(f"Error validating SOP access: {e}")
        return False


def encode_object_id(id_str: str) -> str:
    """
    Encode a potentially encoded object ID using Hashids
    
    Args:
        id_str: The ID to Encode, either a raw ObjectId or a hashid
    
    Returns:
        str: The Encoded ObjectId string
    """
    # Define the allowed characters (same as in your JS version)
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    
    # Create a Hashids instance
    encode_secret = os.environ.get("ENCODE_SECRET", "default-secret")
    hashids = Hashids(salt=encode_secret, min_length=0, alphabet=allowed_chars)
    
    # Check if id is already a valid ObjectId
    try:
        # If this doesn't raise an exception, it's already a valid ObjectId
        ObjectId(id_str)
        return id_str
    except:
        # If it's not a valid ObjectId, try to decode it
        try:
            decoded_hex = hashids.decode_hex(id_str)
            return decoded_hex
        except:
            # If decoding fails, return the original ID
            return id_str