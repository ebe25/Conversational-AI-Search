from openai import OpenAI
from app.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-3-small"):
    # Set dimensions=384 for 384-dim embeddings
    response = client.embeddings.create(input=[text], model=model, dimensions=384)
    embedding = response.data[0].embedding
    if len(embedding) != 384:
        raise ValueError(f"Embedding size {len(embedding)} does not match expected 384 dimensions.")
    return embedding
