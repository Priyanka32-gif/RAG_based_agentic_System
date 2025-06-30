# import libraries
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid
from qdrant_client.http import models as rest


# embedding dimention of "all-MiniLM-L6-v2" model is 384
EMBEDDING_DIM = 384

# name of the collection
COSINE_COLLECTION = "document_chunks_cosine"
DOT_COLLECTION = "document_chunks_dot"

# using qdart locally
client = QdrantClient("localhost", port=6333)

def create_cosine_collection():
    client.recreate_collection(
        collection_name=COSINE_COLLECTION,
        vectors_config=rest.VectorParams(
            size=EMBEDDING_DIM,
            distance=rest.Distance.COSINE
        )
    )
    print(f"Collection `{COSINE_COLLECTION}` created with COSINE similarity.")

def create_dot_collection():
    client.recreate_collection(
        collection_name=DOT_COLLECTION,
        vectors_config=rest.VectorParams(
            size=EMBEDDING_DIM,
            distance=rest.Distance.DOT
        )
    )
    print(f"Collection `{DOT_COLLECTION}` created with DOT similarity.")

# function to store embeddings
def store_embeddings_minimal(texts, embeddings, collection_name):
    """
    Store embeddings and chunk texts into Qdrant.
    """
    points = []
    for text, vector in zip(texts, embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={"text": text}
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(points)} vectors with texts in `{collection_name}`.")

# different search function for consine and dot
def search_cosine(query_vector, top_k=5):
    return client.search(
        collection_name=COSINE_COLLECTION,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )

def search_dot(query_vector, top_k=5):
    return client.search(
        collection_name=DOT_COLLECTION,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )

