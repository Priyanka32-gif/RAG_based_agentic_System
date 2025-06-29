# import libraries
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid
from qdrant_client.http import models as rest


# embedding dimention of "all-MiniLM-L6-v2" model is 384
EMBEDDING_DIM = 384

# name of the collection
DEFAULT_COLLECTION = "document_chunks"

# using qdart locally
client = QdrantClient("localhost", port=6333)

def create_collection(collection_name=DEFAULT_COLLECTION):
    """
    Create a new Qdrant collection for embeddings
    using cosine similarity.
    """

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=EMBEDDING_DIM,
            distance=rest.Distance.COSINE
            )
        )
    print(f"Collection `{collection_name}` created (or recreated).")

# function to store embeddings
def store_embeddings_minimal(texts, embeddings, collection_name=DEFAULT_COLLECTION):
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

# function to retrive anser to query
def search_embeddings(query_vector, top_k=5, collection_name=DEFAULT_COLLECTION):
    """
    Search Qdrant for top_k similar vectors
    using cosine similarity.
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=False
    )
    return search_result
