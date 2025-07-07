# import libraries
import numpy as np

def embed_chunks_openai(chunks, embedding_model):
    """
    Compute OpenAI embeddings for each chunk.
    """
    vectors = embedding_model.embed_documents(chunks)
    return np.array(vectors)
