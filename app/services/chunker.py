# import libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# function to chunk data with recursive chunking
def chunk_data_recursive(text, chunk_size = 1000, chunk_overlap = 10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    documents = [Document(page_content=text)]
    chunks = text_splitter.split_documents(documents)
    return chunks



hf_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def chunk_data_semantic(text, model, similarity_threshold=0.65, max_chunk_size=5, max_token_length=512):

    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)

    if not sentences:
        print("⚠️ No sentences found after tokenization.")
        return []

    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = []
    current_vecs = []

    for i, sent in enumerate(sentences):
        vec = embeddings[i].reshape(1, -1)

        if not current_chunk:
            current_chunk.append(sent)
            current_vecs.append(vec)
            continue

        avg_vec = np.mean(np.vstack(current_vecs), axis=0).reshape(1, -1)
        sim = cosine_similarity(avg_vec, vec)[0][0]

        candidate_chunk = current_chunk + [sent]
        chunk_text = " ".join(candidate_chunk)
        token_count = len(hf_tokenizer.tokenize(chunk_text))


        if sim >= similarity_threshold and len(candidate_chunk) < max_chunk_size and token_count < max_token_length:
            current_chunk.append(sent)
            current_vecs.append(vec)

        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_vecs = [vec]


    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def clean_text(raw_text):
    # 1. Replace multiple newlines with space
    text = re.sub(r'\n+', ' ', raw_text)

    # 2. Remove page numbers (e.g., "103\n Printed atHMGPress,...")
    text = re.sub(r'\b\d{1,3}\b(?=\s)', '', text)

    # 3. Remove special characters except periods (.,!?)
    text = re.sub(r'[\\\/]', '', text)

    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
