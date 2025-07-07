# app/services/chunker.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

# Function to chunk data with recursive chunking
def chunk_data_recursive(text, chunk_size=1000, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = [Document(page_content=text)]
    chunks = text_splitter.split_documents(documents)
    # The output of split_documents is a list of Document objects
    # Extract the text from each Document
    chunk_texts = [doc.page_content for doc in chunks]
    return chunk_texts

# Function to clean text
def clean_text(raw_text):
    # 1. Replace multiple newlines with a space
    text = re.sub(r'\n+', ' ', raw_text)

    # 2. Remove page numbers (e.g., "103\n Printed atHMGPress,...")
    text = re.sub(r'\b\d{1,3}\b(?=\s)', '', text)

    # 3. Remove special characters except periods (.,!?)
    text = re.sub(r'[\\\/]', '', text)

    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
