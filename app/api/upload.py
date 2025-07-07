from fastapi import APIRouter, UploadFile, HTTPException, File, Form
import PyPDF2
import io
import os
from datetime import datetime, timezone
import logging
from starlette.concurrency import run_in_threadpool

from app.services.chunker import chunk_data_recursive, clean_text
from app.services.embedder import embed_chunks_openai
from app.services.vector_store import (
    store_embeddings_minimal,
    create_cosine_collection,
    create_dot_collection,
    COSINE_COLLECTION,
    DOT_COLLECTION
)
from app.services.mango_db import save_metadata_to_mongo
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

# Initialize OpenAI embeddings once
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

@router.post("/upload")
async def upload_document(
    uploaded_file: UploadFile = File(...),
    search_method: str = Form(...),
    email_id: str = Form(...)
):
    logger.info("File received: %s", uploaded_file.filename)
    upload_time = datetime.now(timezone.utc).isoformat()

    if not uploaded_file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed.")

    content = await uploaded_file.read()

    if uploaded_file.filename.endswith(".txt"):
        text = content.decode("utf-8")

    elif uploaded_file.filename.endswith(".pdf"):
        text = ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    else:
        text = ""

    logger.info("Cleaning text...")
    text = clean_text(text)

    chunks = chunk_data_recursive(text)

    logger.info("Running OpenAI embeddings...")
    embeddings = await run_in_threadpool(
        embed_chunks_openai,
        chunks,
        embedding_model
    )

    search_method = search_method.strip().lower()

    if search_method == 'cosine':
        create_cosine_collection()
        collection_name = COSINE_COLLECTION
    elif search_method == 'dot':
        create_dot_collection()
        collection_name = DOT_COLLECTION
    else:
        raise ValueError("Invalid search method. Choose 'cosine' or 'dot'.")

    logger.info("Storing embeddings into vector DB...")
    store_embeddings_minimal(chunks, embeddings, collection_name)

    metadata = {
        "session_id": email_id.strip().lower(),
        "file_name": uploaded_file.filename,
        "total_chunks": len(chunks),
        "embedding_shape": embeddings.shape,
        "embedding_model": "text-embedding-3-small",
        "sample_embedding": embeddings[0].tolist() if len(embeddings) > 0 else [],
        "upload_time": upload_time
    }
    logger.info("Saving metadata to MongoDB...")
    saved_metadata = await save_metadata_to_mongo(metadata)

    return saved_metadata
