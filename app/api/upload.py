from fastapi import APIRouter, UploadFile, HTTPException, File, Form
import PyPDF2
import io
from app.services.chunker import chunk_data_recursive, chunk_data_semantic, clean_text
from sentence_transformers import SentenceTransformer
from app.services.embedder import  embed_chunks
from starlette.concurrency import run_in_threadpool
from datetime import datetime, timezone
from app.services.vector_store import store_embeddings_minimal, create_cosine_collection, create_dot_collection, search_cosine, search_dot, COSINE_COLLECTION, DOT_COLLECTION
from app.services.mango_db import save_metadata_to_mongo


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

@router.post("/upload")
async def upload_document(uploaded_file: UploadFile = File(...), chunking_method: str = Form(...), search_method: str = Form(...)):

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

    chunking_method = chunking_method.strip().lower()

    if chunking_method == 'recursive':
        chunks = chunk_data_recursive(text)
    elif chunking_method == 'semantic':
        chunks = await run_in_threadpool(chunk_data_semantic, text, model)
    else:
        raise HTTPException(status_code=400, detail="Invalid chunking method selected.")

    logger.info("Running embedding...")
    embeddings = await run_in_threadpool(embed_chunks, chunks, model)

    search_method = search_method.strip().lower()

    # Decide collection based on search method
    if search_method == 'cosine':
        create_cosine_collection()
        collection_name = COSINE_COLLECTION
    elif search_method == 'dot':
        create_dot_collection()
        collection_name = DOT_COLLECTION
    else:
        raise ValueError("Invalid search method. Choose 'cosine' or 'dot'.")

    logger.info("Storing embeddings...")
    store_embeddings_minimal(chunks, embeddings, collection_name)

    metadata =  {
        "file_name": uploaded_file.filename,
        "chunking_method": chunking_method,
        "total_chunks": len(chunks),
        "embedding_shape": embeddings.shape,
        "embedding_model":"all-MiniLM-L6-v2",
        "sample_embedding": embeddings[0].tolist() if len(embeddings) > 0 else [],
        "upload_time": upload_time
    }
    logger.info("Saving metadata to MongoDB...")
    await save_metadata_to_mongo(metadata)


