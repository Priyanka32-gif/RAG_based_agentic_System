# import libraries
from pydantic import BaseModel, EmailStr  # Pydantic base class and email validator
from typing import List, Optional         # For optional fields, list support
from datetime import datetime             # DateTime field for timestamping

class UploadMetadata(BaseModel):
    file_name: str                        # Name of the uploaded file (e.g., "doc.pdf")
    chunking_method: str                  # How the file was chunked (e.g., "recursive", "semantic")
    upload_time: Optional[datetime] = None  # Auto-populated upload time (optional)

class InterviewBooking(BaseModel):
    full_name: str                        # User's full name
    email: EmailStr                       # Validated email address
    date: str                             # Interview date in YYYY-MM-DD
    time: str                             # Interview time in HH:MM (24-hour format)

class QueryRequest(BaseModel):
    question: str                         # The user's input question
    session_id: Optional[str] = None      # Session ID to track memory context (if used)

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]  # list of embedding vectors
