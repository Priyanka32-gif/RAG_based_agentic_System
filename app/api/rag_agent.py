from fastapi import APIRouter
from app.models.schemas import QueryRequest, InterviewBooking

router = APIRouter()

@router.post("/query")
async def rag_query(req: QueryRequest):
    return {"response": "Dummy RAG output"}

@router.post("/book-interview")
async def book_interview(data: InterviewBooking):
    return {"message": f"Interview booked for {data.full_name}"}
