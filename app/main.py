# import libraries and routs
from fastapi import FastAPI
from app.api import upload, rag_agent

# initialize fast api
app = FastAPI(title = "RAG Backend System")

# define root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Agentic System"}

# register routers
app.include_router(upload.router, prefix="/api")
