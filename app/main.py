# import libraries and routs
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, rag_agent

# initialize fast api
app = FastAPI(title = "RAG Backend System")

# add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# define root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Agentic System"}

# register routers
app.include_router(upload.router, prefix="/api")
app.include_router(rag_agent.router, prefix="/api")
