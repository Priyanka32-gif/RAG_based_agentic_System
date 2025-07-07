📚 RAG-Based Backend Assistant (FastAPI)
This project is a production-ready backend system built using FastAPI that implements a powerful RAG (Retrieval-Augmented Generation) agent with document understanding and interview booking capabilities.

✨ Key Features
📁 File Upload API
Upload .pdf or .txt files, automatically extract and clean text, chunk using recursive strategy, and generate embeddings using OpenAI (text-embedding-ada-002).

🧠 Vector Storage & Retrieval
Embeddings are stored in Qdrant (with support for both cosine and dot-product similarity search). You can query documents using a semantic agent powered by LangChain.

🤖 Agentic RAG Querying
User queries are answered by a LangChain ReAct agent (not using RetrievalQA), with Redis-based conversational memory for context preservation.

📅 Interview Booking via Natural Language
Users can book interviews simply by chatting (e.g., "I want to book an interview. My name is Alice..."). The system extracts intent and details, stores them in MongoDB, and sends a confirmation email via SMTP.

🧾 Metadata Tracking
Metadata like file name, number of chunks, embedding shape, and model info are saved to MongoDB for traceability.

🛠️ Tech Stack
Backend Framework: FastAPI

LLM & Embeddings: OpenAI GPT-4o / text-embedding-ada-002

Vector DB: Qdrant

RAG Engine: LangChain with custom ReAct Agent

Databases: MongoDB (for metadata, Q&A, booking)

Memory: Redis

Email: SMTP (Gmail configured)