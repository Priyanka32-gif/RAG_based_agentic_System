# import libraries
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from datetime import datetime, timezone
load_dotenv()

# Read MongoDB credentials from environment
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# Connect to MongoDB Atlas
client = AsyncIOMotorClient(MONGO_URL)
db = client[MONGO_DB_NAME]

# delete entry if db
#db["file_metadata"].delete_many({})

# Async function to save metadata
async def save_metadata_to_mongo(metadata: dict):
    # Get current UTC time
    data_saved_time = datetime.now(timezone.utc).isoformat()

    metadata["upload_time"] = data_saved_time
    result = await db["file_metadata"].insert_one(metadata)

    # Convert the ObjectId to string so it can be serialized by FastAPI
    metadata["_id"] = str(result.inserted_id)

    print(" All metadata stored sucessfully!")
    return metadata

# function to retrive metadata from mongo db
async def get_metadata_by_session_id(session_id: str):
    return await db["file_metadata"].find_one({"session_id": session_id})


def serialize_messages(messages):
    serialized = []
    for msg in messages:
        serialized.append({
            "type": type(msg).__name__,
            "content": msg.content
        })
    return serialized


async def save_query_answer(session_id: str, question: str, answer, timestamp: str):
    if isinstance(answer, dict) and "chat_history" in answer:
        answer["chat_history"] = serialize_messages(answer["chat_history"])

    doc = {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "timestamp": timestamp
    }
    await db["conversations"].insert_one(doc)
    print("Conversation stored successfully!")



async def save_booking_to_mongo(data: dict):
    await db["interview_bookings"].insert_one(data)
    print("Booking saved successfully!")
