# import libraries
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from datetime import datetime, timezone


# Get current UTC time
data_saved_time = datetime.now(timezone.utc).isoformat()

# Read MongoDB credentials from environment
MONGO_URL = "mongodb+srv://priyankaregmi527:G1t5P5glW7LOGcr6@cluster0.bp8e1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_DB_NAME = "Metadata_collection"

# Connect to MongoDB Atlas
client = AsyncIOMotorClient(MONGO_URL)
db = client[MONGO_DB_NAME]

# Async function to save metadata
async def save_metadata_to_mongo(metadata: dict):
    metadata["upload_time"] = data_saved_time
    await db["file_metadata"].insert_one(metadata)
    print(" All metadata stored sucessfully!")
