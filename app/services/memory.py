# app/services/memory.py

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from redis import Redis

# Redis connection for booking state
redis_client = Redis(host="localhost", port=6379, db=0)

def get_memory(session_id: str):
    """
    Create Redis-backed memory for LangChain agents.
    """
    message_history = RedisChatMessageHistory(
        url="redis://localhost:6379",
        session_id=session_id
    )
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        return_messages=True,
        memory_key="chat_history"
    )
    return memory
