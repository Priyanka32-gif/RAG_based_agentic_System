# import libraries
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
# use redis memory for conversational context
from redis import Redis

redis_client = Redis(host='localhost', port=6379, db=0)

message_history = RedisChatMessageHistory(redis_client=redis_client, session_id="user_session_id")
memory = ConversationBufferMemory(chat_memory=message_history, return_messages=True)
