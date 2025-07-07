from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from app.services.agent import get_agent
from app.models.schemas import QueryRequest
from app.services.mango_db import save_query_answer, save_booking_to_mongo
from app.services.email_service import send_email
import logging
from app.services.memory import  get_memory, redis_client
from starlette.concurrency import run_in_threadpool
import os

smtp_password = os.getenv("SMTP_PASSWORD")

print(smtp_password)


router = APIRouter()

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

BOOKING_STATE_PREFIX = "booking_state:"

# Helper to get booking state from Redis
def get_booking_state(session_id: str) -> dict:
    redis_key = f"{BOOKING_STATE_PREFIX}{session_id}"
    state = redis_client.hgetall(redis_key)
    return {k.decode(): v.decode() for k, v in state.items()} if state else {}

# Helper to save booking state
def save_booking_state(session_id: str, state: dict):
    redis_key = f"{BOOKING_STATE_PREFIX}{session_id}"
    str_state = {str(k): str(v) for k, v in state.items()}

    for k, v in str_state.items():
        redis_client.hset(redis_key, k, v)


# Helper to clear booking state
def clear_booking_state(session_id: str):
    redis_key = f"{BOOKING_STATE_PREFIX}{session_id}"
    redis_client.delete(redis_key)

@router.post("/query")
async def query_agent(request: QueryRequest):
    try:
        session_id = request.session_id.strip().lower()
        user_input = request.question.strip()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if user is in booking flow
        state = get_booking_state(session_id)

        if state:
            logger.debug("User is in booking flow: %s", state)

            # Check which field we’re expecting
            if "awaiting" in state:
                awaiting = state["awaiting"]

                # Save the field they just provided
                state[awaiting] = user_input

                # Determine next step
                if awaiting == "full_name":
                    state["awaiting"] = "email"
                    save_booking_state(session_id, state)
                    return {
                        "answer": "Please provide your email for the booking."
                    }
                elif awaiting == "email":
                    state["awaiting"] = "date"
                    save_booking_state(session_id, state)
                    return {
                        "answer": "Please provide the date for your interview (YYYY-MM-DD)."
                    }
                elif awaiting == "date":
                    state["awaiting"] = "time"
                    save_booking_state(session_id, state)
                    return {
                        "answer": "Please provide the time for your interview (HH:MM)."
                    }
                elif awaiting == "time":
                    # Booking complete
                    full_name = state.get("full_name")
                    email = state.get("email")
                    date = state.get("date")
                    time_str = state.get("time")

                    booking_data = {
                        "full_name": full_name,
                        "email": email,
                        "date": date,
                        "time": time_str,
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }

                    # Save to Mongo
                    await save_booking_to_mongo(booking_data)

                    # Send confirmation email
                    send_email(
                        sender_email="priyankaregmi527@gmail.com",
                        sender_password=smtp_password,
                        recipient_email=email,
                        subject="Interview Booking Confirmation",
                        body=f"Dear {full_name},\n\nYour interview has been booked for {date} at {time_str}.\n\nThanks!"
                    )

                    clear_booking_state(session_id)

                    return {
                        "answer": f"Booking confirmed for {full_name} on {date} at {time_str}. Confirmation email sent.",
                        "booking_details": booking_data
                    }

        # If user says "book interview" -> start booking flow
        if "book interview" in user_input.lower():
            logger.debug("Starting new booking flow for session: %s", session_id)
            new_state = {
                "awaiting": "full_name"
            }
            save_booking_state(session_id, new_state)
            return {
                "answer": "Sure! Let's book your interview. Please provide your full name."
            }

        # Otherwise handle as normal RAG query
        logger.debug("Processing normal query with RAG agent")

        # Call your GPT-4o-mini RAG agent
        agent = get_agent(session_id=session_id)

        response = await run_in_threadpool(agent.invoke, {"input": user_input})

        # Convert response to string
        response_text = response.get("output", str(response))

        await save_query_answer(
            session_id=session_id,
            question=user_input,
            answer={
                "output": response_text,
            },
            timestamp=timestamp
        )
        # return response to user
        return {
            "question": user_input,
            "answer": response_text
        }
    except Exception as e:
        logger.error("Error in query_agent: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

