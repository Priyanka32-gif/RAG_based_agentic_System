from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.tools import Tool
from app.services.memory import get_memory
from app.services.tools import search_docs
import os
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from functools import partial



load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=OPENAI_KEY
)


def get_tools():
    search_fn = partial(search_docs, embedding_model=embedding_model)
    return [
        Tool(
            name="SearchDocs",
            func=search_fn,
            description=(
                "Search uploaded documents using semantic similarity.\n"
                "Format: 'cosine::<query>' or 'dot::<query>'.\n"
                "If no method is provided, defaults to cosine."
            )
        )
    ]

# Custom ReAct prompt tailored for your RAG + interview booking task

custom_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are a helpful AI backend assistant that can:
- Answer user questions using tools provided.
- Search documents uploaded by users.
- Assist with interview booking by collecting full name, email, date, and time.
- Maintain conversational context and reason step by step.

Available tools:
{tools}

**Instructions:**

Follow this strict format for your output at each step:

Thought: <your reasoning or internal thinking here>
Action: <one tool from [{tool_names}]>
Action Input: "<input to the tool>"

When you have the final answer and no further action is needed, respond like this:

Thought: <your final reasoning here>
Final Answer: <your answer to the user>

**Important Rules:**
- Do NOT include a “Question:” field.
- Only one Action per step.
- Do NOT add any extra text outside the Thought, Action, or Final Answer fields.
- When searching documents, your action format should be:
  Action: SearchDocs
  Action Input: "cosine::<query>" or "dot::<query>"
- When booking an interview, your action format should be:
  Action: BookInterview
  Action Input: {{"name": "<full name>", "email": "<email>", "date": "<YYYY-MM-DD>", "time": "<HH:MM>"}}
- If the user request cannot be fulfilled, explain why in a Final Answer.
- If the user wants to book an interview but hasn’t provided all details, ask for missing info in a Final Answer, rather than generating an Action.

**User input:** {input}

{agent_scratchpad}
"""
)


def get_agent(session_id: str):
    try:
        logger.debug("Initializing LLM...")
        llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=OPENAI_KEY
        )

        logger.debug("Initializing memory...")
        memory = get_memory(session_id=session_id)

        logger.debug("Setting up tools and prompt...")
        tools = get_tools()

        logger.debug("Creating agent...")
        agent = create_react_agent(llm=llm, tools=tools, prompt = custom_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,handle_parsing_errors=True)

        logger.debug("AgentExecutor created successfully.")
        return agent_executor

    except Exception as e:
        logger.error("Failed to initialize LangChain agent: %s", str(e), exc_info=True)
        raise RuntimeError(f"Agent init error: {str(e)}")


