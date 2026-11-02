
"""LangGraph-powered agent orchestration."""

import os
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import build_tools
from src.database.csv_repository import CSVRepository
from src.rag.retriever import PolicyRetriever


# --------------------------------------------------
# LANGUAGE POLICY
# --------------------------------------------------

LANGUAGE_POLICY = """
LANGUAGE AND DATA TRANSLATION RULES:

1. Always respond in the language of the user's latest message.
2. Translate product descriptions and specifications.
3. CSV data may be stored in Brazilian Portuguese.
4. Never copy descriptions in their original language
   when the user communicates in another language.
5. Preserve product names, brands and model numbers.
6. Preserve prices, stock quantities and numerical values.
7. Keep the entire response in one language.
8. Apply these rules to all tool results and final responses.
"""


# --------------------------------------------------
# LLM INITIALIZATION
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_llm():
    """Initialize and reuse the local Ollama model."""

    return ChatOllama(
        model=os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:7b-instruct",
        ),
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ),
        temperature=0.1,
        keep_alive="30m",
    )


# --------------------------------------------------
# AGENT INITIALIZATION
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_agent():
    """Initialize and reuse the compiled LangGraph agent."""

    # Load operational data.
    data_dir = Path(
        os.getenv("DATA_DIR", "data")
    )

    repository = CSVRepository(data_dir)

    # Initialize the policy RAG retriever.
    policy_retriever = PolicyRetriever()

    # Register agent tools.
    tools = build_tools(
        repository,
        policy_retriever,
    )

    # Configure the system instructions.
    system_prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + LANGUAGE_POLICY
    )

    # Compile and return the agent.
    agent = create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=system_prompt,
    )

    return agent


# --------------------------------------------------
# RESPONSE TRANSLATION
# --------------------------------------------------

def translate_response(
    llm,
    response: str,
    language_instruction: str,
) -> str:
    """Translate the complete response into the target language."""

    translation_prompt = """
    You are a professional multilingual translator.

    Translate the complete assistant response into the target language.

    STRICT RULES:

    1. Translate every description and specification.
    2. Translate all Portuguese expressions.
    3. Never mix Portuguese, Spanish or English.
    4. Preserve product names, brands and model numbers exactly.
    5. Preserve prices, stock quantities and numerical values.
    6. Preserve technical material names such as Linden, Spruce,
       Sapele, Nato and Meranti.
    7. Preserve formatting, headings and bullet points.
    8. Do not add, remove or invent product information.
    9. Do not summarize the original response.
    10. Return only the translated response.
    """

    result = llm.invoke(
        [
            SystemMessage(
                content=translation_prompt
            ),
            HumanMessage(
                content=(
                    f"Target language: {language_instruction}\n\n"
                    f"Response to translate:\n{response}"
                )
            ),
        ]
    )

    content = result.content

    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        ).strip()

    return str(content).strip()


# --------------------------------------------------
# MESSAGE HISTORY
# --------------------------------------------------

def build_messages(
    message: str,
    history: list[dict] | None = None,
    language_instruction: str | None = None,
) -> list:
    """Build the message sequence for the agent."""

    messages = []

    # Apply the detected language instruction.
    if language_instruction:
        messages.append(
            SystemMessage(
                content=(
                    "LANGUAGE INSTRUCTION:\n"
                    f"{language_instruction}\n"
                    "Follow this language in every response, "
                    "including after using tools. "
                    "Translate retrieved information when necessary."
                )
            )
        )

    # Restore previous conversation messages.
    for item in history or []:

        role = item.get("role")
        content = item.get("content", "")

        if role == "user":
            messages.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":
            messages.append(
                AIMessage(content=content)
            )

    # Add the current user message.
    messages.append(
        HumanMessage(content=message)
    )

    return messages


# --------------------------------------------------
# AGENT EXECUTION
# --------------------------------------------------

def answer(
    message: str,
    history: list[dict] | None = None,
    language_instruction: str | None = None,
) -> str:
    """Process a user message using the LangGraph agent."""

    # Retrieve the cached agent and LLM.
    agent = get_agent()
    llm = get_llm()

    # Prepare conversation messages.
    messages = build_messages(
        message=message,
        history=history,
        language_instruction=language_instruction,
    )

    # Execute the agent.
    result = agent.invoke(
        {
            "messages": messages,
        }
    )

    # Extract the final response.
    content = result["messages"][-1].content

    if isinstance(content, list):
        final_response = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        ).strip()
    else:
        final_response = str(content).strip()

    # Translate the final response when required.
    if language_instruction and final_response:
        final_response = translate_response(
            llm=llm,
            response=final_response,
            language_instruction=language_instruction,
        )

    return final_response
