
"""Streamlit chat interface for the Empório da Música support agent."""

import os
import re

from dotenv import load_dotenv
from langdetect import detect_langs, LangDetectException


# --------------------------------------------------
# LANGUAGE DETECTION
# --------------------------------------------------

LANGUAGE_MAP = {
    "pt": "Português (Brasil)",
    "es": "Español",
    "en": "English",
}


def detect_explicit_language(text):
    """Detect an explicit request to change the conversation language."""

    text = text.lower()

    language_patterns = {
        "Português (Brasil)": [
            r"\bem português\b",
            r"\bportuguês\b",
            r"\bportugues\b",
            r"\bin portuguese\b",
            r"\bspeak portuguese\b",
        ],
        "Español": [
            r"\ben español\b",
            r"\bespañol\b",
            r"\bespanol\b",
            r"\bin spanish\b",
            r"\bspeak spanish\b",
        ],
        "English": [
            r"\bin english\b",
            r"\benglish please\b",
            r"\bspeak english\b",
            r"\bcontinue in english\b",
            r"\bem inglês\b",
            r"\ben inglés\b",
        ],
    }

    for language, patterns in language_patterns.items():
        if any(re.search(pattern, text) for pattern in patterns):
            return language

    return None


def detect_language(text, current_language=None):
    """Detect language while preserving the current conversation language."""

    # Explicit language requests have priority.
    explicit_language = detect_explicit_language(text)

    if explicit_language:
        return explicit_language

    text = text.strip()

    if not text:
        return current_language or "English"

    words = text.split()

    # Avoid changing languages because of short messages.
    if current_language and (
        len(text) < 45 or len(words) < 7
    ):
        return current_language

    try:
        predictions = detect_langs(text)

        if not predictions:
            return current_language or "English"

        best_prediction = predictions[0]

        detected_code = best_prediction.lang
        confidence = best_prediction.prob

        detected_language = LANGUAGE_MAP.get(detected_code)

        if not detected_language:
            return current_language or "English"

        # Initial language detection.
        if current_language is None:
            if confidence >= 0.65:
                return detected_language

            return "English"

        # Require strong evidence before changing languages.
        if (
            detected_language != current_language
            and confidence >= 0.90
        ):
            return detected_language

        return current_language

    except LangDetectException:
        return current_language or "English"


# --------------------------------------------------
# ENVIRONMENT CONFIGURATION
# --------------------------------------------------

load_dotenv()

# Bypass HTTP proxies for local Ollama connections.
for variable in ("NO_PROXY", "no_proxy"):
    current = os.environ.get(variable, "")
    hosts = [host.strip() for host in current.split(",") if host.strip()]

    for host in ("localhost", "127.0.0.1"):
        if host not in hosts:
            hosts.append(host)

    os.environ[variable] = ",".join(hosts)


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import streamlit as st

from src.agent.orchestrator import answer
from src.memory.conversation import ConversationStore


# --------------------------------------------------
# STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Empório da Música Support",
    page_icon="🎸",
    layout="centered"
)

st.title("🎸 Empório da Música")
st.caption("Your musical-instrument store assistant")


# --------------------------------------------------
# CONVERSATION STORAGE
# --------------------------------------------------

@st.cache_resource
def get_store():
    return ConversationStore(
        os.getenv("SQLITE_PATH", "storage/conversations.db")
    )


store = get_store()


if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = store.create_conversation()


if "messages" not in st.session_state:
    st.session_state.messages = store.get_messages(
        st.session_state.conversation_id
    )


# Restore the conversation language if it is not in session state.
if "conversation_language" not in st.session_state:

    st.session_state.conversation_language = None

    for previous_message in st.session_state.messages:

        if previous_message.get("role") == "user":

            detected_language = detect_language(
                previous_message.get("content", "")
            )

            st.session_state.conversation_language = detected_language
            break


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.subheader("Conversation")

    st.caption("Your chat is stored locally in SQLite.")

    if st.button("Start a new conversation", use_container_width=True):

        st.session_state.conversation_id = store.create_conversation()
        st.session_state.messages = []
        st.session_state.conversation_language = None

        st.rerun()

    st.divider()

    st.caption(
        "Prototype data comes from the supplied CSV snapshot and policy PDF."
    )


# --------------------------------------------------
# DISPLAY CONVERSATION HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT AND RESPONSE
# --------------------------------------------------

prompt = st.chat_input(
    "Ask about instruments, orders, store policies, or opening hours..."
)


if prompt:

    # Retrieve recent conversation history.
    history = st.session_state.messages[-12:]

    # Save user message.
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    store.add_message(
        st.session_state.conversation_id,
        "user",
        prompt
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response.
    with st.chat_message("assistant"):

        with st.spinner("Checking the store information..."):

            try:

                # Detect language while preserving conversation context.
                st.session_state.conversation_language = detect_language(
                    prompt,
                    current_language=st.session_state.conversation_language
                )

                user_language = st.session_state.conversation_language

                # Build language instruction.
                language_instruction = f"""
                The target language for this conversation is {user_language}.

                This language is authoritative.

                You MUST respond exclusively in this language.

                Maintain this language throughout the response.

                Translate all relevant CSV and PDF content into this language.

                Do not change languages because of short messages.

                Only change the target language when the application explicitly
                updates the conversation language.
                """

                # Call the main agent.
                response = answer(
                    prompt,
                    history,
                    language_instruction=language_instruction
                )

            except Exception as error:

                response = (
                    "I'm sorry, but I couldn't reach the local AI service. "
                    "Please check that Ollama is running and the configured model is installed. "
                    f"Technical detail: `{type(error).__name__}: {error}`."
                )

        st.markdown(response)

    # Save assistant response.
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    store.add_message(
        st.session_state.conversation_id,
        "assistant",
        response
    )
