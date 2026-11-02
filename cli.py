"""Command-line interface for the Empório da Música support agent."""
from dotenv import load_dotenv

from src.agent.orchestrator import answer
from src.memory.conversation import ConversationStore


def main() -> None:
    load_dotenv()
    store = ConversationStore()
    conversation_id = store.create_conversation()
    print("Empório da Música Support — type 'exit' to quit.")
    while True:
        message = input("\nYou: ").strip()
        if message.lower() in {"exit", "quit"}:
            print("Assistant: Thanks for chatting. Have a musical day!")
            break
        if not message:
            continue
        history = store.get_messages(conversation_id)
        store.add_message(conversation_id, "user", message)
        try:
            response = answer(message, history)
        except Exception as error:
            response = (
                "I could not reach the local AI service. Check Ollama and the configured model. "
                f"Technical detail: {type(error).__name__}."
            )
        print(f"\nAssistant: {response}")
        store.add_message(conversation_id, "assistant", response)


if __name__ == "__main__":
    main()