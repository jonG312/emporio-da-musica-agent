
"""Unit tests for the SQLite conversation store."""

import pytest

from src.memory.conversation import ConversationStore


@pytest.fixture
def store(tmp_path):
    """Create an isolated SQLite database for testing."""

    database_path = tmp_path / "conversations.db"

    return ConversationStore(database_path)


def test_create_conversation_returns_unique_ids(store):
    """Each conversation should receive a unique identifier."""

    first_id = store.create_conversation()
    second_id = store.create_conversation()

    assert first_id
    assert second_id
    assert first_id != second_id


def test_add_and_get_messages_in_chronological_order(store):
    """Messages should be retrieved in chronological order."""

    conversation_id = store.create_conversation()

    store.add_message(
        conversation_id,
        "user",
        "Hello",
    )

    store.add_message(
        conversation_id,
        "assistant",
        "How can I help?",
    )

    messages = store.get_messages(conversation_id)

    assert len(messages) == 2

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"

    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "How can I help?"


def test_get_messages_limit_returns_most_recent_messages(store):
    """The history limit should return the latest messages."""

    conversation_id = store.create_conversation()

    store.add_message(conversation_id, "user", "First")
    store.add_message(conversation_id, "assistant", "Second")
    store.add_message(conversation_id, "user", "Third")

    messages = store.get_messages(
        conversation_id,
        limit=2,
    )

    assert len(messages) == 2
    assert messages[0]["content"] == "Second"
    assert messages[1]["content"] == "Third"


def test_invalid_role_is_rejected(store):
    """Only supported message roles should be accepted."""

    conversation_id = store.create_conversation()

    with pytest.raises(ValueError):
        store.add_message(
            conversation_id,
            "customer",
            "Hello",
        )


def test_non_string_message_is_rejected(store):
    """Message content must be a string."""

    conversation_id = store.create_conversation()

    with pytest.raises(TypeError):
        store.add_message(
            conversation_id,
            "user",
            123,
        )


def test_delete_conversation_removes_its_messages(store):
    """Deleting a conversation should remove its message history."""

    conversation_id = store.create_conversation()

    store.add_message(
        conversation_id,
        "user",
        "This message will be deleted",
    )

    result = store.delete_conversation(conversation_id)

    assert result is True
    assert store.get_messages(conversation_id) == []


def test_deleting_nonexistent_conversation_returns_false(store):
    """Deleting a nonexistent conversation should return False."""

    result = store.delete_conversation("nonexistent-id")

    assert result is False


def test_limit_below_one_returns_empty_list(store):
    """A non-positive history limit should return an empty list."""

    conversation_id = store.create_conversation()

    store.add_message(
        conversation_id,
        "user",
        "Hello",
    )

    messages = store.get_messages(
        conversation_id,
        limit=0,
    )

    assert messages == []
