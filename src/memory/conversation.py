"""SQLite persistence for customer-support conversations."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

MessageRole = Literal["user", "assistant", "system"]


class ConversationStore:
    """Store conversations and their messages in a local SQLite database.

    A separate connection is opened for each operation. This avoids sharing
    SQLite connections across Streamlit sessions or threads.
    """

    def __init__(self, database_path: str | Path = "storage/conversations.db") -> None:
        self.database_path = Path(database_path).expanduser()
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations (conversation_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages (conversation_id, message_id);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def create_conversation(self) -> str:
        """Create a conversation and return its unique identifier."""
        conversation_id = str(uuid.uuid4())
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO conversations
                   (conversation_id, created_at, updated_at)
                   VALUES (?, ?, ?)""",
                (conversation_id, now, now),
            )
        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
    ) -> None:
        """Append a message to an existing conversation."""
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be 'user', 'assistant', or 'system'")
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, created_at)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, role, content, now),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("The message could not be saved")
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )

    def get_messages(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """Return messages in chronological order as role/content dictionaries.

        If ``limit`` is provided, the most recent messages are returned while
        preserving their original chronological order.
        """
        if limit is not None and limit < 1:
            return []

        with self._connect() as connection:
            if limit is None:
                rows = connection.execute(
                    """SELECT role, content FROM messages
                       WHERE conversation_id = ? ORDER BY message_id ASC""",
                    (conversation_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT role, content FROM (
                           SELECT message_id, role, content FROM messages
                           WHERE conversation_id = ?
                           ORDER BY message_id DESC LIMIT ?
                       ) ORDER BY message_id ASC""",
                    (conversation_id, limit),
                ).fetchall()

        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages; return whether it existed."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        return cursor.rowcount > 0
