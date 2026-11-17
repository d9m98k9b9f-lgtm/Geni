from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Iterable


def data_directory() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return root / "GENI"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "geni"


class LocalStore:
    """Local SQLite storage for conversations, corpora, and reviewed examples."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else data_directory() / "geni.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS examples (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    approved INTEGER NOT NULL DEFAULT 0 CHECK(approved IN (0, 1)),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            if not db.execute("SELECT 1 FROM conversations LIMIT 1").fetchone():
                db.execute("INSERT INTO conversations(title) VALUES (?)", ("Ny konversation",))

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def conversations(self, search: str = "") -> list[sqlite3.Row]:
        with self._connect() as db:
            if search:
                return list(
                    db.execute(
                        """SELECT DISTINCT c.* FROM conversations c
                        LEFT JOIN messages m ON m.conversation_id = c.id
                        WHERE c.title LIKE ? OR m.content LIKE ? ORDER BY c.id DESC""",
                        (f"%{search}%", f"%{search}%"),
                    )
                )
            return list(db.execute("SELECT * FROM conversations ORDER BY id DESC"))

    def new_conversation(self, title: str = "Ny konversation") -> int:
        with self._connect() as db:
            cursor = db.execute("INSERT INTO conversations(title) VALUES (?)", (title,))
            return int(cursor.lastrowid)

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

    def messages(self, conversation_id: int) -> list[sqlite3.Row]:
        with self._connect() as db:
            return list(
                db.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
                    (conversation_id,),
                )
            )

    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Ogiltig meddelanderoll.")
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content),
            )

    def add_document(self, name: str, content: str) -> int:
        if not content.strip():
            raise ValueError("Textfilen är tom.")
        with self._connect() as db:
            cursor = db.execute("INSERT INTO documents(name, content) VALUES (?, ?)", (name, content))
            return int(cursor.lastrowid)

    def documents(self) -> list[sqlite3.Row]:
        with self._connect() as db:
            return list(db.execute("SELECT id, name, created_at FROM documents ORDER BY id DESC"))

    def add_example(self, prompt: str, answer: str) -> int:
        if not prompt.strip() or not answer.strip():
            raise ValueError("Både fråga och önskat svar måste fyllas i.")
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO examples(prompt, answer) VALUES (?, ?)", (prompt, answer)
            )
            return int(cursor.lastrowid)

    def examples(self) -> list[sqlite3.Row]:
        with self._connect() as db:
            return list(db.execute("SELECT * FROM examples ORDER BY id DESC"))

    def approve_example(self, example_id: int, approved: bool = True) -> None:
        with self._connect() as db:
            db.execute("UPDATE examples SET approved = ? WHERE id = ?", (int(approved), example_id))

    def training_texts(self) -> Iterable[str]:
        with self._connect() as db:
            docs = [row[0] for row in db.execute("SELECT content FROM documents")]
            examples = [
                f"Fråga: {row[0]}\nSvar: {row[1]}"
                for row in db.execute("SELECT prompt, answer FROM examples WHERE approved = 1")
            ]
        return [*docs, *examples]
