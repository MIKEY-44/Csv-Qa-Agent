"""Database Service - SQLite for chat history."""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class DatabaseService:
    """SQLite database for persisting chat history."""

    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    filename TEXT,
                    schema TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    question TEXT,
                    answer TEXT,
                    code TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    mode TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.commit()

    def create_session(self, session_id: str, filename: str, schema: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, filename, schema) VALUES (?, ?, ?)",
                (session_id, filename, json.dumps(schema))
            )
            conn.commit()

    def add_message(self, session_id: str, question: str, answer: str, code: str, 
                    success: bool, execution_time: float, mode: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO messages (session_id, question, answer, code, success, execution_time, mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, question, answer, code, success, execution_time, mode)
            )
            conn.commit()

    def get_history(self, session_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_session_stats(self, session_id: str) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT COUNT(*) as total, AVG(success) as success_rate, AVG(execution_time) as avg_time
                   FROM messages WHERE session_id = ?""",
                (session_id,)
            )
            row = cursor.fetchone()
            return {
                "total_questions": row[0],
                "success_rate": round(row[1] * 100, 1) if row[1] else 0,
                "avg_execution_time": round(row[2], 3) if row[2] else 0
            }
