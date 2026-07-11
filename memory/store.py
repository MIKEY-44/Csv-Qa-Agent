"""
csv_qa_agent/memory/store.py
SQLite memory for conversation history and RAG with vector embeddings.
"""
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class SQLiteMemory:
    """SQLite-based memory for conversation history and context."""

    def __init__(self, db_path: str = "memory/agent_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                plan TEXT,
                code TEXT,
                result TEXT,
                status TEXT,
                confidence REAL,
                latency_ms REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS csv_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                csv_path TEXT UNIQUE NOT NULL,
                columns TEXT NOT NULL,
                dtypes TEXT NOT NULL,
                row_count INTEGER,
                sample_data TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, role, content, metadata)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, json.dumps(metadata) if metadata else None))
        conn.commit()
        conn.close()

    def get_conversation(self, session_id: str, limit: int = 50) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, timestamp, metadata
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                'role': row[0],
                'content': row[1],
                'timestamp': row[2],
                'metadata': json.loads(row[3]) if row[3] else None
            }
            for row in reversed(rows)
        ]

    def log_execution(self, session_id: str, question: str, plan: Optional[str] = None,
                      code: Optional[str] = None, result: Optional[str] = None,
                      status: str = "success", confidence: float = 0.0, latency_ms: float = 0.0):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO executions
            (session_id, question, plan, code, result, status, confidence, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, question, plan, code, result, status, confidence, latency_ms))
        conn.commit()
        conn.close()

    def get_similar_executions(self, question: str, limit: int = 5) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        keywords = question.lower().split()
        conditions = ' OR '.join(['LOWER(question) LIKE ?'] * len(keywords))
        query = f"""
            SELECT question, result, status, confidence, code
            FROM executions
            WHERE {conditions}
            AND status = 'success'
            ORDER BY confidence DESC
            LIMIT ?
        """
        params = [f'%{kw}%' for kw in keywords] + [limit]
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                'question': row[0],
                'result': row[1],
                'status': row[2],
                'confidence': row[3],
                'code': row[4]
            }
            for row in rows
        ]

    def cache_csv_metadata(self, csv_path: str, df_info: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO csv_metadata
            (csv_path, columns, dtypes, row_count, sample_data)
            VALUES (?, ?, ?, ?, ?)
        """, (
            csv_path,
            json.dumps(df_info.get('columns', [])),
            json.dumps(df_info.get('dtypes', {})),
            df_info.get('row_count', 0),
            json.dumps(df_info.get('sample', []))
        ))
        conn.commit()
        conn.close()

    def get_csv_metadata(self, csv_path: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT columns, dtypes, row_count, sample_data
            FROM csv_metadata
            WHERE csv_path = ?
        """, (csv_path,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                'columns': json.loads(row[0]),
                'dtypes': json.loads(row[1]),
                'row_count': row[2],
                'sample': json.loads(row[3])
            }
        return None


class SimpleRAG:
    """Simple RAG implementation using keyword matching."""

    def __init__(self, memory: SQLiteMemory):
        self.memory = memory

    def retrieve_relevant_context(self, question: str, csv_path: str, top_k: int = 3) -> Dict[str, Any]:
        context = {
            'similar_past_queries': [],
            'csv_metadata': None,
            'suggested_approach': None
        }
        similar = self.memory.get_similar_executions(question, limit=top_k)
        context['similar_past_queries'] = similar
        metadata = self.memory.get_csv_metadata(csv_path)
        context['csv_metadata'] = metadata
        if similar:
            best = similar[0]
            context['suggested_approach'] = {
                'based_on': best['question'],
                'confidence': best['confidence'],
                'code_pattern': best['code'][:200] if best['code'] else None
            }
        return context
