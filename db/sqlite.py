import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json
import re

class SQLiteDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self.conn
    
    def init_schema(self):
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            self.conn.executescript(f.read())
        self.conn.commit()
    
    def insert_document(self, relpath: str, title: Optional[str], space_key: Optional[str],
                       version: Optional[int], file_type: str, updated_at: Optional[str],
                       sha256: str) -> int:
        cursor = self.conn.execute("""
            INSERT OR IGNORE INTO documents (relpath, title, space_key, version, file_type, updated_at, sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (relpath, title, space_key, version, file_type, updated_at, sha256))
        
        if cursor.rowcount == 0:
            cursor = self.conn.execute("SELECT doc_id FROM documents WHERE relpath = ?", (relpath,))
            return cursor.fetchone()[0]
        
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_chunk(self, doc_id: int, section_path: Optional[str], text: str,
                    token_count: int, extra_meta: Optional[Dict] = None) -> int:
        extra_json = json.dumps(extra_meta) if extra_meta else None
        cursor = self.conn.execute("""
            INSERT INTO chunks (doc_id, section_path, text, token_count, extra_meta)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, section_path, text, token_count, extra_json))
        self.conn.commit()
        return cursor.lastrowid
    
    def document_exists(self, sha256: str) -> bool:
        cursor = self.conn.execute("SELECT 1 FROM documents WHERE sha256 = ?", (sha256,))
        return cursor.fetchone() is not None
    
    def get_document_by_path(self, relpath: str) -> Optional[Dict]:
        cursor = self.conn.execute("SELECT * FROM documents WHERE relpath = ?", (relpath,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def fts_search(self, query: str, limit: int = 50) -> List[Tuple[int, float]]:
        """Full-text search with proper query sanitization"""
        # Sanitize query for FTS5
        sanitized_query = self._sanitize_fts_query(query)
        
        if not sanitized_query:
            return []
        
        try:
            cursor = self.conn.execute("""
                SELECT rowid AS chunk_id, bm25(fts_chunks) AS score
                FROM fts_chunks
                WHERE fts_chunks MATCH ?
                ORDER BY score
                LIMIT ?
            """, (sanitized_query, limit))
            return [(row[0], row[1]) for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            # If FTS5 query still fails, fall back to basic search
            print(f"FTS5 query failed: {e}. Falling back to LIKE search.")
            return self._fallback_search(query, limit)
    
    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query string for FTS5"""
        # Remove special FTS5 characters that cause syntax errors
        # Keep alphanumeric, spaces, and basic punctuation
        query = re.sub(r'[^\w\s\-]', ' ', query)
        
        # Remove multiple spaces
        query = re.sub(r'\s+', ' ', query)
        
        # Split into words and filter
        words = [w.strip() for w in query.split() if len(w.strip()) > 1]
        
        if not words:
            return ""
        
        # Join words with OR for better recall
        return ' OR '.join(words)
    
    def _fallback_search(self, query: str, limit: int) -> List[Tuple[int, float]]:
        """Fallback LIKE-based search when FTS5 fails"""
        words = query.split()
        if not words:
            return []
        
        # Use LIKE for simple matching
        like_pattern = f"%{words[0]}%"
        cursor = self.conn.execute("""
            SELECT chunk_id, 1.0 AS score
            FROM chunks
            WHERE text LIKE ?
            LIMIT ?
        """, (like_pattern, limit))
        return [(row[0], row[1]) for row in cursor.fetchall()]
    
    def get_chunk(self, chunk_id: int) -> Optional[Dict]:
        cursor = self.conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_citation(self, chunk_id: int) -> Optional[Dict]:
        cursor = self.conn.execute("SELECT * FROM v_chunk_citation WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def close(self):
        if self.conn:
            self.conn.close()