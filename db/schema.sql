-- Documents table: one row per source file
CREATE TABLE IF NOT EXISTS documents (
  doc_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  relpath      TEXT UNIQUE NOT NULL,
  title        TEXT,
  space_key    TEXT,
  version      INTEGER,
  file_type    TEXT NOT NULL,
  updated_at   TEXT,
  sha256       TEXT NOT NULL,
  created_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_documents_relpath ON documents(relpath);
CREATE INDEX idx_documents_sha256 ON documents(sha256);
CREATE INDEX idx_documents_space_key ON documents(space_key);

-- Chunks table: one row per text chunk
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id       INTEGER NOT NULL,
  section_path TEXT,
  text         TEXT NOT NULL,
  token_count  INTEGER,
  extra_meta   TEXT,
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
  text,
  content='chunks',
  content_rowid='chunk_id',
  tokenize='porter unicode61'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO fts_chunks(rowid, text) VALUES (new.chunk_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES('delete', old.chunk_id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES('delete', old.chunk_id, old.text);
  INSERT INTO fts_chunks(rowid, text) VALUES (new.chunk_id, new.text);
END;

-- View for easy citation lookup
CREATE VIEW IF NOT EXISTS v_chunk_citation AS
SELECT 
  c.chunk_id,
  c.section_path,
  d.doc_id,
  d.title,
  d.relpath,
  d.space_key,
  d.version,
  d.file_type
FROM chunks c 
JOIN documents d ON c.doc_id = d.doc_id;