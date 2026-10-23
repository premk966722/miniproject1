"""SQLite persistence for ingested source documents, chunks, and optional vectors."""
import json
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH


def _json_object(value):
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


@contextmanager
def connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        yield db
        db.commit()
    finally:
        db.close()


def initialize():
    with connection() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
          source_url TEXT, source_type TEXT NOT NULL, authority TEXT,
          official_domain_verified INTEGER NOT NULL DEFAULT 0, metadata_json TEXT NOT NULL,
          sha256 TEXT NOT NULL, page_count INTEGER NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          text TEXT NOT NULL, page_start INTEGER, page_end INTEGER, section TEXT,
          embedding_json TEXT, metadata_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        """)
        existing = {row["name"] for row in db.execute("PRAGMA table_info(documents)")}
        migrations = {
            "stored_filename": "TEXT",
            "file_size": "INTEGER NOT NULL DEFAULT 0",
            "upload_timestamp": "TEXT NOT NULL DEFAULT ''",
            "chunk_count": "INTEGER NOT NULL DEFAULT 0",
            "ingestion_status": "TEXT NOT NULL DEFAULT 'completed'",
            "document_type": "TEXT",
            "tax_period": "TEXT",
        }
        for column, declaration in migrations.items():
            if column not in existing:
                db.execute(f"ALTER TABLE documents ADD COLUMN {column} {declaration}")
        db.execute("UPDATE documents SET upload_timestamp=created_at WHERE upload_timestamp=''")


def insert_document(document: dict, chunks: list[dict]):
    with connection() as db:
        db.execute("""INSERT INTO documents
            (id,title,filename,source_url,source_type,authority,official_domain_verified,metadata_json,
             sha256,page_count,created_at,stored_filename,file_size,upload_timestamp,chunk_count,
             ingestion_status,document_type,tax_period)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            document["id"], document["title"], document["filename"], document.get("source_url"),
            document["source_type"], document.get("authority"), int(document.get("official_domain_verified", False)),
            json.dumps(document["metadata"], ensure_ascii=False), document["sha256"], document["page_count"],
            document["created_at"], document.get("stored_filename"), document.get("file_size", 0),
            document.get("upload_timestamp", document["created_at"]), len(chunks),
            document.get("ingestion_status", "completed"), document.get("document_type"), document.get("tax_period"),
        ))
        db.executemany("""INSERT INTO chunks
            (id,document_id,text,page_start,page_end,section,embedding_json,metadata_json)
            VALUES(?,?,?,?,?,?,?,?)""", [(
            c["chunk_id"] if "chunk_id" in c else c["id"], document["id"], c["text"],
            c.get("page_number", c.get("page_start")), c.get("page_number", c.get("page_end")),
            c.get("section"), json.dumps(c["embedding"]) if c.get("embedding") else None,
            json.dumps({k: v for k, v in c.items() if k not in ("id", "chunk_id", "text", "page_start", "page_end", "section", "embedding")}, ensure_ascii=False),
        ) for c in chunks])


def list_documents():
    with connection() as db:
        rows = db.execute("SELECT * FROM documents WHERE ingestion_status='completed' ORDER BY upload_timestamp DESC").fetchall()
    return [{**dict(r), "metadata": json.loads(r["metadata_json"]), "official_domain_verified": bool(r["official_domain_verified"])} for r in rows]


def get_document(document_id: str):
    with connection() as db:
        row = db.execute("SELECT * FROM documents WHERE id=? AND ingestion_status='completed'", (document_id,)).fetchone()
    if row is None:
        return None
    return {**dict(row), "metadata": json.loads(row["metadata_json"]),
            "official_domain_verified": bool(row["official_domain_verified"])}


def list_chunks():
    with connection() as db:
        rows = db.execute("""SELECT c.*, d.title, d.filename, d.source_url, d.source_type, d.authority,
          d.document_type, d.tax_period, d.official_domain_verified, d.metadata_json AS document_metadata
          FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.ingestion_status='completed'""").fetchall()
    output = []
    for row in rows:
        item = dict(row)
        item["embedding"] = json.loads(item.pop("embedding_json")) if item["embedding_json"] else None
        item["metadata"] = _json_object(item.pop("metadata_json"))
        item["document_metadata"] = _json_object(item["document_metadata"])
        item["official_domain_verified"] = bool(item["official_domain_verified"])
        output.append(item)
    return output


def chunks_version():
    """Small change token for refreshing in-process retrieval indexes after writes."""
    with connection() as db:
        row = db.execute("""SELECT COUNT(*) AS chunk_count, COALESCE(MAX(c.rowid), 0) AS last_rowid,
            COALESCE(MAX(d.upload_timestamp), '') AS latest_upload
            FROM chunks c JOIN documents d ON d.id=c.document_id
            WHERE d.ingestion_status='completed'""").fetchone()
    return (str(DB_PATH.resolve()), row["chunk_count"], row["last_rowid"], row["latest_upload"])


def keyword_chunks_version():
    """Backward-compatible name for the chunk corpus change token."""
    return chunks_version()
