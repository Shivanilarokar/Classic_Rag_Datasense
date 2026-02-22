from __future__ import annotations

import os
import re
import sys
import uuid
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Add project root to path to support running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pypdf import PdfReader

from Rag_pipeline.config import ragConfig  # your config class
from .Chunker import chunk_document, Chunk  # Import chunking


# -------------------------
# Output models
# -------------------------
@dataclass(slots=True)
class PageSlice:
    page: int
    text: str


@dataclass(slots=True)
class LoadedPDF:
    file_name: str
    file_path: str
    checksum_sha256: str
    text: str
    pages: List[PageSlice]


@dataclass(slots=True)
class IngestionResult:
    status: str  # "INDEXED" | "DUPLICATE"
    namespace: str
    doc_id: str
    version_id: str
    doc_key: str
    checksum_sha256: str
    file_path: str
    pages_count: int
    text_chars: int
    chunks_count: int = 0  # Number of chunks created
    duplicate_of_version_id: Optional[str] = None


# -------------------------
# Ingestion Loader (PDF only)
# -------------------------
class PDFIngestionLoader:
   
    def __init__(
        self,
        config: ragConfig,
        sqlite_path: str = "rag_metadata.db",
        upload_dir: str = "uploads",
    ):
        self.config = config

        ok, errors = self.config.check_config()
        if not ok:
            raise ValueError("Invalid configuration: " + "; ".join(errors))

        self.sqlite_path = sqlite_path
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    # -------------------------
    # Public API: ingest
    # -------------------------
    def ingest_pdf(
        self,
        file_bytes: bytes,
        original_filename: str,
        *,
        user_id: str,
        org_id: str,
        scope: str,                 # "private" | "team" | "org"
        team_id: Optional[str] = None,
        doc_key: Optional[str] = None,  # if not provided, derived from filename
    ) -> IngestionResult:
      

        namespace = self._derive_namespace(org_id=org_id, user_id=user_id, scope=scope, team_id=team_id)

        # Step 1: Save file locally
        saved_path = self._save_file(file_bytes, original_filename, namespace)

        # Step 2: Load PDF (extract text/pages/checksum)
        loaded = self._load_pdf(saved_path)

        # Step 3: Determine doc_key (for versioning)
        final_doc_key = doc_key or self._derive_doc_key(original_filename)

        # Step 4: Duplicate detection (checksum + namespace)
        dup = self._find_duplicate_version(namespace=namespace, checksum=loaded.checksum_sha256)
        if dup is not None:
            # Duplicate found -> do NOT create new version/doc rows
            return IngestionResult(
                status="DUPLICATE",
                namespace=namespace,
                doc_id=dup["doc_id"],
                version_id=dup["version_id"],
                doc_key=dup["doc_key"],
                checksum_sha256=loaded.checksum_sha256,
                file_path=loaded.file_path,
                pages_count=len(loaded.pages),
                text_chars=len(loaded.text),
                chunks_count=0,
                duplicate_of_version_id=dup["version_id"],
            )

        # Step 5: Version strategy:
        #   - if doc exists for (namespace + doc_key) -> new version
        #   - else -> create new doc and first version
        doc_row = self._get_doc_by_key(namespace=namespace, doc_key=final_doc_key)

        if doc_row is None:
            doc_id = str(uuid.uuid4())
            self._insert_document(
                doc_id=doc_id,
                namespace=namespace,
                doc_key=final_doc_key,
                created_by=user_id,
                status="ACTIVE",
            )
        else:
            doc_id = doc_row["doc_id"]

        # mark older versions not latest
        self._mark_versions_not_latest(doc_id=doc_id)

        version_id = str(uuid.uuid4())
        self._insert_version(
            version_id=version_id,
            doc_id=doc_id,
            checksum=loaded.checksum_sha256,
            file_path=loaded.file_path,
            pages_count=len(loaded.pages),
            status="ACTIVE",
            is_latest=1,
        )

        # Create chunks from the document text
        chunks = chunk_document(
            text=loaded.text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        # Store chunks in database
        for chunk in chunks:
            self._insert_chunk(
                version_id=version_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )

        return IngestionResult(
            status="INDEXED",
            namespace=namespace,
            doc_id=doc_id,
            version_id=version_id,
            doc_key=final_doc_key,
            checksum_sha256=loaded.checksum_sha256,
            file_path=loaded.file_path,
            pages_count=len(loaded.pages),
            text_chars=len(loaded.text),
            chunks_count=len(chunks),
        )

    # -------------------------
    # PDF load/extract helpers
    # -------------------------
    def _load_pdf(self, file_path: str) -> LoadedPDF:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if p.suffix.lower() != ".pdf":
            raise ValueError("Only PDF is supported in this loader")

        checksum = self._sha256_file(str(p))

        reader = PdfReader(str(p))
        pages: List[PageSlice] = []
        full_parts: List[str] = []

        for idx, page in enumerate(reader.pages, start=1):
            t = page.extract_text() or ""
            t = self._normalize_text(t)
            pages.append(PageSlice(page=idx, text=t))
            if t:
                full_parts.append(t)

        full_text = "\n\n".join(full_parts).strip()

        return LoadedPDF(
            file_name=p.name,
            file_path=str(p),
            checksum_sha256=checksum,
            text=full_text,
            pages=pages,
        )

    def _save_file(self, file_bytes: bytes, original_filename: str, namespace: str) -> str:
        safe_name = self._sanitize_filename(original_filename)
        # Keep a namespace folder to avoid collisions across tenants
        # Replace colons with underscores for Windows path compatibility
        safe_namespace = namespace.replace(":", "_")
        ns_folder = self.upload_dir / self._sanitize_filename(safe_namespace)
        ns_folder.mkdir(parents=True, exist_ok=True)

        unique_prefix = str(uuid.uuid4())
        out_path = ns_folder / f"{unique_prefix}_{safe_name}"
        out_path.write_bytes(file_bytes)
        return str(out_path)

    @staticmethod
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = (s or "").replace("\x00", " ")
        s = " ".join(s.split())
        return s.strip()

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = name.strip().replace("\\", "_").replace("/", "_")
        name = re.sub(r"[^a-zA-Z0-9._:-]+", "_", name)
        return name[:180] if len(name) > 180 else name

    @staticmethod
    def _derive_doc_key(filename: str) -> str:
      
        base = Path(filename).stem.lower().strip()
        base = re.sub(r"[^a-z0-9]+", "-", base)
        base = re.sub(r"-{2,}", "-", base).strip("-")
        return base or "document"

    @staticmethod
    def _derive_namespace(org_id: str, user_id: str, scope: str, team_id: Optional[str]) -> str:
        scope = (scope or "").lower().strip()
        if scope == "private":
            return f"org:{org_id}:user:{user_id}"
        if scope == "team":
            if not team_id:
                raise ValueError("team_id is required when scope='team'")
            return f"org:{org_id}:team:{team_id}"
        if scope == "org":
            return f"org:{org_id}:public"
        raise ValueError("scope must be one of: private | team | org")

    # -------------------------
    # SQLite: schema + operations
    # -------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    doc_key TEXT NOT NULL,
                    status TEXT NOT NULL,          -- ACTIVE/ARCHIVED/DELETED
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(namespace, doc_key)
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    pages_count INTEGER NOT NULL,
                    status TEXT NOT NULL,          -- ACTIVE/ARCHIVED/DELETED
                    is_latest INTEGER NOT NULL,    -- 1 latest, 0 old
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
                """
            )

            # Chunks table for storing document chunks
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY(version_id) REFERENCES document_versions(version_id)
                );
                """
            )

            # Speed up duplicate checks
            conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_checksum_ns ON document_versions(checksum_sha256);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_ns_key ON documents(namespace, doc_key);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_doc_latest ON document_versions(doc_id, is_latest);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_version ON document_chunks(version_id);")
            conn.commit()

    def _insert_document(self, doc_id: str, namespace: str, doc_key: str, created_by: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, namespace, doc_key, status, created_by)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, namespace, doc_key, status, created_by),
            )

    def _insert_version(
        self,
        version_id: str,
        doc_id: str,
        checksum: str,
        file_path: str,
        pages_count: int,
        status: str,
        is_latest: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO document_versions
                  (version_id, doc_id, checksum_sha256, file_path, pages_count, status, is_latest)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (version_id, doc_id, checksum, file_path, pages_count, status, is_latest),
            )
            conn.execute(
                "UPDATE documents SET updated_at = datetime('now') WHERE doc_id = ?",
                (doc_id,),
            )

    def _insert_chunk(
        self,
        version_id: str,
        chunk_id: int,
        content: str,
        start_char: int,
        end_char: int,
    ) -> None:
        """Insert a chunk into the document_chunks table"""
        with self._connect() as conn:
            chunk_uuid = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO document_chunks
                  (chunk_id, version_id, chunk_index, content, start_char, end_char)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_uuid, version_id, chunk_id, content, start_char, end_char),
            )

    def _mark_versions_not_latest(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE document_versions SET is_latest = 0 WHERE doc_id = ?",
                (doc_id,),
            )

    def _get_doc_by_key(self, namespace: str, doc_key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id, namespace, doc_key, status FROM documents WHERE namespace = ? AND doc_key = ?",
                (namespace, doc_key),
            ).fetchone()
            return dict(row) if row else None

    def _find_duplicate_version(self, namespace: str, checksum: str) -> Optional[Dict[str, Any]]:
        """
        Duplicate detection rule you asked:
        - if checksum already exists for SAME namespace => duplicate
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT d.doc_id, d.doc_key, v.version_id
                FROM document_versions v
                JOIN documents d ON d.doc_id = v.doc_id
                WHERE d.namespace = ? AND v.checksum_sha256 = ? AND d.status != 'DELETED' AND v.status != 'DELETED'
                ORDER BY v.created_at DESC
                LIMIT 1
                """,
                (namespace, checksum),
            ).fetchone()

            return dict(row) if row else None

    def get_latest_namespace(self) -> Optional[str]:
        """
        Best-effort helper for the demo API:
        returns the most recently updated namespace from SQLite.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT namespace, MAX(updated_at) AS last_updated
                FROM documents
                WHERE status != 'DELETED'
                GROUP BY namespace
                ORDER BY last_updated DESC
                LIMIT 1
                """
            ).fetchone()
            return str(row["namespace"]) if row and row["namespace"] else None

    def get_version_source(self, version_id: str) -> Optional[Dict[str, str]]:
        """
        Lookup a human-friendly source for a version (doc_key + file name).
        This keeps Milvus schema minimal while still returning "source + page" to the user.
        """
        if not version_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT d.doc_key, v.file_path
                FROM document_versions v
                JOIN documents d ON d.doc_id = v.doc_id
                WHERE v.version_id = ?
                LIMIT 1
                """,
                (version_id,),
            ).fetchone()

            if not row:
                return None

            file_path = str(row["file_path"]) if row["file_path"] else ""
            file_name = Path(file_path).name if file_path else ""
            display_name = file_name
            # uploaded files are saved as: "<uuid>_<original_filename>.pdf"
            if file_name and re.match(r"^[0-9a-fA-F-]{36}_", file_name):
                display_name = file_name.split("_", 1)[1]
            return {
                "doc_key": str(row["doc_key"]) if row["doc_key"] else "",
                "file_path": file_path,
                "file_name": file_name,
                "display_name": display_name,
            }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    print("PDFIngestionLoader module loaded successfully")
