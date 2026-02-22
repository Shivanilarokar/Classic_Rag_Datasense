from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from pymilvus import MilvusClient, DataType

from Rag_pipeline.config import ragConfig
from Rag_pipeline.ingestion.Chunker import Chunk


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    doc_id: str
    version_id: str
    namespace: str
    chunk_pk: str  # primary key in milvus


class MilvusStore:
    """
    Milvus vector store :
      - ragConfig (milvus_url, milvus_api_key, milvus_db, milvus_collection, top_k_results)
      - Chunk (content, chunk_id, page, etc.) from chunker.py

    Stores metadata needed for bonus features:
      - namespace (user isolation)
      - lifecycle (ACTIVE/ARCHIVED/DELETED)
      - versioning (doc_id, version_id, is_latest)
      - duplicate checks via checksum (optional)
    """

    def __init__(self, config: ragConfig, vector_dim: int):
        self.config = config
        self.vector_dim = vector_dim
        self._client: Optional[MilvusClient] = None
        self._pk_field = "chunk_pk"
        self._vector_field = "embedding"

    # -------------------------
    # Connection / setup
    # -------------------------
    def connect(self) -> None:
        if self._client is not None:
            return

        if not self.config.milvus_url:
            raise ValueError("MILVUS_URI missing in config")
        if not self.config.milvus_api_key:
            raise ValueError("MILVUS_TOKEN missing in config")

        self._client = MilvusClient(uri=self.config.milvus_url, token=self.config.milvus_api_key)

        # NOTE: MilvusClient supports db_name parameter in many setups;
        # if yours requires explicit DB switching, you can recreate client with db_name.
        # Keeping minimal here for compatibility.

    def ensure_collection(self) -> None:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        name = self.config.milvus_collection
        if self._client.has_collection(name):
            # Collection exists already: discover schema so we can use the correct
            # primary key field name (common mismatch: `primary_key` vs `chunk_pk`).
            try:
                desc = self._client.describe_collection(collection_name=name)
                fields = desc.get("fields") if isinstance(desc, dict) else None
                if isinstance(fields, list):
                    # Primary key field
                    for f in fields:
                        if isinstance(f, dict) and f.get("is_primary") and isinstance(f.get("name"), str):
                            self._pk_field = f["name"]
                            break

                    # Vector field (float vector)
                    for f in fields:
                        if not isinstance(f, dict) or not isinstance(f.get("name"), str):
                            continue
                        f_type = f.get("type")
                        if f_type == DataType.FLOAT_VECTOR or str(f_type).upper().endswith("FLOAT_VECTOR"):
                            self._vector_field = f["name"]
                            break

                    # Validate required fields exist; if not, provide a clear error.
                    field_names = {f.get("name") for f in fields if isinstance(f, dict)}
                    required = {
                        self._pk_field,
                        self._vector_field,
                        "namespace",
                        "doc_id",
                        "version_id",
                        "checksum_sha256",
                        "lifecycle",
                        "is_latest",
                        "page",
                        "chunk_index",
                        "text",
                    }
                    missing = sorted(x for x in required if x not in field_names)
                    if missing:
                        raise RuntimeError(
                            "Existing Milvus collection schema doesn't match expected fields. "
                            f"Collection='{name}' missing fields: {missing}. "
                            "Fix: set a NEW MILVUS_COLLECTION name in .env (recommended) "
                            "or delete the existing collection in Zilliz Cloud and rerun."
                        )
            except RuntimeError:
                raise
            except Exception:
                # If describe fails, keep defaults; upsert/search will error with details.
                pass
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)

        # Primary key
        schema.add_field(self._pk_field, DataType.VARCHAR, max_length=128, is_primary=True)

        # Bonus metadata fields
        schema.add_field("namespace", DataType.VARCHAR, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("version_id", DataType.VARCHAR, max_length=64)
        schema.add_field("checksum_sha256", DataType.VARCHAR, max_length=64)
        schema.add_field("lifecycle", DataType.VARCHAR, max_length=16)   # ACTIVE/ARCHIVED/DELETED
        schema.add_field("is_latest", DataType.BOOL)

        # Chunk fields
        schema.add_field("page", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)

        # Vector
        schema.add_field(self._vector_field, DataType.FLOAT_VECTOR, dim=self.vector_dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=self._vector_field,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )

    # -------------------------
    # Write
    # -------------------------
    def upsert_chunks(
        self,
        *,
        namespace: str,
        doc_id: str,
        version_id: str,
        checksum_sha256: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        lifecycle: str = "ACTIVE",
        is_latest: bool = True,
    ) -> int:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")

        data = []
        for i, (c, emb) in enumerate(zip(chunks, embeddings)):
            # Primary key must be unique across whole collection
            chunk_pk = f"{doc_id}:{version_id}:{c.chunk_id}"

            data.append(
                {
                    self._pk_field: chunk_pk,
                    "namespace": namespace,
                    "doc_id": doc_id,
                    "version_id": version_id,
                    "checksum_sha256": checksum_sha256,
                    "lifecycle": lifecycle,
                    "is_latest": is_latest,
                    "page": int(c.page),
                    "chunk_index": int(i),
                    "text": c.content,
                    self._vector_field: emb,
                }
            )

        self._client.upsert(collection_name=self.config.milvus_collection, data=data)
        return len(data)

    # -------------------------
    # Read (search)
    # -------------------------
    def search(
        self,
        *,
        namespace: str,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        only_active: bool = True,
        only_latest: bool = True,
    ) -> List[RetrievedChunk]:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        k = top_k or self.config.top_k_results

        # Milvus filter expression
        expr_parts = [f'namespace == "{namespace}"']
        if only_active:
            expr_parts.append('lifecycle == "ACTIVE"')
        if only_latest:
            expr_parts.append("is_latest == true")
        expr = " && ".join(expr_parts)

        results = self._client.search(
            collection_name=self.config.milvus_collection,
            data=[query_embedding],
            limit=k,
            filter=expr,
            anns_field=self._vector_field,
            output_fields=[
                self._pk_field,
                "namespace",
                "doc_id",
                "version_id",
                "page",
                "chunk_index",
                "text",
                "lifecycle",
                "is_latest",
            ],
        )

        out: List[RetrievedChunk] = []
        for hit in results[0]:
            entity = hit["entity"]
            score = float(hit.get("distance", hit.get("score", 0.0)))

            chunk = Chunk(
                content=entity["text"],
                chunk_id=int(entity["chunk_index"]),  # local id in retrieval response
                start_char=0,
                end_char=len(entity["text"]),
                page=int(entity["page"]),
            )

            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    doc_id=entity["doc_id"],
                    version_id=entity["version_id"],
                    namespace=entity["namespace"],
                    chunk_pk=entity[self._pk_field],
                )
            )

        return out

    # -------------------------
    # Bonus helpers
    # -------------------------
    def mark_doc_not_latest(self, namespace: str, doc_id: str) -> None:
        """
        When a new version is uploaded, call this BEFORE inserting new vectors.
        Then insert new version with is_latest=True.
        """
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        expr = f'namespace == "{namespace}" && doc_id == "{doc_id}"'
        # Update expression syntax depends on Milvus version.
        # If your Milvus does not support update, you can instead:
        # - keep only_latest logic in SQLite, and query by version_id.
        self._client.update(
            collection_name=self.config.milvus_collection,
            filter=expr,
            data={"is_latest": False},
        )

    def delete_doc_version(self, namespace: str, doc_id: str, version_id: str) -> None:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        expr = f'namespace == "{namespace}" && doc_id == "{doc_id}" && version_id == "{version_id}"'
        self._client.delete(collection_name=self.config.milvus_collection, filter=expr)

    def health(self) -> dict[str, Any]:
        try:
            self.connect()
            exists = self._client.has_collection(self.config.milvus_collection)
            return {
                "connected": True,
                "collection": self.config.milvus_collection,
                "collection_exists": exists,
                "vector_dim": self.vector_dim,
            }
        except Exception as exc:
            return {"connected": False, "error": str(exc)}
