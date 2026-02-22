# rag_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from Rag_pipeline.config import ragConfig

# Ingestion
from Rag_pipeline.ingestion.Document_loader import PDFIngestionLoader
from Rag_pipeline.ingestion.Chunker import PDFChunker
from Rag_pipeline.Embedding import OpenAIEmbedder
from Rag_pipeline.vectordbstorage.milvustore import MilvusStore, RetrievedChunk

# Retrieval
from Rag_pipeline.retrieval import Retriever

# Generation
from Rag_pipeline.Generation.reranker import LLMReranker
from Rag_pipeline.Generation.generation import AnswerGenerator, Answer


class RagPipeline:
    """
    Compatible with the files we built:
      - config.py -> ragConfig
      - document_loader.py -> PDFIngestionLoader (handles namespace, dedup, version, lifecycle via sqlite)
      - chunker.py -> PDFChunker (LangChain RecursiveCharacterTextSplitter)
      - embeddings.py -> OpenAIEmbedder
      - milvus_store.py -> MilvusStore
      - retriever.py -> Retriever
      - generation/reranker.py -> LLMReranker
      - generation/generator.py -> AnswerGenerator
    """

    def __init__(
        self,
        config: ragConfig,
        sqlite_path: str = "rag_metadata.db",
        upload_dir: str = "uploads",
    ) -> None:
        self.config = config

        # Components
        self.ingestion_loader = PDFIngestionLoader(
            config=self.config,
            sqlite_path=sqlite_path,
            upload_dir=upload_dir,
        )
        self.chunker = PDFChunker(self.config)
        self.embedder = OpenAIEmbedder(self.config)

        # Vector dim is required for Milvus schema. Keep it configurable to avoid
        # calling the embedding API during service startup.
        dim = int(self.config.embedding_dim)

        self.store = MilvusStore(self.config, vector_dim=dim)
        self.retriever = Retriever(self.config, self.embedder, self.store)

        self.reranker = LLMReranker(self.config)
        self.generator = AnswerGenerator(self.config)

        # Connect once (you can also connect lazily in methods)
        self.store.connect()
        self.store.ensure_collection()

    # -------------------------
    # Ingestion (Upload pipeline)
    # -------------------------
    def ingest_pdf_upload(
        self,
        *,
        file_bytes: bytes,
        original_filename: str,
        user_id: str,
        org_id: str,
        scope: str,                  # "private" | "team" | "org"
        team_id: Optional[str] = None,
        doc_key: Optional[str] = None,
    ) -> dict:
        """
        Implements your upload ingestion flow end-to-end:

        1) Save file
        2) Extract text + checksum
        3) SQLite dedup + version
        4) Chunk + embed
        5) Store in Milvus with namespace/doc_id/version_id metadata

        Returns a dict (easy to return via API).
        """

        # 1-5 handled by ingestion_loader (dedup/version/lifecycle saved in sqlite)
        ingest_result = self.ingestion_loader.ingest_pdf(
            file_bytes=file_bytes,
            original_filename=original_filename,
            user_id=user_id,
            org_id=org_id,
            scope=scope,
            team_id=team_id,
            doc_key=doc_key,
        )

        # If duplicate, skip vector ingestion (already indexed)
        if ingest_result.status == "DUPLICATE":
            return {
                "status": "DUPLICATE",
                "namespace": ingest_result.namespace,
                "doc_id": ingest_result.doc_id,
                "version_id": ingest_result.version_id,
                "doc_key": ingest_result.doc_key,
                "checksum": ingest_result.checksum_sha256,
                "file_path": ingest_result.file_path,
            }

        # Load extracted content (from saved file path)
        loaded_pdf = self.ingestion_loader._load_pdf(ingest_result.file_path)  # internal helper

        # Chunk page-wise (keeps citations correct)
        chunks = self.chunker.chunk_loaded_pdf(loaded_pdf)
        if not chunks:
            return {
                "status": "FAILED",
                "reason": "No chunks created from PDF",
                "namespace": ingest_result.namespace,
                "doc_id": ingest_result.doc_id,
                "version_id": ingest_result.version_id,
            }

        # Embed
        embeddings = self.embedder.embed_chunks(chunks)
        if not embeddings:
            return {
                "status": "FAILED",
                "reason": "No embeddings created",
                "namespace": ingest_result.namespace,
                "doc_id": ingest_result.doc_id,
                "version_id": ingest_result.version_id,
            }

        # Store in Milvus
        inserted = self.store.upsert_chunks(
            namespace=ingest_result.namespace,
            doc_id=ingest_result.doc_id,
            version_id=ingest_result.version_id,
            checksum_sha256=ingest_result.checksum_sha256,
            chunks=chunks,
            embeddings=embeddings,
            lifecycle="ACTIVE",
            is_latest=True,
        )

        return {
            "status": "INDEXED",
            "namespace": ingest_result.namespace,
            "doc_id": ingest_result.doc_id,
            "version_id": ingest_result.version_id,
            "doc_key": ingest_result.doc_key,
            "checksum": ingest_result.checksum_sha256,
            "file_path": ingest_result.file_path,
            "chunks_indexed": inserted,
        }

    # -------------------------
    # Ask (RAG: retrieve -> rerank -> generate)
    # -------------------------
    def ask(
        self,
        *,
        question: str,
        namespace: str,
    ) -> Answer:
        retrieved = self.retriever.retrieve(
            question,
            namespace=namespace,
            only_active=True,
            only_latest=True,
        )

        # Optionally apply rerank
        reranked = self.reranker.rerank(
            question,
            retrieved,
            top_n=self.config.rerank_top_n,
            include_reasons=False,
        )

        return self.generator.generate(question, reranked)

    # -------------------------
    # Health
    # -------------------------
    def health(self) -> dict:
        ok, errors = self.config.check_config()
        milvus = self.store.health()
        return {
            "config_ok": ok,
            "config_errors": errors,
            "milvus": milvus,
            "embed_model": self.config.embed_model,
            "chat_model": self.config.chat_model,
            "collection": self.config.milvus_collection,
            "top_k": self.config.top_k_results,
            "rerank_top_n": self.config.rerank_top_n,
            "default_namespace": self.config.default_namespace,
        }


if __name__ == "__main__":
    cfg = ragConfig.load_from_env()
    pipeline = RagPipeline(cfg)

    # Quick local test: ingest PDFs from data/documents (if present).
    # This avoids requiring a special `sample.pdf` in repo root.
    docs_dir = Path("data/documents")
    pdfs = sorted(docs_dir.glob("*.pdf")) if docs_dir.exists() else []

    if not pdfs:
        print("No PDFs found. Put PDFs under data/documents/ (or upload via the API).")
        raise SystemExit(1)

    last_ns: str | None = None
    for pdf_path in pdfs:
        res = pipeline.ingest_pdf_upload(
            file_bytes=pdf_path.read_bytes(),
            original_filename=pdf_path.name,
            user_id="u123",
            org_id="o999",
            scope="org",
            doc_key=pdf_path.stem,
        )
        print(res)
        last_ns = res.get("namespace") or last_ns

    if last_ns:
        ans = pipeline.ask(
            question="What is our leave policy?",
            namespace=last_ns,
        )
        print(ans.answer)
