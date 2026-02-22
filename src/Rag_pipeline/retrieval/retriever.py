from __future__ import annotations

from Rag_pipeline.Embedding import OpenAIEmbedder
from Rag_pipeline.config import ragConfig
from Rag_pipeline.vectordbstorage.milvustore import MilvusStore, RetrievedChunk


class Retriever:
    """
    Retrieval step of Classic RAG:
      1) Embed the question
      2) Search Milvus in the correct namespace
      3) Return RetrievedChunks 
    """

    def __init__(
        self,
        config: ragConfig,
        embedder: OpenAIEmbedder,
        store: MilvusStore,
        top_k: int | None = None,
    ):
        self.config = config
        self.embedder = embedder
        self.store = store
        self.top_k = top_k or self.config.top_k_results

    def retrieve(
        self,
        question: str,
        *,
        namespace: str,
        only_active: bool = True,
        only_latest: bool = True,
    ) -> list[RetrievedChunk]:
        if not question or not question.strip():
            return []

        query_vector = self.embedder.embed_query(question)

        return self.store.search(
            namespace=namespace,
            query_embedding=query_vector,
            top_k=self.top_k,
            only_active=only_active,
            only_latest=only_latest,
        )
