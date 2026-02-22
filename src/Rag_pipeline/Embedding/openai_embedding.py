from __future__ import annotations

from typing import List

from openai import OpenAI

from Rag_pipeline.config import ragConfig
from Rag_pipeline.ingestion.Chunker import Chunk


class OpenAIEmbedder:
    """
    OpenAI embedding wrapper """

    def __init__(self, config: ragConfig) -> None:
        if not config.openai_key:
            raise ValueError("OpenAI API key is missing in configuration")

        self.client = OpenAI(api_key=config.openai_key)
        self.model = config.embed_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        return self.embed_texts([c.content for c in chunks])

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Query text cannot be empty")

        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )

        return response.data[0].embedding

