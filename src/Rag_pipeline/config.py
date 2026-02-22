import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load values from .env file
load_dotenv()


@dataclass
class ragConfig:
    # OpenAI Congigurations
    openai_key: str
    embed_model: str = "text-embedding-ada-002"
    embedding_dim: int = 1536
    chat_model: str = "gpt-4o-mini"

    # Milvus Congigurations
    milvus_url: str = ""
    milvus_api_key: str = ""
    milvus_db: str = "default"
    milvus_collection: str = "rag_chunks"

    # RAG chunk Congigurations
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k_results: int = 5
    rerank_top_n: int = 5
    default_namespace: str = ""

    @classmethod
    def load_from_env(cls):
        """
        Create config object using environment variables
        """

        return cls(
            openai_key=os.getenv("OPENAI_API_KEY", ""),
            embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002"),
            embedding_dim=int(os.getenv("EMBED_DIM", "1536")),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),

            milvus_url=os.getenv("MILVUS_URI", ""),
            milvus_api_key=os.getenv("MILVUS_TOKEN", os.getenv("MILVUS_API_KEY", "")),
            milvus_db=os.getenv("MILVUS_DATABASE", "default"),
            milvus_collection=os.getenv("MILVUS_COLLECTION", "rag_chunks"),

            chunk_size=int(os.getenv("CHUNK_SIZE", os.getenv("CHUNK_SIZE_TOKENS", "500"))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", os.getenv("CHUNK_OVERLAP_TOKENS", "80"))),
            top_k_results=int(os.getenv("TOP_K", "5")),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "5")),
            default_namespace=os.getenv("DEFAULT_NAMESPACE", ""),
        )

    def check_config(self):
        """
        Check if important values are missing
        """

        errors = []

        if not self.openai_key:
            errors.append("OpenAI API key is missing")

        if not self.milvus_url:
            errors.append("Milvus URL is missing")

        if not self.milvus_api_key:
            errors.append("Milvus API key is missing")

        if self.embedding_dim <= 0:
            errors.append("Embedding dimension (EMBED_DIM) must be a positive integer")

        if self.chunk_overlap >= self.chunk_size:
            errors.append("Chunk overlap must be smaller than chunk size")

        if self.rerank_top_n <= 0:
            errors.append("Rerank top_n (RERANK_TOP_N) must be a positive integer")

        if len(errors) > 0:
            return False, errors

        return True, []
