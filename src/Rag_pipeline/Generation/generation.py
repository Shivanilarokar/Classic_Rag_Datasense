# generation/generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from openai import OpenAI

import re

from Rag_pipeline.config import ragConfig
from Rag_pipeline.vectordbstorage.milvustore import RetrievedChunk


SYSTEM_PROMPT = """You are an internal knowledge assistant using Retrieval-Augmented Generation (RAG).

Hard rules:
- Use ONLY the provided Context passages. Do not use outside knowledge.
- If the Context does not contain the answer, reply exactly:
    I don't know based on the provided documents.
- Do not invent policy details, steps, numbers, dates, names, or definitions.

Citations:
- Use chunk identifiers when citing. Cite chunks like [C:docid:chunkpk].
- If a sentence is supported by multiple chunks, cite all relevant chunk ids like [C:...][C:...].
- Cite only chunks you actually used.
- Do NOT add a "Sources:" free-form section inside the answer. The API will return a separate `sources` structure.

Answer style:
- Be concise, direct, and specific. Prefer short paragraphs or bullets.
- When asked "how/steps", provide the steps/requirements as stated in the context.
- If the context contains conflicting information, state that there is a conflict and cite both sides (by page or chunk id).
"""


@dataclass(slots=True)
class Answer:
    question: str
    answer: str
    sources: List[RetrievedChunk]
    citations: List[Dict[str, Any]]


class AnswerGenerator:
    """
      - ragConfig (openai_key, chat_model)
      - RetrievedChunk (from milvus_store.py)
    """

    def __init__(self, config: ragConfig) -> None:
        if not config.openai_key:
            raise ValueError("OpenAI API key is missing in configuration")

        self.client = OpenAI(api_key=config.openai_key)
        self.model = config.chat_model

    def generate(self, question: str, retrieved_chunks: List[RetrievedChunk]) -> Answer:
        context, citations = self._build_context(retrieved_chunks)

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Write the best answer grounded in the context.\n"
            "When you cite, use the chunk citation format shown in the context (e.g. [C:docid:chunkpk]).\n"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        text = (resp.choices[0].message.content or "").strip() or "No answer generated."
        text = self._strip_sources_section(text)

        return Answer(
            question=question,
            answer=text,
            sources=retrieved_chunks,
            citations=citations,
        )

    @staticmethod
    def _strip_sources_section(text: str) -> str:
        """
        The model sometimes appends a Sources section even when asked not to.
        Strip anything from a trailing 'Sources:' header onwards.
        """
        if not text:
            return text
        m = re.search(r"\n\s*Sources\s*:\s*\n", text, flags=re.IGNORECASE)
        if not m:
            return text
        return text[: m.start()].rstrip()

    @staticmethod
    def _build_context(retrieved_chunks: List[RetrievedChunk]) -> tuple[str, List[Dict[str, Any]]]:
        if not retrieved_chunks:
            return "No context found.", []

        blocks: List[str] = []
        citations: List[Dict[str, Any]] = []

        for idx, item in enumerate(retrieved_chunks, start=1):
            # item.chunk is our Chunk dataclass (content/page/etc.)
            page = item.chunk.page
            text = (item.chunk.content or "").strip()

            # Keep context bounded
            if len(text) > 1200:
                text = text[:1200] + " ..."

            # Label context blocks with chunk-based citations only.
            chunk_label = f"C:{item.doc_id}:{item.chunk_pk}"

            blocks.append(
                f"[{chunk_label}] doc_id={item.doc_id} version_id={item.version_id} page={page} score={item.score:.4f}\n"
                f"{text}"
            )

            citations.append(
                {
                    "ref": chunk_label,
                    "doc_id": item.doc_id,
                    "version_id": item.version_id,
                    "page": page,
                    "chunk_pk": item.chunk_pk,
                    "score": item.score,
                    "namespace": item.namespace,
                }
            )

        return "\n\n".join(blocks), citations
