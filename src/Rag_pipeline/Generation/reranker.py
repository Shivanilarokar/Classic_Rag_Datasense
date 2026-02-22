# generation/reranker.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from Rag_pipeline.config import ragConfig
from Rag_pipeline.vectordbstorage.milvustore import RetrievedChunk


@dataclass(slots=True)
class RerankedChunk:
    item: RetrievedChunk
    rank: int
    reason: str


class LLMReranker:
    """
    Simple, correct Classic-RAG reranker.

    Input: question + RetrievedChunk list (from Milvus)
    Output: reordered list (top_n) with optional reasons.

    Why LLM rerank?
      - Vector search gives "semantic similar"
      - LLM rerank gives "directly answers the question"
    """

    def __init__(self, config: ragConfig):
        if not config.openai_key:
            raise ValueError("OpenAI API key is missing in configuration")

        self.client = OpenAI(api_key=config.openai_key)
        self.model = config.chat_model

    def rerank(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        top_n: int = 5,
        include_reasons: bool = False,
    ) -> List[RetrievedChunk] | List[RerankedChunk]:
        if not retrieved_chunks:
            return [] if not include_reasons else []

        # Keep prompt small & stable
        passages = []
        for i, c in enumerate(retrieved_chunks):
            # Trim to reduce tokens while keeping meaning
            txt = (c.chunk.content or "").strip().replace("\n", " ")
            txt = txt[:900]
            passages.append(f"[{i}] (page={c.chunk.page}) {txt}")

        n = len(retrieved_chunks)
        k = max(1, min(int(top_n), n))

        user_prompt = (
            "You are a relevance judge for a Retrieval-Augmented Generation (RAG) system.\n"
            "Task: rank the retrieved chunks by how useful they are for answering the question.\n\n"
            "Ranking rules (most important first):\n"
            "1) Direct answer: Passage explicitly answers the question or contains key facts needed.\n"
            "2) Specificity: Prefer concrete steps, requirements, definitions, numbers, dates over generic text.\n"
            "3) Coverage: If the question has multiple parts, prefer passages that cover more parts.\n"
            "4) Terminology match: Prefer passages that mention the exact entities/terms from the question.\n"
            "5) Reduce redundancy: If two passages are similar, rank the clearer/more complete one higher.\n\n"
            "Do not invent facts. Use ONLY the passage text.\n"
            "Ignore writing style; judge usefulness for answering.\n\n"
            "Output requirements:\n"
            f"- Return ONLY a single valid JSON object.\n"
            f"- order must be a list of unique integers covering ALL passages 0..{n-1}, sorted most relevant first.\n"
            f"- reasons must be a JSON object mapping indices (as strings) to 1 short sentence.\n"
            f"- Provide reasons ONLY for the top {k} indices.\n"
            "- No markdown, no code fences, no extra keys.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "order": [2, 0, 1],\n'
            '  "reasons": {"2": "Most directly answers the question", "0": "Contains required steps"}\n'
            "}\n\n"
            f"Question: {question}\n\n"
            "Passages:\n"
            + "\n".join(passages)
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return only JSON. No extra text."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        raw = (resp.choices[0].message.content or "").strip()
        parsed = self._safe_parse_json(raw)

        order = parsed.get("order")
        reasons = parsed.get("reasons") or {}

        # Fallback if model output is bad
        if not isinstance(order, list) or not order:
            order = list(range(len(retrieved_chunks)))

        # Build ordered list safely
        seen = set()
        ordered: List[RetrievedChunk] = []
        for idx in order:
            if isinstance(idx, int) and 0 <= idx < len(retrieved_chunks) and idx not in seen:
                ordered.append(retrieved_chunks[idx])
                seen.add(idx)

        # Append any missing chunks (keeps stable behavior)
        for i, c in enumerate(retrieved_chunks):
            if i not in seen:
                ordered.append(c)

        ordered = ordered[:top_n]

        if not include_reasons:
            return ordered

        out: List[RerankedChunk] = []
        for rank, item in enumerate(ordered, start=1):
            # Find original index for reasons mapping
            original_index = self._find_original_index(item, retrieved_chunks)
            reason = ""
            if original_index is not None:
                reason = reasons.get(str(original_index), "") if isinstance(reasons, dict) else ""
            out.append(RerankedChunk(item=item, rank=rank, reason=reason))
        return out

    @staticmethod
    def _safe_parse_json(s: str) -> dict:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _find_original_index(item: RetrievedChunk, retrieved_chunks: List[RetrievedChunk]) -> Optional[int]:
        """
        Find original index based on chunk_pk (stable primary key).
        """
        for i, c in enumerate(retrieved_chunks):
            if c.chunk_pk == item.chunk_pk:
                return i
        return None
