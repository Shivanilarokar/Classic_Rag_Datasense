from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from Rag_pipeline.config import ragConfig

if TYPE_CHECKING:
    from .Document_loader import LoadedPDF


@dataclass(slots=True)
class Chunk:
    content: str
    chunk_id: int
    start_char: int
    end_char: int
    page: int


def chunk_document(*, text: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
  

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    docs = splitter.create_documents([(text or "").strip()])
    chunks: List[Chunk] = []
    for idx, d in enumerate(docs):
        start = int(d.metadata.get("start_index", 0))
        content = d.page_content
        chunks.append(
            Chunk(
                content=content,
                chunk_id=idx,
                start_char=start,
                end_char=start + len(content),
                page=0,
            )
        )
    return chunks


class PDFChunker:

    def __init__(self, config: ragConfig):
        self.config = config

        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,     # character-based (simple, fine for starter)
            add_start_index=True,    # gives start positions within the input string
        )

    def chunk_loaded_pdf(self, doc: LoadedPDF) -> List[Chunk]:
       
        all_chunks: List[Chunk] = []
        chunk_id = 0

        for page_slice in doc.pages:
            page_num = page_slice.page
            page_text = (page_slice.text or "").strip()
            if not page_text:
                continue

            docs = self.splitter.create_documents([page_text])

            for d in docs:
                start = int(d.metadata.get("start_index", 0))
                content = d.page_content

                all_chunks.append(
                    Chunk(
                        content=content,
                        chunk_id=chunk_id,
                        start_char=start,
                        end_char=start + len(content),
                        page=page_num,
                    )
                )
                chunk_id += 1

        return all_chunks
