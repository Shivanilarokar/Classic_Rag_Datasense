from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from Rag_pipeline.config import ragConfig
from Rag_pipeline.pipeline import RagPipeline  # the pipeline file we created


app = FastAPI(title="Classic RAG (PDF + Milvus + OpenAI)")

# Create pipeline once at startup
cfg = ragConfig.load_from_env()
pipeline = RagPipeline(cfg, sqlite_path="rag_metadata.db", upload_dir="uploads")


# -----------------------
# Request/Response Models
# -----------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list


# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    return pipeline.health()


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    org_id: str = Form(...),
    scope: str = Form(...),  # private | team | org
    team_id: str | None = Form(None),
    doc_key: str | None = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    result = pipeline.ingest_pdf_upload(
        file_bytes=file_bytes,
        original_filename=file.filename,
        user_id=user_id,
        org_id=org_id,
        scope=scope,
        team_id=team_id,
        doc_key=doc_key,
    )

    # If ingestion failed for any reason
    if result.get("status") == "FAILED":
        raise HTTPException(status_code=500, detail=result.get("reason", "Ingestion failed"))

    return result


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Namespace selection:
    # - Prefer DEFAULT_NAMESPACE from env (if set)
    # - Else fall back to the most recently updated namespace in SQLite (best-effort local demo UX)
    namespace = (cfg.default_namespace or "").strip()
    if not namespace:
        namespace = (pipeline.ingestion_loader.get_latest_namespace() or "").strip()
    if not namespace:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF first via /documents/upload.",
        )

    ans = pipeline.ask(
        question=req.question,
        namespace=namespace,
    )

    # Return sources only once (not duplicated inside the answer).
    # Keep only sources that are actually referenced like [S1], [S2] in the answer text.
    import re

    refs_used = set(re.findall(r"\[(S\d+)\]", ans.answer or ""))
    citations = ans.citations or []

    if refs_used:
        citations = [c for c in citations if c.get("ref") in refs_used]

    # Deduplicate by (version_id, page) while preserving order.
    sources_map: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []

    for c in citations:
        doc_id = str(c.get("doc_id") or "")
        version_id = str(c.get("version_id") or "")
        ref = str(c.get("ref") or "")
        page = int(c.get("page") or 0)

        meta = pipeline.ingestion_loader.get_version_source(version_id) or {}
        source_name = (
            meta.get("display_name")
            or meta.get("doc_key")
            or meta.get("file_name")
            or meta.get("file_path")
            or doc_id
            or "unknown"
        )

        key = (doc_id, version_id)
        if key not in sources_map:
            sources_map[key] = {
                "source": source_name,
                "citations": [],  # [{"ref": "S2", "page": 21}, ...]
            }
            order.append(key)

        entry = sources_map[key]
        if ref and page:
            # Dedup by ref to keep the list clean/stable.
            if not any(x.get("ref") == ref for x in entry["citations"]):
                entry["citations"].append({"ref": ref, "page": page})

    sources = [sources_map[k] for k in order]

    return AskResponse(
        answer=ans.answer,
        sources=sources,
    )
