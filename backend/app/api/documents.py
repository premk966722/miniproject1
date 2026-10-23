"""PDF document upload and listing endpoints."""
import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import EMBEDDING_MODEL, MAX_PDF_BYTES, UPLOAD_DIR
from ..embeddings import embed_texts
from ..schemas import DocumentUploadResponse
from ..services.ingestion import IngestionError, MAX_PDF_BYTES as SERVICE_MAX_BYTES, ingest_pdf_bytes
from ..storage import get_document, insert_document, list_documents

log = logging.getLogger("taxshield.documents")
router = APIRouter(prefix="/api/documents", tags=["Documents"])
UPLOAD_LIMIT = min(MAX_PDF_BYTES, SERVICE_MAX_BYTES)
READ_BLOCK_BYTES = 64 * 1024


async def _read_limited(file: UploadFile) -> bytes:
    content = bytearray()
    while True:
        block = await file.read(READ_BLOCK_BYTES)
        if not block:
            break
        if len(content) + len(block) > UPLOAD_LIMIT:
            raise HTTPException(413, "PDF uploads are limited to 20 MB.")
        content.extend(block)
    if not content:
        raise HTTPException(400, "The uploaded PDF is empty.")
    return bytes(content)


def _public_document(row: dict) -> dict:
    return {
        "id": row["id"], "title": row["title"], "filename": row["filename"],
        "file_size": row["file_size"], "upload_timestamp": row["upload_timestamp"],
        "page_count": row["page_count"], "chunk_count": row["chunk_count"],
        "ingestion_status": row["ingestion_status"], "source_url": row["source_url"],
        "source_type": row["source_type"], "official_domain_verified": row["official_domain_verified"],
        "document_type": row["document_type"], "tax_period": row["tax_period"],
        "metadata": row["metadata"],
    }


@router.post("/upload", response_model=DocumentUploadResponse, summary="Upload and ingest a searchable PDF")
async def upload_document(
    file: UploadFile = File(...),
    source_url: str | None = Form(default=None),
    document_title: str | None = Form(default=None),
    document_type: str | None = Form(default=None),
    tax_period: str | None = Form(default=None),
):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Only files with a .pdf extension are accepted.")
    if file.content_type != "application/pdf":
        raise HTTPException(415, "The uploaded file must use application/pdf content type.")
    data = await _read_limited(file)
    try:
        extracted = await asyncio.to_thread(
            ingest_pdf_bytes, data, filename, content_type=file.content_type,
            document_title=document_title, source_url=source_url,
            document_type=document_type, tax_period=tax_period,
        )
    except IngestionError as exc:
        raise HTTPException(400, str(exc)) from exc

    existing = get_document(extracted["id"])
    if existing:
        return {
            **_public_document(existing),
            "extraction_limitations": existing["metadata"].get("extraction_limitations", []),
            "chunks_created": existing["chunk_count"],
        }

    # Dense vectors are retained for the existing retrieval service when it is configured.
    try:
        vectors = await embed_texts([chunk["text"] for chunk in extracted["chunks"]])
        if vectors and len(vectors) == len(extracted["chunks"]):
            for chunk, vector in zip(extracted["chunks"], vectors):
                chunk["embedding"] = vector
                chunk["embedding_model"] = EMBEDDING_MODEL
                chunk["embedding_dimensions"] = len(vector)
    except Exception:
        # Ingestion remains valid without optional retrieval embeddings.
        log.warning("Embedding generation unavailable for document_id=%s", extracted["id"])

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{extracted['id']}_{uuid.uuid4().hex}.pdf"
    destination = (UPLOAD_DIR / stored_filename).resolve()
    if destination.parent != UPLOAD_DIR.resolve():
        raise HTTPException(400, "Invalid upload destination.")
    temporary = UPLOAD_DIR / f".{uuid.uuid4().hex}.part"
    timestamp = datetime.now(timezone.utc).isoformat()
    document = {
        "id": extracted["id"], "title": extracted["title"], "filename": extracted["filename"],
        "stored_filename": stored_filename, "source_url": extracted["source_url"],
        "source_type": "user_upload", "authority": "User-provided document",
        "official_domain_verified": False, "metadata": extracted["metadata"],
        "sha256": hashlib.sha256(data).hexdigest(), "file_size": len(data),
        "page_count": extracted["page_count"], "created_at": timestamp,
        "upload_timestamp": timestamp, "chunk_count": extracted["chunk_count"],
        "ingestion_status": "completed", "document_type": extracted["document_type"],
        "tax_period": extracted["tax_period"],
    }
    try:
        await asyncio.to_thread(temporary.write_bytes, data)
        await asyncio.to_thread(temporary.replace, destination)
        insert_document(document, extracted["chunks"])
    except Exception as exc:
        for path in (temporary, destination):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("Could not clean failed upload artifact")
        log.exception("PDF ingestion persistence failed for document_id=%s", extracted["id"])
        raise HTTPException(500, "The PDF could not be saved and indexed. No successful document record was created.") from exc

    log.info("PDF ingestion completed document_id=%s page_count=%d chunk_count=%d",
             extracted["id"], extracted["page_count"], extracted["chunk_count"])
    return {
        **_public_document(document),
        "extraction_limitations": extracted["metadata"]["extraction_limitations"],
        "chunks_created": extracted["chunk_count"],
    }


@router.get("", response_model=dict, summary="List successfully ingested PDFs")
def documents():
    return {"documents": [_public_document(row) for row in list_documents()]}
