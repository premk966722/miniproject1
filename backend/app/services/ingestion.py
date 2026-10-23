"""Safe, page-aware PDF extraction for TAX-SHIELD uploads."""
import hashlib
import re
from pathlib import PurePath
from urllib.parse import urlparse

import pymupdf as fitz

MAX_PDF_BYTES = 20 * 1024 * 1024
CHUNK_CHARS = 800
CHUNK_OVERLAP_CHARS = 120


class IngestionError(ValueError):
    """Raised when a PDF cannot be safely ingested as searchable evidence."""


def sanitize_filename(filename: str | None) -> str:
    # Strip both POSIX and Windows path components before removing unsafe characters.
    leaf = (filename or "document.pdf").replace("\\", "/").split("/")[-1]
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", leaf).strip(" .")
    return safe[:180] or "document.pdf"


def _valid_source_url(source_url: str | None) -> str | None:
    if source_url is None or not source_url.strip():
        return None
    value = source_url.strip()
    parsed = urlparse(value)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise IngestionError("Source URL must use http or https and include a host name.")
    return value


def _boundary(text: str, target: int) -> int:
    """Choose a nearby paragraph/sentence/word boundary before the target."""
    floor = max(1, target // 2)
    window = text[floor:target]
    for pattern in (r"\n\s*\n", r"[.!?][\"’')\]]?\s+", r"\s+"):
        matches = list(re.finditer(pattern, window))
        if matches:
            return floor + matches[-1].end()
    return target


def split_page_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Chunk one page with a character overlap while favoring legal prose boundaries."""
    if chunk_chars < 200 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("Chunk settings require chunk_chars >= 200 and 0 <= overlap < chunk_chars")
    text = re.sub(r"[\t\r\f\v ]+", " ", text.replace("\r\n", "\n")).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        target = min(start + chunk_chars, len(text))
        end = target if target == len(text) else _boundary(text, target)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap_chars)
        # Prefer a word boundary within the overlap. If an unusually long token
        # fills the entire overlap, keep the raw offset rather than dropping text.
        while next_start < end and not text[next_start - 1].isspace() and not text[next_start].isspace():
            next_start += 1
        if next_start >= end:
            next_start = max(start + 1, end - overlap_chars)
        start = next_start
    return chunks


def ingest_pdf_bytes(
    data: bytes,
    filename: str,
    *,
    content_type: str = "application/pdf",
    document_title: str | None = None,
    source_url: str | None = None,
    document_type: str | None = None,
    tax_period: str | None = None,
) -> dict:
    """Extract text and metadata. Does not write files or mark a document persisted."""
    safe_filename = sanitize_filename(filename)
    if not safe_filename.lower().endswith(".pdf"):
        raise IngestionError("Only files with a .pdf extension are accepted.")
    if content_type != "application/pdf":
        raise IngestionError("The uploaded file must use application/pdf content type.")
    if not data:
        raise IngestionError("The uploaded PDF is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise IngestionError("PDF uploads are limited to 20 MB.")
    if not data.startswith(b"%PDF-"):
        raise IngestionError("The uploaded file does not have a valid PDF signature.")
    source_url = _valid_source_url(source_url)
    document_id = "doc_" + hashlib.sha256(data).hexdigest()[:24]
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError("The PDF is corrupted or cannot be opened.") from exc
    try:
        if pdf.needs_pass:
            raise IngestionError("Password-protected PDFs are not supported. Upload an unlocked PDF.")
        if len(pdf) == 0:
            raise IngestionError("The PDF has no pages.")
        metadata = pdf.metadata or {}
        title = (document_title or metadata.get("title") or "").strip() or PurePath(safe_filename).stem
        chunks = []
        empty_pages = []
        for page_index, page in enumerate(pdf, start=1):
            page_text = page.get_text("text", sort=True) or ""
            if not page_text.strip():
                empty_pages.append(page_index)
                continue
            for chunk_index, chunk_text in enumerate(split_page_text(page_text), start=1):
                chunk_hash = hashlib.sha256(
                    f"{document_id}|{page_index}|{chunk_index}|{chunk_text}".encode("utf-8")
                ).hexdigest()[:24]
                chunks.append({
                    "document_id": document_id,
                    "chunk_id": f"c_{chunk_hash}",
                    "document_title": title,
                    "filename": safe_filename,
                    "page_number": page_index,
                    "text": chunk_text,
                    "source_url": source_url,
                    "document_type": document_type.strip() if document_type and document_type.strip() else None,
                    "tax_period": tax_period.strip() if tax_period and tax_period.strip() else None,
                })
        if not chunks:
            raise IngestionError("No selectable text was found. OCR image-only PDFs before uploading.")
        return {
            "id": document_id,
            "document_id": document_id,
            "title": title,
            "filename": safe_filename,
            "file_size": len(data),
            "page_count": len(pdf),
            "chunk_count": len(chunks),
            "ingestion_status": "completed",
            "upload_timestamp": None,
            "source_url": source_url,
            "document_type": document_type.strip() if document_type and document_type.strip() else None,
            "tax_period": tax_period.strip() if tax_period and tax_period.strip() else None,
            "source_type": "user_upload",
            "official_domain_verified": False,
            "metadata": {
                "pdf_title": metadata.get("title") or None,
                "author": metadata.get("author") or None,
                "creation_date": metadata.get("creationDate") or None,
                "empty_pages": empty_pages,
                "extraction_limitations": ([f"No text could be extracted from page(s): {', '.join(map(str, empty_pages))}."]
                    if empty_pages else []),
                "source_url_user_supplied": source_url is not None,
            },
            "chunks": chunks,
        }
    finally:
        pdf.close()
