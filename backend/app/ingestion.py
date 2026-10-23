"""PDF validation, extraction and legal-context-preserving chunking."""
import hashlib
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import urlparse

from pypdf import PdfReader

from .config import CHUNK_OVERLAP_WORDS, MAX_CHUNK_WORDS, MAX_PDF_BYTES

OFFICIAL_HOSTS = {"incometax.gov.in", "incometaxindia.gov.in", "egazette.gov.in", "indiacode.nic.in"}
SECTION_RE = re.compile(r"(?im)^\s*((?:section|sec\.?|article|rule|schedule)\s+[\w()/-]+[^\n]{0,100})")
YEAR_RE = re.compile(r"\b(20\d{2})\s*(?:-|to|–)\s*(\d{2,4})\b", re.I)
EFFECTIVE_RE = re.compile(r"(?i)(?:effective|with effect from|comes into force|commencing on)\s+(?:from\s+)?([^.;\n]{3,60})")


class PDFIngestionError(ValueError):
    pass


def official_domain(url: str | None) -> bool:
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
        return any(host == domain or host.endswith("." + domain) for domain in OFFICIAL_HOSTS)
    except ValueError:
        return False


def chunk_pages(pages: list[dict], max_words: int = MAX_CHUNK_WORDS, overlap: int = CHUNK_OVERLAP_WORDS, doc_id: str = "doc") -> list[dict]:
    if max_words < 100 or overlap < 0 or overlap >= max_words:
        raise ValueError("Chunk settings must use max_words >= 100 and 0 <= overlap < max_words")
    out = []
    words: list[str] = []
    page_numbers: list[int] = []
    sections: list[str] = []

    def flush():
        if not words:
            return
        text = " ".join(words).strip()
        first, last = min(page_numbers), max(page_numbers)
        section = next((s for s in reversed(sections) if s), None)
        suffix = hashlib.sha256(f"{doc_id}|{first}|{last}|{text}".encode()).hexdigest()[:14]
        out.append({"id": f"c_{suffix}", "text": text, "page_start": first, "page_end": last, "section": section})

    for page in pages:
        pnum = int(page["page"])
        text = re.sub(r"\s+", " ", page.get("text", "")).strip()
        if not text:
            continue
        section = None
        # Add a page heading when present, while preserving page attribution.
        sec_match = SECTION_RE.search(page.get("text", ""))
        if sec_match:
            section = sec_match.group(1).strip()
        page_words = text.split()
        cursor = 0
        while cursor < len(page_words):
            room = max_words - len(words)
            take = page_words[cursor:cursor + room]
            words.extend(take)
            page_numbers.extend([pnum] * len(take))
            sections.append(section)
            cursor += len(take)
            if len(words) >= max_words:
                flush()
                keep = min(overlap, len(words))
                words = words[-keep:] if keep else []
                page_numbers = page_numbers[-keep:] if keep else []
                sections = sections[-1:] if keep and sections else []
    flush()
    return out


def ingest_pdf(filename: str, content_type: str, data: bytes, source_url: str | None = None, source_type: str = "upload") -> tuple[dict, list[dict]]:
    if not filename.lower().endswith(".pdf") or content_type not in ("application/pdf", "application/octet-stream"):
        raise PDFIngestionError("Only PDF uploads are accepted.")
    if not data or len(data) > MAX_PDF_BYTES:
        raise PDFIngestionError(f"PDF must be between 1 byte and {MAX_PDF_BYTES // (1024 * 1024)} MB.")
    if not data.startswith(b"%PDF-"):
        raise PDFIngestionError("The uploaded file does not have a valid PDF signature.")
    if source_url:
        parsed_url = urlparse(source_url)
        if parsed_url.scheme not in ("https", "http") or not parsed_url.hostname:
            raise PDFIngestionError("Source links must use http or https and include a host name.")
    safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", filename.split("/")[-1].split("\\")[-1])[:180] or "document.pdf"
    try:
        reader = PdfReader(BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise PDFIngestionError("Encrypted PDFs are not supported. Upload an unlocked PDF.")
        pages = [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(reader.pages)]
    except PDFIngestionError:
        raise
    except Exception as exc:
        raise PDFIngestionError("Could not read this PDF. It may be damaged or image-only; use a text-searchable PDF.") from exc
    if not any(p["text"].strip() for p in pages):
        raise PDFIngestionError("No selectable text was found. OCR the PDF before uploading it.")
    info = reader.metadata or {}
    title = str(info.title or safe_name.rsplit(".", 1)[0])[:240]
    all_text = "\n".join(p["text"] for p in pages)
    year_match = YEAR_RE.search(title + " " + all_text[:3000])
    effective_match = EFFECTIVE_RE.search(all_text[:12000])
    metadata = {"pdf_title": str(info.title or ""), "author": str(info.author or ""), "creation_date": str(info.creation_date or ""),
                "year_range": year_match.group(0) if year_match else None,
                "effective_date_text": effective_match.group(1).strip() if effective_match else None,
                "source_url": source_url, "source_type": source_type,
                "official_domain_match": official_domain(source_url), "source_url_user_supplied": bool(source_url)}
    doc_id = uuid.uuid4().hex
    document = {"id": doc_id, "title": title, "filename": safe_name, "source_url": source_url,
                "source_type": "official_url_pdf" if source_type == "official_url_pdf" and official_domain(source_url) else "user_upload",
                "authority": str(info.author or "") or "User-provided document",
                "official_domain_verified": False, "metadata": metadata,
                "sha256": hashlib.sha256(data).hexdigest(), "page_count": len(reader.pages),
                "created_at": datetime.now(timezone.utc).isoformat()}
    chunks = chunk_pages(pages, doc_id=doc_id)
    for chunk in chunks:
        chunk["metadata"] = {"year_range": metadata["year_range"], "effective_date_text": metadata["effective_date_text"],
                             "source_type": document["source_type"], "document_type": "pdf"}
    return document, chunks
