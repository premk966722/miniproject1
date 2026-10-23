"""Build citations exclusively from retrieved, stored evidence metadata."""
import re
from urllib.parse import urlparse

CITATION_MARKER = re.compile(r"\[(c_[A-Za-z0-9_-]{1,100})\]")
BRACKETED_CHUNK_ID = re.compile(r"\[(c_[^\]\s]+)\]")
OFFICIAL_DOMAINS = {"incometax.gov.in", "incometaxindia.gov.in", "indiacode.nic.in"}


class CitationValidationError(ValueError):
    """The generated answer cannot be safely linked to retrieved passages."""


def _is_verified_government_source(hit: dict) -> bool:
    if hit.get("official_domain_verified") is not True:
        return False
    if hit.get("source_type") not in {"government_reference", "official_url_pdf", "government_document"}:
        return False
    source_url = hit.get("source_url")
    if not isinstance(source_url, str):
        return False
    try:
        parsed = urlparse(source_url)
        host = (parsed.hostname or "").lower().rstrip(".")
        return parsed.scheme == "https" and any(
            host == domain or host.endswith(f".{domain}") for domain in OFFICIAL_DOMAINS
        )
    except ValueError:
        return False


def source_status(hit: dict) -> str:
    source_type = hit.get("source_type") or "unknown"
    if _is_verified_government_source(hit):
        return "government_origin_verified"
    if source_type == "user_upload":
        return "user_uploaded_unverified"
    return "provenance_unverified"


def _citation(hit: dict, citation_id: str) -> dict:
    source_type = hit.get("source_type") or "unknown"
    citation_source_status = source_status(hit)
    is_verified = citation_source_status == "government_origin_verified"
    page_number = hit.get("page_number")
    title = hit.get("document_title") or ""
    text = hit.get("text") or ""
    section = hit.get("section_reference")
    source_url = hit.get("source_url") if isinstance(hit.get("source_url"), str) else None
    score = hit.get("rrf_score")
    if not isinstance(score, (int, float)):
        score = 0.0
    authority = hit.get("authority") if isinstance(hit.get("authority"), str) else None
    document_title = title or None
    # The additional aliases preserve the existing React evidence cards.
    return {
        "citation_id": citation_id,
        "document_id": hit.get("document_id") or "",
        "chunk_id": citation_id,
        "document_title": title,
        "page_number": page_number,
        "section_reference": section if isinstance(section, str) else None,
        "source_url": source_url,
        "source_status": citation_source_status,
        "evidence_excerpt": text,
        "id": citation_id,
        "title": document_title,
        "authority": authority,
        "url": source_url,
        "reference": section if isinstance(section, str) else None,
        "excerpt": text,
        "score": float(score),
        "page_start": page_number,
        "page_end": page_number,
        "section": section if isinstance(section, str) else None,
        "source_type": source_type,
        "official_domain_verified": is_verified,
    }


def validate_answer_citations(answer: str, hits: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Strip invalid markers and return citation objects built from retrieved hits only."""
    if not isinstance(answer, str):
        raise CitationValidationError("Answer text must be a string.")
    by_id = {}
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        chunk_id = hit.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            by_id.setdefault(chunk_id, hit)

    accepted_ids = []
    invalid_ids = []

    def validate_marker(match):
        citation_id = match.group(1)
        if citation_id not in by_id:
            invalid_ids.append(citation_id)
            return ""
        if citation_id not in accepted_ids:
            accepted_ids.append(citation_id)
        return match.group(0)

    clean_answer = CITATION_MARKER.sub(validate_marker, answer)
    for marker in BRACKETED_CHUNK_ID.findall(answer):
        if marker not in by_id and marker not in invalid_ids:
            invalid_ids.append(marker)
            clean_answer = clean_answer.replace(f"[{marker}]", "")
    clean_answer = re.sub(r"[ \t]+([,.;])", r"\1", clean_answer)
    clean_answer = re.sub(r"[ \t]{2,}", " ", clean_answer).strip()
    citations = [_citation(by_id[citation_id], citation_id) for citation_id in accepted_ids]
    return clean_answer, citations, invalid_ids
