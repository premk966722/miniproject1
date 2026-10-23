"""Cached BM25 keyword retrieval over successfully ingested SQLite chunks."""
import re
import threading
import unicodedata

from rank_bm25 import BM25Okapi

from .. import storage
from ..config import TOP_K

MAX_TOP_K = 20
TOKEN_PATTERN = re.compile(r"[a-z]+[0-9]*|[0-9]+[a-z]*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Normalize consistently while retaining statutory numbers and mixed IDs."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return TOKEN_PATTERN.findall(normalized)


def _metadata(row: dict) -> dict:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def _search_result(row: dict, score: float) -> dict:
    metadata = _metadata(row)
    document_metadata = row.get("document_metadata")
    if not isinstance(document_metadata, dict):
        document_metadata = {}
    return {
        "document_id": str(row.get("document_id") or ""),
        "chunk_id": str(row.get("id") or ""),
        "document_title": str(row.get("title") or row.get("document_title") or ""),
        "filename": str(row.get("filename") or ""),
        "page_number": row.get("page_start") if row.get("page_start") is not None else row.get("page_end"),
        "text": row.get("text") or "",
        "source_url": row.get("source_url"),
        "document_type": row.get("document_type") or metadata.get("document_type") or document_metadata.get("document_type"),
        "tax_period": row.get("tax_period") or metadata.get("tax_period") or document_metadata.get("tax_period"),
        "section_reference": row.get("section") or metadata.get("section_reference"),
        "source_type": row.get("source_type") or "unknown",
        "authority": row.get("authority"),
        "official_domain_verified": row.get("official_domain_verified") is True,
        "document_metadata": document_metadata,
        "retrieval_method": "bm25",
        "retrieval_score": float(score),
    }


class KeywordSearchEngine:
    """Thread-safe, refresh-on-write BM25 index backed by TAX-SHIELD SQLite."""

    def __init__(self, top_k: int = TOP_K):
        self.top_k = max(1, min(int(top_k), MAX_TOP_K))
        self._lock = threading.RLock()
        self._version = None
        self._rows: list[dict] = []
        self._tokenized: list[list[str]] = []
        self._index: BM25Okapi | None = None

    def _refresh_if_needed(self) -> None:
        version = storage.keyword_chunks_version()
        if version == self._version:
            return
        rows = []
        tokenized = []
        for row in storage.list_chunks():
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            tokens = tokenize(" ".join((row.get("title") or "", row.get("section") or "", text)))
            if tokens:
                rows.append(row)
                tokenized.append(tokens)
        self._rows = rows
        self._tokenized = tokenized
        self._index = BM25Okapi(tokenized) if tokenized else None
        if self._index is not None:
            # rank-bm25's default epsilon can itself be negative when a tiny
            # corpus contains mostly ubiquitous terms. Keep matching terms
            # searchable in demos and small document collections.
            for term, idf in self._index.idf.items():
                if idf <= 0:
                    self._index.idf[term] = 0.01
        self._version = version

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        limit = self.top_k if top_k is None else max(1, min(int(top_k), MAX_TOP_K))
        with self._lock:
            self._refresh_if_needed()
            if self._index is None:
                return []
            scores = self._index.get_scores(query_tokens)
            candidates = []
            seen = set()
            for index, raw_score in enumerate(scores):
                score = float(raw_score)
                row = self._rows[index]
                chunk_id = row.get("id")
                # Ignore zero-score and invalid candidates, and guard against duplicate DB rows.
                if score <= 0 or not isinstance(chunk_id, str) or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                candidates.append((index, score))
            candidates.sort(key=lambda pair: (-pair[1], str(self._rows[pair[0]].get("id") or "")))
            return [_search_result(self._rows[index], score) for index, score in candidates[:limit]]


keyword_search = KeywordSearchEngine()


def search_chunks(query: str, top_k: int | None = None) -> list[dict]:
    return keyword_search.search(query, top_k=top_k)
