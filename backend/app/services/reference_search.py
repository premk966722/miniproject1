"""Cached BM25 retrieval over curated official references.

The upload corpus lives in SQLite and is searched by ``keyword_search`` and
``semantic_search``.  This module searches the department references shipped
with the application (``sources.json`` and ``knowledge_base.json``) so that a
fresh installation can answer common tax questions before any PDF is uploaded.
"""
import hashlib
import json
import logging
import threading
from pathlib import Path
from urllib.parse import urlparse

from rank_bm25 import BM25Okapi

from ..config import DATA_DIR, TOP_K
from .citation_validation import OFFICIAL_DOMAINS
from .keyword_search import tokenize

MAX_TOP_K = 20
REFERENCE_FILES = ("sources.json", "knowledge_base.json")

log = logging.getLogger("taxshield.reference_search")


def _is_official_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return parsed.scheme == "https" and any(
        host == domain or host.endswith(f".{domain}") for domain in OFFICIAL_DOMAINS
    )


def _chunk_id(filename: str, entry: dict) -> str:
    identity = "\x1f".join((
        filename,
        str(entry.get("id") or ""),
        str(entry.get("url") or ""),
        str(entry.get("text") or ""),
    ))
    return "c_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _search_result(entry: dict, filename: str, score: float) -> dict:
    url = entry.get("url")
    source_url = url if isinstance(url, str) else None
    chunk_id = _chunk_id(filename, entry)
    title = str(entry.get("title") or "").strip()
    reference = entry.get("reference")
    return {
        "document_id": f"reference:{entry.get('id')}",
        "chunk_id": chunk_id,
        "document_title": title,
        "filename": filename,
        "page_number": None,
        "text": str(entry.get("text") or "").strip(),
        "source_url": source_url,
        "document_type": entry.get("document_type") or "official_reference",
        "tax_period": entry.get("tax_period"),
        "section_reference": reference if isinstance(reference, str) else None,
        "source_type": "government_reference",
        "authority": entry.get("authority") or "Income Tax Department, Government of India",
        "official_domain_verified": bool(source_url and _is_official_url(source_url)),
        "document_metadata": {
            "pdf_title": title,
            "reference": reference,
            "checked": entry.get("checked"),
            "reference_file": filename,
        },
        "retrieval_method": "reference_bm25",
        "retrieval_score": float(score),
    }


class ReferenceSearchEngine:
    """Thread-safe BM25 index that reloads when a reference file changes."""

    def __init__(self, data_dir: Path = DATA_DIR, files: tuple[str, ...] = REFERENCE_FILES,
                 top_k: int = TOP_K):
        self.data_dir = Path(data_dir)
        self.files = files
        self.top_k = max(1, min(int(top_k), MAX_TOP_K))
        self._lock = threading.RLock()
        self._signature = None
        self._entries: list[dict] = []
        self._sources: list[str] = []
        self._tokenized: list[list[str]] = []
        self._index: BM25Okapi | None = None

    def _signature_for_files(self) -> tuple:
        signature = []
        for name in self.files:
            path = self.data_dir / name
            signature.append((name, path.stat().st_mtime_ns, path.stat().st_size)
                             if path.is_file() else (name, 0, 0))
        return tuple(signature)

    def _refresh_if_needed(self) -> None:
        signature = self._signature_for_files()
        if signature == self._signature:
            return
        entries: list[dict] = []
        sources: list[str] = []
        tokenized: list[list[str]] = []
        seen_ids: set[str] = set()
        for name in self.files:
            path = self.data_dir / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                log.exception("Could not read official reference file %s", path)
                continue
            if not isinstance(payload, list):
                log.warning("Official reference file %s must contain a JSON list", path)
                continue
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                text = entry.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                chunk_id = _chunk_id(name, entry)
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                title = str(entry.get("title") or "")
                tokens = tokenize(" ".join((
                    title, title, title,
                    str(entry.get("reference") or ""),
                    text,
                )))
                if tokens:
                    entries.append(entry)
                    sources.append(name)
                    tokenized.append(tokens)
        index = BM25Okapi(tokenized) if tokenized else None
        if index is not None:
            # A shared term can receive a non-positive IDF in a small corpus.
            # Keep it mildly searchable rather than dropping the match.
            for term, idf in index.idf.items():
                if idf <= 0:
                    index.idf[term] = 0.01
        self._entries = entries
        self._sources = sources
        self._tokenized = tokenized
        self._index = index
        self._signature = signature

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 1200:
            raise ValueError("Query must contain 1 to 1200 non-whitespace characters.")
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
            for position, raw_score in enumerate(scores):
                score = float(raw_score)
                if score <= 0:
                    continue
                candidates.append((score, position))
            candidates.sort(key=lambda item: (
                -item[0], _chunk_id(self._sources[item[1]], self._entries[item[1]])
            ))
            return [
                _search_result(self._entries[position], self._sources[position], score)
                for score, position in candidates[:limit]
            ]


    def entry_count(self) -> int:
        """Number of loaded reference passages after refreshing the cache."""
        with self._lock:
            self._refresh_if_needed()
            return len(self._entries)


reference_engine = ReferenceSearchEngine()


def search_references(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return the highest-scoring official reference passages for a query."""
    return reference_engine.search(query, top_k=top_k)


def reference_entry_count() -> int:
    """Number of loaded official reference passages, for health reporting."""
    return reference_engine.entry_count()
