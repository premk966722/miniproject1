"""Hybrid evidence retrieval using BM25, semantic search, and RRF rank fusion."""
import logging

from .. import config
from . import keyword_search as keyword_service
from . import semantic_search as semantic_service

MAX_TOP_K = 20
log = logging.getLogger("taxshield.hybrid_retriever")


class HybridRetrievalError(RuntimeError):
    """Raised when one of the required hybrid retrieval services fails."""


def _merge_result(target: dict, source: dict) -> None:
    """Fill absent evidence fields without overwriting stored source values."""
    for key in ("document_id", "chunk_id", "document_title", "filename", "page_number",
                "text", "source_url", "document_type", "tax_period", "section_reference",
                "source_type", "authority", "official_domain_verified", "document_metadata"):
        if target.get(key) in (None, "") and source.get(key) not in (None, ""):
            target[key] = source[key]


def reciprocal_rank_fusion(
    keyword_results: list[dict],
    semantic_results: list[dict],
    *,
    rrf_k: int = config.RRF_K,
    top_k: int = 5,
) -> list[dict]:
    """Fuse ranked lists with 1/(rrf_k + one-based rank) contributions."""
    if not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("RRF_K must be a positive integer.")
    if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    combined: dict[str, dict] = {}
    lists = (("bm25", keyword_results), ("semantic", semantic_results))
    for method, results in lists:
        if not isinstance(results, list):
            raise HybridRetrievalError(f"The {method} retrieval service returned an invalid result list.")
        seen_in_list = set()
        rank = 0
        for result in results:
            if not isinstance(result, dict):
                continue
            chunk_id = result.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)
            rank += 1
            if chunk_id not in combined:
                combined[chunk_id] = {
                    "document_id": result.get("document_id"),
                    "chunk_id": chunk_id,
                    "document_title": result.get("document_title"),
                    "filename": result.get("filename"),
                    "page_number": result.get("page_number"),
                    "text": result.get("text") or "",
                    "source_url": result.get("source_url"),
                    "document_type": result.get("document_type"),
                    "tax_period": result.get("tax_period"),
                    "section_reference": result.get("section_reference"),
                    "source_type": result.get("source_type", "unknown"),
                    "authority": result.get("authority"),
                    "official_domain_verified": result.get("official_domain_verified") is True,
                    "document_metadata": result.get("document_metadata") if isinstance(result.get("document_metadata"), dict) else {},
                    "rrf_score": 0.0,
                    "bm25_score": None,
                    "semantic_score": None,
                    "retrieval_methods": [],
                }
            fused = combined[chunk_id]
            _merge_result(fused, result)
            score_key = "bm25_score" if method == "bm25" else "semantic_score"
            raw_score = result.get("retrieval_score")
            if raw_score is not None:
                try:
                    fused[score_key] = float(raw_score)
                except (TypeError, ValueError):
                    pass
            if method not in fused["retrieval_methods"]:
                fused["retrieval_methods"].append(method)
            fused["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(combined.values(), key=lambda item: (-item["rrf_score"], item["chunk_id"]))
    return ranked[:top_k]


def hybrid_search(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Retrieve independently, then fuse the ranked keyword and semantic results."""
    if not isinstance(query, str) or not query.strip() or len(query) > 1200:
        raise ValueError("Query must contain 1 to 1200 non-whitespace characters.")
    if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
    try:
        keyword_results = keyword_service.search_chunks(query.strip(), config.KEYWORD_CANDIDATES)
    except Exception as exc:
        raise HybridRetrievalError("Keyword retrieval failed.") from exc

    # Semantic search is optional: if sentence-transformers or faiss is not
    # installed, BM25 evidence should still be usable rather than failing the
    # whole chat request.
    try:
        semantic_results = semantic_service.semantic_search(query.strip(), config.SEMANTIC_CANDIDATES)
    except Exception:
        log.warning("Semantic retrieval unavailable; continuing with BM25-only results", exc_info=True)
        semantic_results = []

    return reciprocal_rank_fusion(
        keyword_results,
        semantic_results,
        rrf_k=config.RRF_K,
        top_k=top_k,
    )
