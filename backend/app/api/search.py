"""Keyword-search API backed by the persistent extracted-chunk store."""
import logging

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from ..schemas import (
    HybridSearchResponse,
    KeywordSearchRequest,
    KeywordSearchResponse,
)
from ..services.hybrid_retriever import HybridRetrievalError, hybrid_search
from ..services.semantic_search import EmbeddingModelUnavailable, semantic_search
from ..services.keyword_search import search_chunks

log = logging.getLogger("taxshield.keyword_search")
router = APIRouter(prefix="/api/search", tags=["Search"])


@router.post("/keyword", response_model=KeywordSearchResponse,
             summary="Search ingested PDF passages with BM25")
async def keyword_search(request: KeywordSearchRequest):
    try:
        results = await run_in_threadpool(search_chunks, request.query, request.top_k)
    except Exception as exc:
        log.exception("Keyword retrieval failed")
        raise HTTPException(status_code=503, detail="Keyword search is temporarily unavailable.") from exc
    return {"query": request.query, "retrieval_method": "bm25", "results": results}


@router.post("/semantic", response_model=KeywordSearchResponse,
             summary="Search ingested PDF passages by semantic similarity")
async def semantic(request: KeywordSearchRequest):
    try:
        results = await run_in_threadpool(semantic_search, request.query, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingModelUnavailable as exc:
        log.warning("Semantic retrieval unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Semantic search is unavailable. Check the configured embedding model and semantic dependencies.",
        ) from exc
    except Exception as exc:
        log.exception("Semantic retrieval failed")
        raise HTTPException(status_code=503, detail="Semantic search is temporarily unavailable.") from exc
    return {"query": request.query, "retrieval_method": "semantic", "results": results}


@router.post("/hybrid", response_model=HybridSearchResponse,
             summary="Fuse BM25 and semantic retrieval with reciprocal rank fusion")
async def hybrid(request: KeywordSearchRequest):
    try:
        results = await run_in_threadpool(hybrid_search, request.query, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HybridRetrievalError as exc:
        log.exception("Hybrid retrieval failed")
        raise HTTPException(status_code=503, detail="Hybrid search is temporarily unavailable.") from exc
    return {
        "query": request.query,
        "retrieval_method": "hybrid",
        "result_count": len(results),
        "results": results,
    }
