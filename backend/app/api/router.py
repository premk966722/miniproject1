import logging
from importlib.util import find_spec

from fastapi import APIRouter, HTTPException

from ..answering import NOTICE, compose
from ..config import (APP_NAME, EMBEDDING_MODEL, EMBEDDING_PROVIDER, LLM_API_KEY, LLM_BASE_URL,
                      LLM_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL, TOP_K)
from ..retrieval import retriever
from ..schemas import AskRequest, AskResponse, Citation
from ..schemas import ChatRequest, ChatResponse
from ..services.answer_service import AnswerRetrievalError, answer_question
from ..services.reference_search import reference_entry_count
from ..services.answer_generator import (
    LLMConfigurationError, LLMMalformedResponse, LLMProviderError, LLMRateLimitError,
)
from ..storage import list_chunks, list_documents

router = APIRouter(prefix="/api", tags=["TAX-SHIELD API"])
log = logging.getLogger("taxshield.chat")


def _citation(doc: dict) -> Citation:
    return Citation(
        id=doc["id"], title=doc["title"], authority=doc.get("authority") or "Uploaded document",
        url=doc.get("source_url") or doc.get("url"),
        reference=doc.get("section") or doc.get("reference"), excerpt=doc["text"],
        score=round(doc.get("score", 0), 5), page_start=doc.get("page_start"),
        page_end=doc.get("page_end"), section=doc.get("section"),
        source_type=doc.get("source_type", "unknown"),
        official_domain_verified=bool(doc.get("official_domain_verified", False)),
    )


async def _chat(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    query = f"{question} {request.tax_year or ''}".strip()
    hits = await retriever.search(query, k=TOP_K, filters=request.filters)
    answer, validated_citations, invalid = await compose(question, hits)
    score = min(1.0, hits[0]["score"] / (2 / 61)) if hits else 0.0
    return AskResponse(
        answer=answer, citations=[_citation({**d, "score": d.get("score", 0)}) for d in validated_citations],
        confidence=round(score, 2), grounded=bool(validated_citations), notice=NOTICE,
        citation_warning="Unrecognized citation markers were removed." if invalid else None,
    )


@router.get("/health", summary="Backend health", description="Reports the API and retrieval configuration health.")
def health():
    provider = (LLM_PROVIDER or "").lower()
    llm_configured = bool(
        LLM_MODEL and ((provider in ("openai", "openai-compatible", "groq") and LLM_API_KEY and LLM_BASE_URL)
                       or (provider == "ollama" and OLLAMA_BASE_URL))
    )
    return {
        "status": "healthy", "application": APP_NAME,
        "seed_sources": len(retriever.seed), "uploaded_documents": len(list_documents()),
        "uploaded_chunks": len(list_chunks()), "embedding_provider": EMBEDDING_PROVIDER,
        "official_reference_chunks": reference_entry_count(),
        "retrieval_mode": "bm25+dense-vector+rrf" if EMBEDDING_PROVIDER != "none" else "bm25+tfidf-fallback+rrf",
        "embedding_model": EMBEDDING_MODEL if EMBEDDING_PROVIDER != "none" else None,
        "chat_retrieval_mode": "bm25+sentence-transformers+faiss+rrf",
        "chat_readiness": {
            "faiss_installed": find_spec("faiss") is not None,
            "sentence_transformers_installed": find_spec("sentence_transformers") is not None,
            "llm_provider": provider or "unconfigured",
            "llm_provider_configured": bool(llm_configured),
        },
    }


@router.get("/sources", summary="List built-in government references")
def sources():
    return [{"id": d["id"], "title": d["title"], "authority": d["authority"], "url": d["url"],
             "reference": d["reference"], "checked": d.get("checked"), "source_type": "government_reference",
             "official_domain_verified": True} for d in retriever.seed]


@router.post("/chat", response_model=ChatResponse, summary="Answer a tax research question from hybrid-retrieved evidence")
async def chat(request: ChatRequest):
    tax_period = request.tax_period or request.tax_year
    try:
        return await answer_question(request.question, tax_period=tax_period, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AnswerRetrievalError as exc:
        log.warning("Hybrid retrieval unavailable for chat request")
        raise HTTPException(status_code=503, detail="Evidence retrieval is temporarily unavailable.") from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMMalformedResponse as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse, include_in_schema=False)
async def ask_legacy(request: AskRequest):
    """Backward-compatible Stage 1 alias."""
    return await _chat(request)
