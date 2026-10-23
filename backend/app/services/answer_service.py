"""Orchestration for hybrid evidence retrieval, grounded generation, and citations."""
import logging
import time

from starlette.concurrency import run_in_threadpool

from .. import config
from ..answering import NOTICE
from .answer_generator import (
    LLMConfigurationError, LLMMalformedResponse, LLMProviderError, LLMRateLimitError,
    LLMServiceError, generate_answer,
)
from .citation_validation import validate_answer_citations
from .extractive_answer import build_extractive_answer, conversational_reply, guidance_answer
from .hybrid_retriever import HybridRetrievalError, hybrid_search
from .reference_search import search_references

log = logging.getLogger("taxshield.answer_service")

# After a provider failure, briefly stop calling the LLM so that every follow-up
# question is answered immediately from the retrieved evidence instead of
# waiting on another rate-limited request.
_llm_cooldown_until = 0.0
_RATE_LIMIT_COOLDOWN_SECONDS = 45.0
_PROVIDER_COOLDOWN_SECONDS = 20.0


class AnswerRetrievalError(RuntimeError):
    """The hybrid evidence retriever could not complete the request."""


def _reset_llm_cooldown() -> None:
    """Reset the provider cooldown (used by tests)."""
    global _llm_cooldown_until
    _llm_cooldown_until = 0.0


def _llm_in_cooldown() -> bool:
    return time.monotonic() < _llm_cooldown_until


def _note_llm_failure(exc: Exception) -> None:
    global _llm_cooldown_until
    seconds = _RATE_LIMIT_COOLDOWN_SECONDS if isinstance(exc, LLMRateLimitError) else _PROVIDER_COOLDOWN_SECONDS
    _llm_cooldown_until = max(_llm_cooldown_until, time.monotonic() + seconds)


def _fallback_reason(exc: Exception) -> str:
    if isinstance(exc, LLMRateLimitError):
        return "rate_limited"
    if isinstance(exc, LLMConfigurationError):
        return "not_configured"
    if isinstance(exc, LLMMalformedResponse):
        return "malformed_response"
    if isinstance(exc, LLMProviderError):
        return "provider_error"
    return "unavailable"


def _fuse_evidence(
    uploaded_hits: list[dict],
    reference_hits: list[dict],
    top_k: int,
    *,
    rrf_k: int = config.RRF_K,
) -> list[dict]:
    """Fuse uploaded and official-reference rankings with RRF.

    Uploaded chunks win exact rank ties so a user-provided document remains the
    primary evidence when both corpora contain an equally ranked passage.
    """
    combined: dict[str, dict] = {}
    ranked_lists = (
        ("uploaded_hybrid", uploaded_hits, 0),
        ("official_reference", reference_hits, 1),
    )
    for method, results, source_priority in ranked_lists:
        rank = 0
        seen = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            chunk_id = result.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            fused = combined.get(chunk_id)
            if fused is None:
                fused = {
                    **result,
                    "rrf_score": 0.0,
                    "retrieval_methods": [],
                    "_source_priority": source_priority,
                }
                combined[chunk_id] = fused
            else:
                fused["_source_priority"] = min(fused["_source_priority"], source_priority)
                for key, value in result.items():
                    if fused.get(key) in (None, "") and value not in (None, ""):
                        fused[key] = value
            methods = fused["retrieval_methods"]
            for existing in result.get("retrieval_methods") or []:
                if isinstance(existing, str) and existing not in methods:
                    methods.append(existing)
            if method not in methods:
                methods.append(method)
            fused["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(
        combined.values(),
        key=lambda item: (-item["rrf_score"], item["_source_priority"], item["chunk_id"]),
    )
    for hit in ranked:
        hit.pop("_source_priority", None)
    return ranked[:top_k]


def _result(*, question, answer, status, insufficient, citations, hits, top_k, tax_period,
            uploaded_hits=(), reference_hits=(), retrieval_error=None, generation_mode="llm",
            fallback_reason=None, warning=None, notice=NOTICE) -> dict:
    """Assemble the response shape shared by every answer path."""
    if generation_mode == "conversational":
        retrieval_method = "conversational"
    elif generation_mode == "guidance":
        retrieval_method = "none"
    else:
        retrieval_method = "hybrid_rrf+official_reference" if reference_hits else "hybrid_rrf"
    top_score = max((float(hit.get("rrf_score", 0.0)) for hit in hits), default=0.0)
    # Compatibility signal for the current UI; it is a normalized rank signal, not a probability.
    confidence_signal = min(1.0, top_score * (config.RRF_K + 1) / 2)
    return {
        "question": question.strip(),
        "answer": answer,
        "status": status,
        "insufficient_evidence": insufficient,
        "citations": citations,
        "retrieval_metadata": {
            "retrieval_method": retrieval_method,
            "retrieved_count": len(hits),
            "top_k": top_k,
            "requested_tax_period": tax_period,
            "rrf_k": config.RRF_K,
            "keyword_candidates": config.KEYWORD_CANDIDATES,
            "semantic_candidates": config.SEMANTIC_CANDIDATES,
            "uploaded_candidate_count": len(uploaded_hits),
            "reference_candidate_count": len(reference_hits),
            "reference_retrieval_used": bool(reference_hits),
            "reference_fallback_used": retrieval_error is not None,
            "retrieval_methods": sorted({method for hit in hits for method in hit.get("retrieval_methods", [])}),
            "generation_mode": generation_mode,
            "generation_fallback_reason": fallback_reason,
            "confidence_signal_type": "normalized_rrf_rank_signal_not_probability",
        },
        "grounded": bool(citations) and not insufficient,
        "confidence": round(confidence_signal, 4),
        "notice": notice,
        "citation_warning": warning,
    }


async def answer_question(question: str, tax_period: str | None = None, top_k: int = config.TOP_K) -> dict:
    if not isinstance(question, str) or not question.strip() or len(question) > 1200:
        raise ValueError("Question must contain 1 to 1200 non-whitespace characters.")
    if not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    conversational = conversational_reply(question)
    if conversational is not None:
        return _result(
            question=question, answer=conversational, status="answered", insufficient=False,
            citations=[], hits=[], top_k=top_k, tax_period=tax_period,
            generation_mode="conversational",
        )

    query = question.strip()
    if tax_period and tax_period.strip():
        query = f"{query} {tax_period.strip()}"

    retrieval_error = None
    try:
        uploaded_hits = await run_in_threadpool(hybrid_search, query, top_k)
    except HybridRetrievalError as exc:
        log.warning("Uploaded-document retrieval failed; falling back to official references")
        uploaded_hits = []
        retrieval_error = exc

    try:
        reference_hits = await run_in_threadpool(search_references, query, top_k)
    except Exception:
        log.exception("Official reference retrieval failed")
        reference_hits = []

    if retrieval_error is not None and not reference_hits:
        raise AnswerRetrievalError("Evidence retrieval failed.") from retrieval_error

    hits = _fuse_evidence(uploaded_hits, reference_hits, top_k)

    # A fresh workspace with no PDFs may still have no match for an off-topic
    # question.  Return useful scope guidance instead of calling the model.
    if not hits:
        return _result(
            question=question, answer=guidance_answer(), status="insufficient_evidence",
            insufficient=True, citations=[], hits=[], top_k=top_k, tax_period=tax_period,
            uploaded_hits=uploaded_hits, reference_hits=reference_hits,
            retrieval_error=retrieval_error, generation_mode="guidance",
        )

    generation_mode = "llm"
    fallback_reason = None
    warning = None
    notice = NOTICE

    if _llm_in_cooldown():
        generated = build_extractive_answer(question, hits, reason="cooldown")
        if generated is None:
            return _result(
                question=question, answer=guidance_answer(), status="insufficient_evidence",
                insufficient=True, citations=[], hits=hits, top_k=top_k, tax_period=tax_period,
                uploaded_hits=uploaded_hits, reference_hits=reference_hits,
                retrieval_error=retrieval_error, generation_mode="guidance",
                fallback_reason="cooldown",
            )
        generation_mode = "extractive_fallback"
        fallback_reason = "cooldown"
    else:
        try:
            generated = await generate_answer(question.strip(), hits, tax_period=tax_period)
        except Exception as exc:  # noqa: BLE001 - every provider failure must still yield an answer
            if isinstance(exc, LLMServiceError):
                log.warning("LLM generation failed (%s); using extractive evidence", type(exc).__name__)
            else:
                log.exception("Unexpected generation failure; using extractive evidence")
            _note_llm_failure(exc)
            fallback_reason = _fallback_reason(exc)
            generated = build_extractive_answer(question, hits, reason="unavailable")
            if generated is None:
                return _result(
                    question=question, answer=guidance_answer(), status="insufficient_evidence",
                    insufficient=True, citations=[], hits=hits, top_k=top_k, tax_period=tax_period,
                    uploaded_hits=uploaded_hits, reference_hits=reference_hits,
                    retrieval_error=retrieval_error, generation_mode="guidance",
                    fallback_reason=fallback_reason,
                )
            generation_mode = "extractive_fallback"

    answer, citations, invalid = validate_answer_citations(generated["answer"], hits)
    insufficient = generated["insufficient_evidence"]
    needs_clarification = generated["needs_clarification"]
    if invalid:
        warning = "Unrecognized citation markers were removed; only retrieved passages are cited."

    # The model may refuse or return unverifiable markers even when relevant
    # passages exist.  In that case quote the retrieved evidence directly rather
    # than ending the conversation with an error or an empty answer.
    if (insufficient or not citations) and hits:
        reason = "model_insufficient" if insufficient else "invalid_citations"
        fallback = build_extractive_answer(question, hits, reason=reason)
        if fallback is not None:
            generated = fallback
            generation_mode = "extractive_fallback"
            fallback_reason = reason
            answer, citations, invalid = validate_answer_citations(fallback["answer"], hits)
            insufficient = False
            needs_clarification = False
            warning = None

    if not citations and not insufficient:
        insufficient = True
        needs_clarification = False
        warning = warning or "The generated response did not cite a retrieved evidence passage."
        answer = (
            "I found related passages, but I could not verify a citable answer from them. "
            "Try naming the assessment year, section, or form (for example, 'AY 2026-27', "
            "'section 80C', or 'ITR-2'), or upload the relevant PDF and ask again."
        )

    if insufficient and not citations:
        answer = guidance_answer()
        needs_clarification = False

    if generation_mode == "extractive_fallback":
        note = generated.get("fallback_note")
        if note:
            notice = f"{NOTICE} {note}"

    if insufficient:
        status = "insufficient_evidence"
    elif needs_clarification:
        status = "needs_clarification"
        clarification = generated.get("clarification_question")
        if clarification and clarification.casefold() not in answer.casefold():
            answer = f"{answer}\n\n{clarification}".strip()
    elif warning:
        status = "citation_warning" if not citations else "answered"
    else:
        status = "answered"

    return _result(
        question=question, answer=answer, status=status, insufficient=insufficient,
        citations=citations, hits=hits, top_k=top_k, tax_period=tax_period,
        uploaded_hits=uploaded_hits, reference_hits=reference_hits,
        retrieval_error=retrieval_error, generation_mode=generation_mode,
        fallback_reason=fallback_reason, warning=warning, notice=notice,
    )

