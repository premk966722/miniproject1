"""Deterministic answers built directly from retrieved evidence.

The primary answer path uses the configured LLM.  That path can be unavailable
(rate limits, timeouts, provider outages) or can return a summary whose citation
markers do not validate.  In those cases we still need to respond to the user
instead of surfacing an error or an empty "insufficient evidence" state.

This module turns the highest-ranked retrieved passages into a short, readable
extractive answer with real ``[chunk_id]`` markers.  It never invents facts: the
text is quoted from the retrieved passages and every paragraph is attributable
to a stored source.
"""
import math
import re

from .keyword_search import tokenize

MAX_PASSAGES = 2
MAX_EXCERPT_CHARS = 760

# Tax-year qualifiers are useful in body text but should not decide which topic
# a passage covers.  Otherwise a generic AY 2026-27 slab page can outrank the
# specific entry a question is asking about.
_PERIOD_TERMS = {"ay", "fy", "year", "2024", "2025", "2026", "2027", "24", "25", "26", "27"}


def _stem(token: str) -> str:
    """Light plural normalization so 'deduction' matches 'deductions'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("es"):
        base = token[:-2]
        if base.endswith(("s", "x", "z", "ch", "sh")):
            return base
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {_stem(token) for token in tokenize(text)}


# Query words that add no retrieval signal on their own.
_STOPWORDS = {_stem(word) for word in {
    "a", "an", "and", "applies", "apply", "are", "as", "at", "be", "been", "but", "by", "can",
    "claim", "could", "did", "do", "does", "doing", "example", "for", "from", "get", "give",
    "had", "has", "have", "help", "how", "i", "if", "in", "into", "is", "it", "its", "know",
    "make", "many", "may", "me", "mean", "means", "much", "my", "need", "of", "on", "or",
    "our", "please", "say", "should", "so", "take", "tell", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "under", "us", "use",
    "want", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "work", "would", "you", "your",
}}

# A broad set of income-tax vocabulary used to decide whether a question is in
# scope for an evidence extract.  This prevents off-topic questions from being
# answered with unrelated tax passages.
_TAX_HINTS = {_stem(word) for word in {
    "80c", "80ccd", "80d", "80tta", "87a", "115bac", "10iea", "24b", "44ad", "44ada",
    "234a", "234b", "234c", "234f", "139", "143", "148", "194", "195", "206",
    "54ec", "54f", "80ccc", "80dd", "80ddb", "80e", "80ee", "80eea", "80g", "80gg", "80u",
    "act", "advance", "allowance", "assessment", "audit", "ay", "bonus", "business",
    "cess", "challan", "compensation", "crore", "crores", "ctc", "deduction", "deductions",
    "deduct", "depreciation", "dividend", "donation", "due", "elss", "exemption", "exemptions",
    "filing", "financial", "form", "fy", "gain", "gains", "hra", "income", "insurance",
    "interest", "invest", "investment", "itr", "lakh", "lakhs", "late", "liability", "loan",
    "loss", "mediclaim", "nps", "pan", "pay", "paycheck", "penalty", "perquisite", "ppf",
    "presumptive", "professional", "property", "rebate", "refund", "regime", "rent",
    "residential", "retirement", "return", "revenue", "rupee", "rupees", "salary", "saving",
    "savings", "section", "slab", "slabs", "standard", "surcharge", "tax", "taxable", "taxation",
    "taxed", "taxes", "taxing", "tcs", "tds", "year",
}}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_SPACE = re.compile(r"\s+")


def _meaningful_terms(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in _STOPWORDS and len(token) > 1}


def _term_weights(hits: list[dict], terms: set[str]) -> dict[str, tuple[float, float]]:
    """Return (title weight, body weight) for each query term.

    Titles are the strongest topical signal, and a term that appears in only a
    few titles should dominate a generic word such as "tax".  Body text uses a
    gentler inverse-document-frequency weight so a long passage cannot win just
    by repeating common words.
    """
    total = max(1, len(hits))
    body_counts = {term: 0 for term in terms}
    title_counts = {term: 0 for term in terms}
    for hit in hits:
        title = _tokens(str(hit.get("document_title") or ""))
        body = _tokens(" ".join(str(hit.get(key) or "") for key in
                                ("document_title", "section_reference", "text")))
        for term in terms:
            if term in title:
                title_counts[term] += 1
            if term in body:
                body_counts[term] += 1
    return {
        term: (
            0.0 if term in _PERIOD_TERMS
            else (total / title_counts[term]) ** 2 if title_counts[term] else 0.0,
            1.0 + math.log((total + 1) / (body_counts[term] + 0.5)),
        )
        for term in terms
    }


def _hit_score(hit: dict, weights: dict[str, tuple[float, float]]) -> float:
    """Rank hits for extraction using title, section, and passage overlap."""
    title = _tokens(str(hit.get("document_title") or ""))
    section = _tokens(str(hit.get("section_reference") or ""))
    text = _tokens(str(hit.get("text") or ""))
    return (
        3 * sum(weights.get(token, (0.0, 0.0))[0] for token in title)
        + sum(weights.get(token, (0.0, 0.0))[1] for token in section)
        + sum(weights.get(token, (0.0, 0.0))[1] for token in text)
    )


def _best_excerpt(text: str, terms: set[str], limit: int = MAX_EXCERPT_CHARS) -> str:
    """Return the most query-relevant window of a passage."""
    clean = _SPACE.sub(" ", str(text or "")).strip()
    if not clean:
        return ""
    if len(clean) <= limit:
        return clean
    sentences = [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(clean) if sentence.strip()]
    if not sentences:
        return clean[:limit].rstrip() + "…"
    scores = [len(terms & _tokens(sentence)) for sentence in sentences]
    best = max(range(len(sentences)), key=lambda index: (scores[index], -index))
    parts = [sentences[best]]
    length = len(parts[0])
    if best > 0 and length + len(sentences[best - 1]) + 1 <= limit:
        parts.insert(0, sentences[best - 1])
        length += len(sentences[best - 1]) + 1
    index = best + 1
    while index < len(sentences) and length + len(sentences[index]) + 1 <= limit:
        parts.append(sentences[index])
        length += len(sentences[index]) + 1
        index += 1
    excerpt = " ".join(parts)
    return excerpt[:limit].rstrip() + ("…" if len(excerpt) > limit else "")


def _passage_label(hit: dict) -> str:
    title = str(hit.get("document_title") or "").strip() or "Official reference"
    section = str(hit.get("section_reference") or "").strip()
    period = str(hit.get("tax_period") or "").strip()
    details = " · ".join(part for part in (section, period) if part)
    return f"{title} ({details})" if details else title


def _intro(reason: str) -> str:
    if reason == "invalid_citations":
        return ("I could not verify a synthesized answer against the retrieved passages, "
                "so here are the relevant source passages themselves:")
    if reason == "model_insufficient":
        return "Here are the closest passages from the indexed official sources:"
    if reason == "cooldown":
        return ("The live answer model is cooling down after hitting its rate limit, "
                "so here are the most relevant passages from the official sources:")
    if reason == "unavailable":
        return ("The live answer model is temporarily unavailable, "
                "so here are the most relevant passages from the official sources:")
    return "Here are the most relevant passages from the indexed official sources:"


def _fallback_note(reason: str) -> str:
    if reason == "invalid_citations":
        return ("A synthesized answer could not be tied to the retrieved passages; "
                "the source extracts above are shown instead.")
    if reason == "cooldown":
        return "The live model is rate limited; the source extracts above were returned directly."
    if reason == "model_insufficient":
        return "The model did not produce a citable summary; the closest official passages are shown above."
    return "The live model was unavailable; the source extracts above were returned directly."


def build_extractive_answer(question: str, hits: list[dict], reason: str = "unavailable") -> dict | None:
    """Build a citable extractive answer from retrieved passages.

    Returns ``None`` when the question is out of scope for the available
    evidence.  Callers should then return a short guidance message rather than
    an unrelated passage.
    """
    usable = [hit for hit in (hits or [])
              if isinstance(hit, dict) and str(hit.get("chunk_id") or "").strip()
              and str(hit.get("text") or "").strip()]
    if not usable:
        return None

    terms = _meaningful_terms(question)
    weights = _term_weights(usable, terms)
    ranked = sorted(
        ((_hit_score(hit, weights), position, hit) for position, hit in enumerate(usable)),
        key=lambda item: (-item[0], item[1]),
    )
    in_scope = bool(terms & _TAX_HINTS)
    if not in_scope:
        return None

    positive = [(score, hit) for score, _, hit in ranked if score > 0]
    if positive:
        threshold = max(1.0, positive[0][0] * 0.9)
        selected = [hit for score, hit in positive if score >= threshold][:MAX_PASSAGES]
    else:
        selected = usable[:MAX_PASSAGES] if in_scope else []
    if not selected:
        return None

    blocks = []
    for hit in selected:
        excerpt = _best_excerpt(hit.get("text"), terms)
        if not excerpt:
            continue
        blocks.append(f"{_passage_label(hit)}\n{excerpt} [{hit['chunk_id']}]")
    if not blocks:
        return None

    answer = f"{_intro(reason)}\n\n" + "\n\n".join(blocks)
    return {
        "answer": answer,
        "insufficient_evidence": False,
        "needs_clarification": False,
        "clarification_question": None,
        "fallback_note": _fallback_note(reason),
    }


def guidance_answer() -> str:
    """A helpful response when no indexed evidence can be matched."""
    return (
        "I could not find a complete, citable answer to this question in the indexed Indian "
        "income-tax sources. I can help with income-tax slabs, old versus new regime, deductions, "
        "ITR forms, advance tax, TDS, capital gains, salary and HRA, presumptive taxation, and "
        "filing due dates for AY 2026-27.\n\n"
        "Try rephrasing the question, or upload the relevant PDF in the Document library "
        "and ask again. You can also check the official Income Tax Department portal at "
        "https://www.incometax.gov.in/."
    )


def conversational_reply(question: str) -> str | None:
    """Handle greetings, thanks, and capability questions without an LLM call."""
    normalized = _SPACE.sub(" ", str(question or "")).strip().casefold().strip("!?., ")
    if not normalized:
        return None

    if re.match(r"^(hi|hello|hey|namaste|namaskar|yo|hola|good (morning|afternoon|evening))\b", normalized):
        return (
            "Namaste! I am TAX-SHIELD, your Indian income-tax research assistant. "
            "Ask me about slabs, the old versus new regime, deductions, ITR forms, advance tax, "
            "TDS, capital gains, HRA, presumptive taxation, or filing due dates for AY 2026-27."
        )
    if re.search(r"\b(thanks|thank you|thx|dhanyavad|dhanyawad)\b", normalized):
        return "You're welcome! Ask another income-tax question whenever you're ready."
    if normalized in {"who are you", "what are you", "what can you do", "what do you do",
                      "help", "how do i use this", "how does this work"}:
        return (
            "I am TAX-SHIELD, an evidence-first Indian income-tax research assistant. "
            "I retrieve passages from indexed official references and uploaded PDFs, then answer "
            "with citations you can inspect. Try asking about AY 2026-27 slabs, regime choice, "
            "80C deductions, ITR forms, advance tax, capital gains, or filing due dates."
        )
    return None
