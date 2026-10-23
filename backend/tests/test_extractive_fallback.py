import asyncio

from app.services import answer_service, extractive_answer


def reference_hit(**changes):
    value = {
        "document_id": "reference:ay-2026-27-new-regime-slabs",
        "chunk_id": "c_5d8976650324f62e",
        "document_title": "Income-tax slabs under the new regime for AY 2026-27",
        "text": (
            "For AY 2026-27 a resident individual below 60 years in the new tax regime "
            "pays nil tax up to Rs 4,00,000, 5% from Rs 4,00,001 to Rs 8,00,000, "
            "and 10% from Rs 8,00,001 to Rs 12,00,000."
        ),
        "source_url": "https://www.incometax.gov.in/",
        "document_type": "official_reference",
        "tax_period": "AY 2026-27",
        "section_reference": "Section 115BAC",
        "source_type": "government_reference",
        "authority": "Income Tax Department, Government of India",
        "official_domain_verified": True,
        "rrf_score": 1 / 61,
        "retrieval_methods": ["official_reference"],
    }
    value.update(changes)
    return value


def test_greeting_gets_a_reply_without_retrieval(monkeypatch):
    def should_not_retrieve(*args, **kwargs):
        raise AssertionError("greetings must not trigger retrieval")
    monkeypatch.setattr(answer_service, "hybrid_search", should_not_retrieve)
    result = asyncio.run(answer_service.answer_question("hello"))
    assert result["status"] == "answered"
    assert result["citations"] == []
    assert result["retrieval_metadata"]["generation_mode"] == "conversational"
    assert "TAX-SHIELD" in result["answer"]


def test_rate_limit_still_returns_citable_official_evidence(monkeypatch):
    monkeypatch.setattr(answer_service, "hybrid_search", lambda query, top_k: [])
    monkeypatch.setattr(answer_service, "search_references", lambda query, top_k: [reference_hit()])

    async def rate_limited(*args, **kwargs):
        raise answer_service.LLMRateLimitError("rate limited")
    monkeypatch.setattr(answer_service, "generate_answer", rate_limited)

    result = asyncio.run(answer_service.answer_question(
        "What are the new regime slabs for AY 2026-27?", "AY 2026-27", 6
    ))
    assert result["status"] == "answered"
    assert result["insufficient_evidence"] is False
    assert result["grounded"] is True
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_status"] == "government_origin_verified"
    assert "[c_5d8976650324f62e]" in result["answer"]
    assert result["retrieval_metadata"]["generation_mode"] == "extractive_fallback"
    assert result["retrieval_metadata"]["generation_fallback_reason"] == "rate_limited"


def test_cooldown_answers_next_question_without_calling_the_provider(monkeypatch):
    monkeypatch.setattr(answer_service, "hybrid_search", lambda query, top_k: [])
    monkeypatch.setattr(answer_service, "search_references", lambda query, top_k: [reference_hit()])

    async def rate_limited(*args, **kwargs):
        raise answer_service.LLMRateLimitError("rate limited")
    monkeypatch.setattr(answer_service, "generate_answer", rate_limited)
    asyncio.run(answer_service.answer_question("new regime slabs AY 2026-27", "AY 2026-27", 6))

    async def must_not_call(*args, **kwargs):
        raise AssertionError("provider must be skipped during cooldown")
    monkeypatch.setattr(answer_service, "generate_answer", must_not_call)

    result = asyncio.run(answer_service.answer_question("new regime slabs for AY 2026-27", "AY 2026-27", 6))
    assert result["status"] == "answered"
    assert result["citations"]
    assert result["retrieval_metadata"]["generation_fallback_reason"] == "cooldown"


def test_off_topic_question_gets_scope_guidance(monkeypatch):
    monkeypatch.setattr(answer_service, "hybrid_search", lambda query, top_k: [])
    monkeypatch.setattr(answer_service, "search_references", lambda query, top_k: [])
    result = asyncio.run(answer_service.answer_question("Who won the football world cup?"))
    assert result["status"] == "insufficient_evidence"
    assert result["retrieval_metadata"]["generation_mode"] == "guidance"
    assert "income-tax" in result["answer"]


def test_extractive_answer_selects_the_relevant_passage():
    hits = [
        reference_hit(document_title="Return-filing due dates", chunk_id="c_due",
                      text="The due date for filing a return for AY 2026-27 is 31 July 2026."),
        reference_hit(),
    ]
    built = extractive_answer.build_extractive_answer(
        "What are the new regime slabs for AY 2026-27?", hits
    )
    assert built is not None
    assert "c_5d8976650324f62e" in built["answer"]
    assert "Rs 4,00,000" in built["answer"]


def test_extractive_answer_does_not_add_a_loosely_related_second_topic():
    hits = [
        reference_hit(document_title="Salary: standard deduction and HRA exemption", chunk_id="c_hra",
                      section_reference="Sections 10(13A) and 16(ia)",
                      text="HRA exemption is the least of actual HRA, rent minus 10% of salary, and 50% of salary in metro cities."),
        reference_hit(document_title="Senior and super-senior citizen basic exemption limits", chunk_id="c_senior",
                      section_reference="Basic exemption limits",
                      text="A resident senior citizen has a basic exemption limit of Rs 3,00,000."),
    ]
    built = extractive_answer.build_extractive_answer(
        "How much HRA exemption can a salaried employee claim?", hits
    )
    assert built is not None
    assert "c_hra" in built["answer"]
    assert "c_senior" not in built["answer"]


def test_extractive_answer_matches_plural_title_terms():
    hits = [
        reference_hit(document_title="Deductions and exemptions available in the new tax regime",
                      chunk_id="c_new_deductions", section_reference="Section 115BAC",
                      text="Section 80C deductions are generally not available in the new tax regime."),
        reference_hit(document_title="Salary: standard deduction and HRA exemption", chunk_id="c_salary",
                      text="The standard deduction is available to salaried employees."),
    ]
    built = extractive_answer.build_extractive_answer(
        "Can I claim 80C deduction in the new regime?", hits
    )
    assert built is not None
    assert "c_new_deductions" in built["answer"]
    assert "c_salary" not in built["answer"]


def test_extractive_answer_declines_off_topic_questions():
    hits = [reference_hit(text="A resident individual in India must satisfy the residential status test.")]
    built = extractive_answer.build_extractive_answer("Who is the prime minister of India?", hits)
    assert built is None


def test_extractive_answer_does_not_treat_geography_as_capital_gains():
    hits = [reference_hit(
        document_title="Capital-gains rates for transfers on or after 23 July 2024",
        text="Long-term capital gains on listed equity are taxed at 12.5%.",
    )]
    built = extractive_answer.build_extractive_answer("What is the capital of France?", hits)
    assert built is None


def test_extractive_answer_keeps_capital_gains_in_scope():
    hits = [reference_hit(
        document_title="Capital-gains rates for transfers on or after 23 July 2024",
        text="Long-term capital gains on listed equity are taxed at 12.5%.",
    )]
    built = extractive_answer.build_extractive_answer("How are capital gains taxed?", hits)
    assert built is not None
    assert "12.5%" in built["answer"]
