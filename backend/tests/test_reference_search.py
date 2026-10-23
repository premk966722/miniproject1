import asyncio

from app.services import answer_service, reference_search


def test_reference_corpus_loads_curated_official_entries():
    assert reference_search.reference_entry_count() >= 15


def test_reference_search_returns_citable_official_passages():
    hits = reference_search.search_references(
        "new tax regime slabs for AY 2026-27", top_k=4
    )
    assert hits
    top = hits[0]
    assert top["chunk_id"].startswith("c_")
    assert top["source_type"] == "government_reference"
    assert top["official_domain_verified"] is True
    assert top["source_url"].startswith("https://")
    assert "new regime" in top["document_title"].casefold()
    assert top["section_reference"]


def test_reference_chunk_ids_are_stable_and_unique():
    first = reference_search.search_references("section 80C deduction", top_k=20)
    second = reference_search.search_references("section 80C deduction", top_k=20)
    assert [hit["chunk_id"] for hit in first] == [hit["chunk_id"] for hit in second]
    assert len({hit["chunk_id"] for hit in first}) == len(first)


def test_fusion_prefers_uploaded_evidence_on_exact_rank_tie():
    uploaded = [{
        "chunk_id": "c_uploaded", "document_id": "doc-1", "document_title": "Uploaded PDF",
        "text": "Uploaded evidence", "retrieval_methods": ["bm25", "semantic"], "rrf_score": 2 / 61,
    }]
    references = [{
        "chunk_id": "c_reference", "document_id": "reference:one", "document_title": "Official FAQ",
        "text": "Official evidence", "retrieval_methods": [], "rrf_score": 0.0,
    }]
    fused = answer_service._fuse_evidence(uploaded, references, top_k=2)
    assert [hit["chunk_id"] for hit in fused] == ["c_uploaded", "c_reference"]
    assert fused[0]["rrf_score"] >= fused[1]["rrf_score"]
    assert "official_reference" in fused[1]["retrieval_methods"]
    assert "bm25" in fused[0]["retrieval_methods"]


def test_answer_uses_official_references_when_no_documents_are_uploaded(monkeypatch):
    monkeypatch.setattr(answer_service, "hybrid_search", lambda query, top_k: [])

    async def fake_generate(question, hits, tax_period=None):
        assert hits and hits[0]["source_type"] == "government_reference"
        return {
            "answer": f"The applicable slabs are stated in the reference [{hits[0]['chunk_id']}].",
            "insufficient_evidence": False,
            "needs_clarification": False,
            "clarification_question": None,
        }

    monkeypatch.setattr(answer_service, "generate_answer", fake_generate)
    result = asyncio.run(answer_service.answer_question(
        "What are the new tax regime slabs for AY 2026-27?", "AY 2026-27", 5
    ))
    assert result["status"] == "answered"
    assert result["grounded"] is True
    assert result["citations"][0]["source_status"] == "government_origin_verified"
    assert result["retrieval_metadata"]["reference_retrieval_used"] is True
