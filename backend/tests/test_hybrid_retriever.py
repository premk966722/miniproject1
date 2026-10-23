import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.api import search as search_api
from app.services import hybrid_retriever as hybrid


def result(chunk_id, score, *, method="bm25", **overrides):
    values = {
        "document_id": f"doc_{chunk_id}", "chunk_id": chunk_id,
        "document_title": f"Title {chunk_id}", "filename": f"{chunk_id}.pdf",
        "page_number": 4, "text": f"Stored text for {chunk_id}",
        "source_url": None, "document_type": None, "tax_period": None,
        "retrieval_method": method, "retrieval_score": score,
    }
    values.update(overrides)
    return values


def test_rrf_uses_one_based_ranks_and_fuses_both_methods():
    keyword = [result("a", 90), result("b", 1)]
    semantic = [result("b", 0.99, method="semantic"), result("a", 0.20, method="semantic")]
    fused = hybrid.reciprocal_rank_fusion(keyword, semantic, rrf_k=60, top_k=5)
    assert [item["chunk_id"] for item in fused] == ["a", "b"]
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0]["bm25_score"] == 90
    assert fused[0]["semantic_score"] == pytest.approx(0.20)
    assert fused[0]["retrieval_methods"] == ["bm25", "semantic"]


def test_deduplicates_chunk_ids_and_preserves_actual_metadata():
    keyword = [result("a", 4, document_title="Tax Act", page_number=12,
                      source_url="https://www.indiacode.nic.in/a.pdf"),
               result("a", 3), result("b", 2)]
    semantic = [result("a", 0.8, method="semantic", tax_period="2025-26")]
    fused = hybrid.reciprocal_rank_fusion(keyword, semantic, rrf_k=10, top_k=20)
    by_id = {item["chunk_id"]: item for item in fused}
    assert len(fused) == 2
    assert by_id["a"]["rrf_score"] == pytest.approx(1 / 11 + 1 / 11)
    assert by_id["b"]["rrf_score"] == pytest.approx(1 / 12)
    assert by_id["a"]["document_title"] == "Tax Act"
    assert by_id["a"]["page_number"] == 12
    assert by_id["a"]["source_url"] == "https://www.indiacode.nic.in/a.pdf"
    assert by_id["a"]["tax_period"] == "2025-26"
    assert by_id["a"]["text"] == "Stored text for a"


@pytest.mark.parametrize(("keyword", "semantic", "expected"), [
    ([], [result("s", 0.8, method="semantic")], ["s"]),
    ([result("k", 2)], [], ["k"]),
    ([], [], []),
])
def test_empty_retrieval_side_is_handled(keyword, semantic, expected):
    fused = hybrid.reciprocal_rank_fusion(keyword, semantic, top_k=5)
    assert [item["chunk_id"] for item in fused] == expected


def test_hybrid_search_calls_existing_services_with_configured_candidate_counts(monkeypatch):
    calls = []

    def fake_keyword(query, top_k):
        calls.append(("bm25", query, top_k))
        return [result("same", 3)]

    def fake_semantic(query, top_k):
        calls.append(("semantic", query, top_k))
        return [result("same", 0.7, method="semantic")]

    monkeypatch.setattr(hybrid.keyword_service, "search_chunks", fake_keyword)
    monkeypatch.setattr(hybrid.semantic_service, "semantic_search", fake_semantic)
    monkeypatch.setattr(config, "KEYWORD_CANDIDATES", 13)
    monkeypatch.setattr(config, "SEMANTIC_CANDIDATES", 17)
    monkeypatch.setattr(config, "RRF_K", 25)

    hits = hybrid.hybrid_search("  section 80C  ", top_k=1)
    assert calls == [("bm25", "section 80C", 13), ("semantic", "section 80C", 17)]
    assert len(hits) == 1
    assert hits[0]["rrf_score"] == pytest.approx(2 / 26)


@pytest.mark.parametrize("query,top_k", [(" ", 5), ("deduction", 0), ("deduction", 21)])
def test_hybrid_search_rejects_invalid_arguments(query, top_k):
    with pytest.raises(ValueError):
        hybrid.hybrid_search(query, top_k=top_k)


def test_invalid_numeric_configuration_has_clear_error(monkeypatch):
    monkeypatch.setenv("KEYWORD_CANDIDATES", "0")
    with pytest.raises(ValueError, match="KEYWORD_CANDIDATES must be between 1 and 100"):
        config._int_setting("KEYWORD_CANDIDATES", 20, 1, 100)


def test_hybrid_api_response_empty_and_populated(monkeypatch):
    populated = [hybrid.reciprocal_rank_fusion([result("one", 2)], [], top_k=5)[0]]
    response_results = {"value": populated}
    monkeypatch.setattr(search_api, "hybrid_search", lambda query, top_k: response_results["value"])
    with TestClient(main.app) as client:
        response = client.post("/api/search/hybrid", json={"query": "investment deduction", "top_k": 5})
        invalid = client.post("/api/search/hybrid", json={"query": "", "top_k": 5})
        excessive = client.post("/api/search/hybrid", json={"query": "investment", "top_k": 21})
        response_results["value"] = []
        empty = client.post("/api/search/hybrid", json={"query": "investment", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "investment deduction"
    assert body["retrieval_method"] == "hybrid"
    assert body["result_count"] == 1
    assert body["results"][0]["chunk_id"] == "one"
    assert body["results"][0]["bm25_score"] == 2
    assert body["results"][0]["semantic_score"] is None
    assert invalid.status_code == 422
    assert excessive.status_code == 422
    assert empty.status_code == 200
    assert empty.json()["result_count"] == 0 and empty.json()["results"] == []


def test_hybrid_api_returns_service_unavailable_on_retrieval_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise hybrid.HybridRetrievalError("backend down")

    monkeypatch.setattr(search_api, "hybrid_search", fail)
    with TestClient(main.app) as client:
        response = client.post("/api/search/hybrid", json={"query": "investment deduction"})
    assert response.status_code == 503
