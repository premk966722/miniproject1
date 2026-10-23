from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import main, storage
from app.services.keyword_search import KeywordSearchEngine, tokenize


@pytest.fixture
def database(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "keyword.sqlite3")
    storage.initialize()
    return tmp_path


def add_document(document_id: str, title: str, text: str, *, page: int = 1,
                 source_url: str | None = None, document_type: str | None = None,
                 tax_period: str | None = None):
    timestamp = datetime.now(timezone.utc).isoformat()
    chunk_id = f"c_{document_id}"
    storage.insert_document({
        "id": document_id, "title": title, "filename": f"{document_id}.pdf",
        "source_url": source_url, "source_type": "user_upload", "authority": "User-provided document",
        "official_domain_verified": False, "metadata": {}, "sha256": document_id,
        "page_count": page, "created_at": timestamp, "upload_timestamp": timestamp,
        "file_size": 123, "ingestion_status": "completed", "document_type": document_type,
        "tax_period": tax_period,
    }, [{"chunk_id": chunk_id, "text": text, "page_number": page,
         "document_title": title, "filename": f"{document_id}.pdf", "source_url": source_url,
         "document_type": document_type, "tax_period": tax_period}])
    return chunk_id


def test_tokenizer_preserves_legal_numbers_and_years():
    assert tokenize("Section 80C, 10(13A), AY 2025-26") == ["section", "80c", "10", "13a", "ay", "2025", "26"]


def test_exact_keyword_and_section_search_rank_and_preserve_evidence(database):
    expected_id = add_document(
        "doc_80c", "Income Tax Act", "Deduction under section 80C includes eligible investments.",
        page=10, source_url="https://www.indiacode.nic.in/act.pdf", document_type="Act", tax_period="2025-26",
    )
    add_document("doc_return", "Return guide", "How to file an income tax return online.", page=2)
    add_document("doc_deduction", "Deduction guide", "Deduction rules for housing loan interest.", page=4)

    results = KeywordSearchEngine(top_k=5).search("deduction under section 80C")
    assert results[0]["chunk_id"] == expected_id
    assert results[0]["document_id"] == "doc_80c"
    assert results[0]["document_title"] == "Income Tax Act"
    assert results[0]["filename"] == "doc_80c.pdf"
    assert results[0]["page_number"] == 10
    assert results[0]["text"] == "Deduction under section 80C includes eligible investments."
    assert results[0]["source_url"] == "https://www.indiacode.nic.in/act.pdf"
    assert results[0]["document_type"] == "Act"
    assert results[0]["tax_period"] == "2025-26"
    assert results[0]["retrieval_method"] == "bm25"
    assert results[0]["retrieval_score"] > 0
    assert len({result["chunk_id"] for result in results}) == len(results)


def test_empty_database_and_no_matching_terms_return_empty_results(database):
    engine = KeywordSearchEngine()
    assert engine.search("Section 80C") == []
    add_document("doc_tds", "TDS document", "Tax deducted at source provisions.")
    assert engine.search("cryptocurrency mining tariff") == []


def test_top_k_is_configurable_and_index_refreshes_after_ingestion(database):
    add_document("doc_a", "Capital gains", "Capital gains are taxable on transfer.")
    engine = KeywordSearchEngine(top_k=4)
    assert engine.search("capital gains")
    add_document("doc_b", "Capital gains exemption", "Capital gains exemption under section 54.")
    assert len(engine.search("capital gains", top_k=1)) == 1
    assert len(engine.search("capital gains", top_k=20)) == 2


def test_keyword_api_validation_empty_store_and_response(database):
    with TestClient(main.app) as client:
        empty = client.post("/api/search/keyword", json={"query": "deduction", "top_k": 5})
        blank = client.post("/api/search/keyword", json={"query": "   "})
        too_many = client.post("/api/search/keyword", json={"query": "deduction", "top_k": 21})
        assert empty.status_code == 200
        assert empty.json() == {"query": "deduction", "retrieval_method": "bm25", "results": []}
        assert blank.status_code == 422
        assert too_many.status_code == 422

        chunk_id = add_document("doc_api", "Section 80C", "Deduction under section 80C for investments.", page=3)
        response = client.post("/api/search/keyword", json={"query": "section 80C", "top_k": 1})
        assert response.status_code == 200
        payload = response.json()
        assert payload["retrieval_method"] == "bm25"
        assert len(payload["results"]) == 1
        assert payload["results"][0]["chunk_id"] == chunk_id
        assert payload["results"][0]["page_number"] == 3
