import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app import ingestion, retrieval, storage
from app.answering import compose, validate_citations
from app.ingestion import PDFIngestionError, chunk_pages, ingest_pdf
from app.retrieval import HybridRetriever


def make_searchable_pdf(text="Section 10 exemption guidance effective from 1 April 2026"):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    safe_text = text.replace("(", "\\(").replace(")", "\\)").encode("ascii")
    stream.set_data(b"BT /F1 12 Tf 20 240 Td (" + safe_text + b") Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Title": "Income Tax Guidance 2025-26"})
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def test_chunking_retains_overlap_and_cross_page_metadata():
    pages = [{"page": 1, "text": " ".join(["section"] * 100)}, {"page": 2, "text": "second page wording"}]
    chunks = chunk_pages(pages, max_words=100, overlap=20, doc_id="d")
    assert len(chunks) == 2
    assert chunks[1]["page_start"] == 1
    assert chunks[1]["page_end"] == 2
    assert chunks[1]["text"].startswith("section section")


def test_upload_rejects_type_signature_and_size(monkeypatch):
    with pytest.raises(PDFIngestionError, match="Only PDF"):
        ingest_pdf("file.txt", "text/plain", b"hello")
    with pytest.raises(PDFIngestionError, match="signature"):
        ingest_pdf("file.pdf", "application/pdf", b"not a pdf")
    monkeypatch.setattr(ingestion, "MAX_PDF_BYTES", 8)
    with pytest.raises(PDFIngestionError, match="between"):
        ingest_pdf("file.pdf", "application/pdf", b"%PDF-123456789")


def test_extracts_pdf_metadata_pages_and_version(monkeypatch):
    class Metadata:
        title, author, creation_date = "Finance Act 2025-26", "CBDT", "D:20260101000000Z"
    class Page:
        def extract_text(self):
            return "Section 10 Exemption provisions effective from 1 April 2026. " + ("legal content " * 800)
    class Reader:
        is_encrypted = False
        metadata = Metadata()
        pages = [Page(), Page()]
    monkeypatch.setattr(ingestion, "PdfReader", lambda *args, **kwargs: Reader())
    doc, chunks = ingest_pdf("..\\tax.pdf", "application/pdf", b"%PDF-1.4 content")
    assert doc["filename"] == "tax.pdf"
    assert doc["metadata"]["year_range"] == "2025-26"
    assert doc["metadata"]["effective_date_text"].startswith("1 April 2026")
    assert doc["page_count"] == 2
    assert chunks and chunks[0]["page_start"] == 1
    assert all(c["id"].startswith("c_") for c in chunks)
    assert doc["official_domain_verified"] is False


def test_real_pdf_text_extraction():
    doc, chunks = ingest_pdf("guidance.pdf", "application/pdf", make_searchable_pdf())
    assert doc["page_count"] == 1
    assert doc["metadata"]["year_range"] == "2025-26"
    assert "Section 10 exemption guidance" in chunks[0]["text"]
    assert chunks[0]["page_start"] == 1


def test_bm25_keyword_retrieval_and_rrf():
    docs = [{"title": "Section 80C deductions", "text": "deduction investment section 80c"},
            {"title": "Return filing", "text": "income tax return filing"}]
    ranked = HybridRetriever._bm25("section 80C", docs)
    assert ranked[0][0] == 0
    fused = HybridRetriever._rrf([(1, 1), (0, .5)], [(0, .8), (1, .4)])
    assert fused[0] > 0 and fused[0] == fused[1]


def test_dense_semantic_retrieval_and_metadata_filter(monkeypatch, tmp_path):
    seed = tmp_path / "empty.json"
    seed.write_text("[]", encoding="utf-8")
    retriever_obj = HybridRetriever(seed)
    monkeypatch.setattr(retrieval, "list_chunks", lambda: [
        {"id": "c_aaaaaaaa", "document_id": "1", "text": "Relief for salaried workers", "title": "Allowance relief", "source_url": None,
         "source_type": "user_upload", "authority": "User-provided document", "official_domain_verified": False, "page_start": 3, "page_end": 3,
         "section": "Section 10", "metadata": {"embedding_model": "text-embedding-3-small", "financial_year": "2025-26"},
         "document_metadata": {}, "embedding": [1.0, 0.0]},
        {"id": "c_bbbbbbbb", "document_id": "2", "text": "Corporate tax audit", "title": "Audit rules", "source_url": None,
         "source_type": "user_upload", "authority": "User-provided document", "official_domain_verified": False, "page_start": 5, "page_end": 5,
         "section": None, "metadata": {"embedding_model": "text-embedding-3-small", "financial_year": "2024-25"},
         "document_metadata": {}, "embedding": [0.0, 1.0]},
    ])
    async def fake_embed(texts):
        if texts == ["worker relief"]:
            return [[1.0, 0.0]]
        return None
    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)
    results = asyncio.run(retriever_obj.search("worker relief", filters={"financial_year": "2025-26"}))
    assert len(results) == 1
    assert results[0]["id"] == "c_aaaaaaaa"


def test_citation_validation_rejects_fabricated_ids():
    hit = {"id": "c_0123456789abcdef", "title": "A", "text": "Evidence"}
    answer, cited, invalid = validate_citations("Verified [c_0123456789abcdef]. Made up [c_deadbeefdeadbeef].", [hit])
    assert "[c_deadbeefdeadbeef]" not in answer
    assert len(cited) == 1 and invalid == ["c_deadbeefdeadbeef"]


def test_insufficient_evidence_returns_no_citations():
    answer, citations, invalid = asyncio.run(compose("unrelated tax question", []))
    assert "not find enough relevant evidence" in answer
    assert citations == [] and invalid == []


def test_api_upload_and_validation_errors(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import main
    from app.api import documents as documents_api
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.sqlite3")
    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path / "uploads")
    main.initialize()
    client = TestClient(main.app)
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/chat", json={"question": "x"}).status_code == 422
    bad_type = client.post("/api/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert bad_type.status_code == 415
    bad_sig = client.post("/api/documents/upload", files={"file": ("fake.pdf", b"not pdf", "application/pdf")})
    assert bad_sig.status_code == 400
    uploaded = client.post("/api/documents/upload", files={"file": ("guidance.pdf", make_searchable_pdf(), "application/pdf")})
    assert uploaded.status_code == 200
    assert uploaded.json()["chunks_created"] == 1
    listing = client.get("/api/documents").json()["documents"]
    assert any(d["id"] == uploaded.json()["id"] and not d["official_domain_verified"] for d in listing)
    from app.services import answer_service
    rows = storage.list_chunks()
    chunk = rows[0]
    monkeypatch.setattr(answer_service, "hybrid_search", lambda query, top_k: [{
        "document_id": chunk["document_id"], "chunk_id": chunk["id"],
        "document_title": chunk["title"], "filename": chunk["filename"],
        "page_number": chunk["page_start"], "text": chunk["text"],
        "source_url": chunk["source_url"], "document_type": chunk["document_type"],
        "tax_period": chunk["tax_period"], "section_reference": chunk.get("section"),
        "source_type": "user_upload", "official_domain_verified": False,
        "rrf_score": 1 / 61, "retrieval_methods": ["bm25"],
    }])
    async def fake_generation(question, hits, tax_period=None):
        return {"answer": f"The passage describes the exemption [{chunk['id']}].",
                "insufficient_evidence": False, "needs_clarification": False,
                "clarification_question": None}
    monkeypatch.setattr(answer_service, "generate_answer", fake_generation)
    answer = client.post("/api/chat", json={"question": "What does Section 10 exemption guidance say?", "filters": {"source": "uploaded"}})
    assert answer.status_code == 200
    assert answer.json()["grounded"] is True
    assert answer.json()["citations"][0]["source_type"] == "user_upload"
