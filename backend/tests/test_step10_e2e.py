"""Synthetic end-to-end wiring test; its PDF text is not legal evidence."""
from io import BytesIO

import numpy as np
import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app import main, storage
from app.api import documents as documents_api, search as search_api
from app.services import answer_service, hybrid_retriever, semantic_search


class FixtureModel:
    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append([1.0, 0.0] if any(term in lowered for term in
                             ("investment", "contribution", "80c", "deduction")) else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FixtureIndex:
    def __init__(self, dimension):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        self.vectors = np.vstack((self.vectors, vectors))

    def search(self, queries, count):
        scores = queries @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :count]
        return np.take_along_axis(scores, order, axis=1), order.astype(np.int64)


class FixtureFaiss:
    IndexFlatIP = FixtureIndex

    @staticmethod
    def normalize_L2(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors /= norms

    @staticmethod
    def write_index(index, filename):
        with open(filename, "wb") as handle:
            np.save(handle, index.vectors)

    @staticmethod
    def read_index(filename):
        vectors = np.load(filename)
        index = FixtureIndex(vectors.shape[1])
        index.add(vectors)
        return index


def make_fixture_pdf():
    pdf = fitz.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "QA TEST FIXTURE ONLY. Section 80C is a search token. Investment contributions and deduction are example terms, not legal guidance.")
    second = pdf.new_page()
    second.insert_text((72, 72), "QA TEST FIXTURE ONLY. Astronomy, telescopes and galaxy observations are unrelated sample text.")
    result = pdf.tobytes()
    pdf.close()
    return result


@pytest.fixture
def end_to_end_client(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "e2e.sqlite3")
    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path / "private_uploads")
    monkeypatch.setattr(semantic_search, "_faiss_module", lambda: FixtureFaiss)
    engine = semantic_search.SemanticSearchEngine(index_dir=tmp_path / "indexes", similarity_threshold=0.0)
    model = FixtureModel()
    engine._load_model = lambda: model
    monkeypatch.setattr(semantic_search, "semantic_engine", engine)
    monkeypatch.setattr(search_api, "semantic_search", engine.search)
    monkeypatch.setattr(hybrid_retriever.semantic_service, "semantic_search", engine.search)

    async def fake_llm(question, hits, tax_period=None):
        assert hits and hits[0]["chunk_id"]
        return {"answer": f"The synthetic passage contains these test terms [{hits[0]['chunk_id']}].",
                "insufficient_evidence": False, "needs_clarification": False, "clarification_question": None}

    monkeypatch.setattr(answer_service, "generate_answer", fake_llm)
    storage.initialize()
    with TestClient(main.app) as client:
        yield client


def test_pdf_to_hybrid_chat_uses_stored_evidence_and_real_chunk_citations(end_to_end_client):
    client = end_to_end_client
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["chat_retrieval_mode"] == "bm25+sentence-transformers+faiss+rrf"

    upload = client.post("/api/documents/upload", files={
        "file": ("synthetic_qa_fixture.pdf", make_fixture_pdf(), "application/pdf"),
    })
    assert upload.status_code == 200, upload.text
    document = upload.json()
    assert document["page_count"] == 2
    assert document["chunk_count"] == 2
    assert document["source_type"] == "user_upload"
    assert document["official_domain_verified"] is False

    listed = client.get("/api/documents").json()["documents"]
    assert len(listed) == 1 and listed[0]["id"] == document["id"]
    assert len(storage.list_chunks()) == 2

    query = "Section 80C investment contribution"
    keyword = client.post("/api/search/keyword", json={"query": query, "top_k": 5})
    semantic = client.post("/api/search/semantic", json={"query": "investment contribution deduction", "top_k": 5})
    hybrid = client.post("/api/search/hybrid", json={"query": query, "top_k": 5})
    assert keyword.status_code == semantic.status_code == hybrid.status_code == 200
    assert keyword.json()["results"][0]["page_number"] == 1
    assert semantic.json()["results"][0]["page_number"] == 1
    fused = hybrid.json()["results"][0]
    assert set(fused["retrieval_methods"]) == {"bm25", "semantic"}
    assert fused["rrf_score"] == pytest.approx(2 / 61)

    chat = client.post("/api/chat", json={
        "question": "What does the fixture say about Section 80C investment contributions?",
        "tax_period": "FY 2025-26", "top_k": 5,
    })
    assert chat.status_code == 200, chat.text
    answer = chat.json()
    assert answer["status"] == "answered"
    assert len(answer["citations"]) == 1
    citation = answer["citations"][0]
    assert citation["chunk_id"] == fused["chunk_id"]
    assert citation["document_id"] == document["id"]
    assert citation["page_number"] == fused["page_number"] == 1
    assert citation["evidence_excerpt"] == fused["text"]
    assert citation["source_status"] == "user_uploaded_unverified"
    assert answer["retrieval_metadata"]["requested_tax_period"] == "FY 2025-26"
