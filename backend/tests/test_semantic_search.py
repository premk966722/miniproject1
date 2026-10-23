import json
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, storage
from app.api import search as search_api
from app.services import semantic_search as semantic_module


class FakeModel:
    def __init__(self, dimension=4):
        self.dimension = dimension
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        output = []
        for text in texts:
            value = text.lower()
            if "investment" in value or "deduction" in value:
                vector = [1.0, 0.0, 0.0, 0.0]
            elif "passport" in value or "travel" in value:
                vector = [0.0, 1.0, 0.0, 0.0]
            elif "assessment" in value:
                vector = [0.0, 0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 0.0, 1.0]
            output.append(vector[:self.dimension])
        return np.asarray(output, dtype=np.float32)


class FakeIndex:
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


class FakeFaiss:
    @staticmethod
    def IndexFlatIP(dimension):
        return FakeIndex(dimension)

    @staticmethod
    def normalize_L2(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors /= norms

    @staticmethod
    def write_index(index, filename):
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump({"dimension": index.d, "vectors": index.vectors.tolist()}, handle)

    @staticmethod
    def read_index(filename):
        with open(filename, encoding="utf-8") as handle:
            data = json.load(handle)
        index = FakeIndex(data["dimension"])
        index.add(np.asarray(data["vectors"], dtype=np.float32))
        return index


@pytest.fixture
def database(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "semantic.sqlite3")
    storage.initialize()
    monkeypatch.setattr(semantic_module, "_faiss_module", lambda: FakeFaiss)
    return tmp_path


def add_chunk(document_id, text, *, title="Tax document", page=1):
    timestamp = datetime.now(timezone.utc).isoformat()
    chunk_id = f"c_{document_id}"
    storage.insert_document({
        "id": document_id, "title": title, "filename": f"{document_id}.pdf",
        "source_url": "https://www.incometax.gov.in/source.pdf", "source_type": "user_upload",
        "authority": "User-provided document", "official_domain_verified": False,
        "metadata": {}, "sha256": document_id, "page_count": page,
        "created_at": timestamp, "upload_timestamp": timestamp, "file_size": 512,
        "ingestion_status": "completed", "document_type": "Act", "tax_period": "2025-26",
    }, [{"chunk_id": chunk_id, "text": text, "page_number": page,
         "document_title": title, "filename": f"{document_id}.pdf",
         "document_type": "Act", "tax_period": "2025-26"}])
    return chunk_id


def make_engine(tmp_path, model=None, threshold=0.2):
    engine = semantic_module.SemanticSearchEngine(index_dir=tmp_path / "faiss", similarity_threshold=threshold)
    model = model or FakeModel()
    engine._load_model = lambda: model
    return engine, model


def test_empty_index_returns_empty_without_loading_model(database):
    engine, _ = make_engine(database)
    assert engine.search("investment deductions") == []
    assert engine._model is None


def test_semantic_ranking_and_metadata_preservation(database):
    relevant_id = add_chunk("investment", "Eligible investment contributions qualify for a tax deduction.", title="Finance Act", page=7)
    add_chunk("travel", "Passport and travel document renewal information.", title="Travel notice", page=2)
    engine, _ = make_engine(database, threshold=-0.1)
    results = engine.search("tax saving investment benefits", top_k=2)
    assert results[0]["chunk_id"] == relevant_id
    assert results[0]["document_id"] == "investment"
    assert results[0]["document_title"] == "Finance Act"
    assert results[0]["filename"] == "investment.pdf"
    assert results[0]["page_number"] == 7
    assert results[0]["text"] == "Eligible investment contributions qualify for a tax deduction."
    assert results[0]["source_url"] == "https://www.incometax.gov.in/source.pdf"
    assert results[0]["document_type"] == "Act"
    assert results[0]["tax_period"] == "2025-26"
    assert results[0]["retrieval_method"] == "semantic"
    assert results[0]["retrieval_score"] == pytest.approx(1.0)
    assert results[1]["retrieval_score"] < results[0]["retrieval_score"]


def test_dimension_validation_and_similarity_threshold(database):
    add_chunk("invest", "Investment deduction provisions.")
    class InconsistentModel(FakeModel):
        def encode(self, texts, **kwargs):
            self.calls.append(list(texts))
            return np.zeros((len(texts), 3), dtype=np.float32)

    bad_model = InconsistentModel()
    engine, _ = make_engine(database, model=bad_model)
    with pytest.raises(semantic_module.EmbeddingModelUnavailable, match="dimension"):
        engine.search("investment")

    class QueryMismatchModel(FakeModel):
        def encode(self, texts, **kwargs):
            if texts == ["mismatched query"]:
                return np.zeros((1, 3), dtype=np.float32)
            return super().encode(texts, **kwargs)

    query_engine, _ = make_engine(database, model=QueryMismatchModel())
    with pytest.raises(semantic_module.EmbeddingModelUnavailable, match="Query embedding dimension"):
        query_engine.search("mismatched query")

    # A documented cosine threshold excludes a valid but lower-similarity candidate.
    high_threshold, _ = make_engine(database, threshold=0.99)
    assert high_threshold.search("investment")
    # Nonmatching meaning is rejected below the configured cosine threshold.
    assert high_threshold.search("passport") == []


def test_index_persistence_reload_refresh_and_duplicate_prevention(database):
    first_id = add_chunk("investment", "Eligible investment tax deduction.")
    model = FakeModel()
    first, _ = make_engine(database, model=model)
    first.search("investment deduction")
    assert first._index.ntotal == 1
    assert first.index_path.is_file() and first.manifest_path.is_file()

    restored, _ = make_engine(database, model=model)
    restored.search("investment deduction")
    assert restored._index.ntotal == 1
    # Restore reuses document vectors and encodes only the query.
    assert len(model.calls) == 3
    assert model.calls[1] == ["investment deduction"]
    assert model.calls[2] == ["investment deduction"]

    add_chunk("new_deduction", "Investment deduction for eligible savings.")
    updated = restored.search("investment deduction", top_k=20)
    assert restored._index.ntotal == 2
    assert len({hit["chunk_id"] for hit in updated}) == len(updated)
    assert len(updated) == 2
    assert first_id in {hit["chunk_id"] for hit in updated}
    restored.search("investment deduction", top_k=20)
    assert restored._index.ntotal == 2


def test_corrupt_persisted_index_is_rebuilt(database):
    add_chunk("investment", "Eligible investment deduction.")
    engine, _ = make_engine(database)
    engine.search("investment")
    engine.index_path.write_bytes(b"corrupt")
    restored, _ = make_engine(database)
    assert restored.search("investment")
    assert restored._index.ntotal == 1


def test_semantic_api_validates_and_returns_actual_metadata(database, monkeypatch):
    chunk_id = add_chunk("api_doc", "Investment deductions under tax law.", page=5)
    engine, _ = make_engine(database)
    monkeypatch.setattr(search_api, "semantic_search", engine.search)
    with TestClient(main.app) as client:
        invalid = client.post("/api/search/semantic", json={"query": "  "})
        response = client.post("/api/search/semantic", json={"query": "tax saving investment", "top_k": 1})
        excessive = client.post("/api/search/semantic", json={"query": "investment", "top_k": 21})
    assert invalid.status_code == 422
    assert excessive.status_code == 422
    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_method"] == "semantic"
    assert payload["results"][0]["chunk_id"] == chunk_id
    assert payload["results"][0]["page_number"] == 5


def test_semantic_api_reports_model_failure_as_service_unavailable(database, monkeypatch):
    def fail_search(*args, **kwargs):
        raise semantic_module.EmbeddingModelUnavailable("model unavailable")

    monkeypatch.setattr(search_api, "semantic_search", fail_search)
    with TestClient(main.app) as client:
        response = client.post("/api/search/semantic", json={"query": "investment deductions"})
    assert response.status_code == 503
    assert "Semantic search is unavailable" in response.json()["detail"]
