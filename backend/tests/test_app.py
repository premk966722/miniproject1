from fastapi.testclient import TestClient

from app.main import app


def test_application_initializes_and_root_is_healthy():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "application": "TAX-SHIELD"}


def test_health_response_structure():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["application"] == "TAX-SHIELD"
        assert "retrieval_mode" in body
        assert body["chat_retrieval_mode"] == "bm25+sentence-transformers+faiss+rrf"
        assert isinstance(body["chat_readiness"]["faiss_installed"], bool)
        assert isinstance(body["chat_readiness"]["llm_provider_configured"], bool)


def test_openapi_docs_are_available():
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200


def test_api_rejects_oversized_json_before_route_parsing():
    with TestClient(app) as client:
        response = client.post("/api/chat", content=b"x" * (64 * 1024 + 1),
                               headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert "64 KB" in response.json()["detail"]
