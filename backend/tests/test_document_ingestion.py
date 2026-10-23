from io import BytesIO

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app import main, storage
from app.api import documents as documents_api
from app.services import ingestion


def make_pdf(pages: list[str], *, password: str | None = None) -> bytes:
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    out = BytesIO()
    options = {}
    if password:
        options = {"encryption": fitz.PDF_ENCRYPT_AES_256, "owner_pw": "owner", "user_pw": password}
    pdf.save(out, **options)
    pdf.close()
    return out.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "taxshield.sqlite3")
    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path / "private_uploads")
    main.initialize()
    with TestClient(main.app) as test_client:
        yield test_client


def test_extracts_metadata_and_preserves_page_numbers_and_empty_pages():
    result = ingestion.ingest_pdf_bytes(make_pdf(["", "Section 10. Tax exemption." ]), "..\\tax.pdf")
    assert result["page_count"] == 2
    assert result["metadata"]["empty_pages"] == [1]
    assert result["metadata"]["extraction_limitations"]
    assert result["chunks"][0]["page_number"] == 2
    assert result["chunks"][0]["document_id"] == result["id"]
    assert result["chunks"][0]["filename"] == "tax.pdf"
    assert result["official_domain_verified"] is False


def test_chunks_have_character_overlap_and_are_bounded():
    text = " ".join(f"sentence{i}." for i in range(250))
    chunks = ingestion.split_page_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= ingestion.CHUNK_CHARS for chunk in chunks)
    assert any(len(set(chunks[i][-100:].split()) & set(chunks[i + 1][:100].split())) > 5
               for i in range(len(chunks) - 1))


@pytest.mark.parametrize("filename,content_type,data", [
    ("tax.txt", "text/plain", b"hello"),
    ("tax.pdf", "application/pdf", b"not a PDF"),
])
def test_rejects_bad_type_or_signature(client, filename, content_type, data):
    response = client.post("/api/documents/upload", files={"file": (filename, data, content_type)})
    assert response.status_code in (400, 415)
    assert client.get("/api/documents").json()["documents"] == []


def test_rejects_empty_corrupt_and_encrypted_pdfs(client):
    empty = client.post("/api/documents/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    corrupt = client.post("/api/documents/upload", files={"file": ("bad.pdf", b"%PDF-1.7\nnot really a pdf", "application/pdf")})
    encrypted = client.post("/api/documents/upload", files={"file": ("locked.pdf", make_pdf(["tax content"], password="secret"), "application/pdf")})
    assert empty.status_code == 400
    assert corrupt.status_code == 400
    assert encrypted.status_code == 400
    assert client.get("/api/documents").json()["documents"] == []


def test_upload_size_limit_is_enforced_while_reading(client, monkeypatch):
    monkeypatch.setattr(documents_api, "UPLOAD_LIMIT", 30)
    response = client.post("/api/documents/upload", files={"file": ("large.pdf", make_pdf(["more than thirty bytes"]), "application/pdf")})
    assert response.status_code == 413


def test_successful_upload_listing_and_persistent_sqlite_record(client, tmp_path):
    pdf_bytes = make_pdf(["Section 80C deduction rules apply." ])
    response = client.post(
        "/api/documents/upload",
        files={"file": ("tax guide.pdf", pdf_bytes, "application/pdf")},
        data={"source_url": "https://www.incometax.gov.in/example.pdf", "document_type": "Act", "tax_period": "2025-26"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ingestion_status"] == "completed"
    assert body["page_count"] == 1 and body["chunk_count"] == body["chunks_created"] == 1
    assert body["official_domain_verified"] is False
    assert body["source_url"] == "https://www.incometax.gov.in/example.pdf"
    assert client.get("/api/documents").json()["documents"][0]["id"] == body["id"]

    # Reopening the same SQLite file returns the saved document and chunk rows.
    main.initialize()
    assert storage.get_document(body["id"])["ingestion_status"] == "completed"
    chunks = storage.list_chunks()
    assert len(chunks) == 1 and chunks[0]["page_start"] == 1
    assert list((tmp_path / "private_uploads").glob("*.pdf"))


def test_root_and_health_remain_available(client):
    assert client.get("/").json() == {"status": "healthy", "application": "TAX-SHIELD"}
    health = client.get("/api/health").json()
    assert health["status"] == "healthy"
    assert health["application"] == "TAX-SHIELD"

