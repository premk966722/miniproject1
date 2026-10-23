from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..config import TOP_K


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1200)
    tax_year: str | None = Field(default=None, max_length=24)
    filters: dict[str, str] = Field(default_factory=dict, max_length=8)

    @field_validator("filters")
    @classmethod
    def supported_filters_only(cls, value):
        allowed = {"financial_year", "assessment_year", "applicable_period", "document_type", "source"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unsupported metadata filters: {', '.join(sorted(unknown))}")
        return value


class Citation(BaseModel):
    id: str
    title: str
    authority: str
    url: str | None = None
    reference: str | None = None
    excerpt: str
    score: float
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    source_type: str
    official_domain_verified: bool


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    grounded: bool
    notice: str
    citation_warning: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1200)
    tax_period: str | None = Field(default=None, max_length=64)
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    # Kept for the current frontend until its separate integration step.
    tax_year: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty or whitespace.")
        return value


class AnswerCitation(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: str
    document_title: str
    page_number: int | None = None
    section_reference: str | None = None
    source_url: str | None = None
    source_status: str
    evidence_excerpt: str
    # Compatibility aliases consumed by the current evidence cards.
    id: str
    title: str | None = None
    authority: str | None = None
    url: str | None = None
    reference: str | None = None
    excerpt: str
    score: float
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    source_type: str
    official_domain_verified: bool


class ChatResponse(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "insufficient_evidence", "needs_clarification", "citation_warning"]
    insufficient_evidence: bool
    citations: list[AnswerCitation]
    retrieval_metadata: dict
    # Compatibility fields for the existing frontend.
    grounded: bool
    confidence: float
    notice: str
    citation_warning: str | None = None


class UploadResponse(BaseModel):
    id: str
    title: str
    filename: str
    page_count: int
    chunks_created: int
    source_type: str
    official_domain_verified: bool
    metadata: dict


class IngestedDocument(BaseModel):
    id: str
    title: str
    filename: str
    file_size: int
    upload_timestamp: str
    page_count: int
    chunk_count: int
    ingestion_status: str
    source_url: str | None = None
    source_type: str
    official_domain_verified: bool
    document_type: str | None = None
    tax_period: str | None = None
    metadata: dict


class DocumentUploadResponse(IngestedDocument):
    extraction_limitations: list[str]
    chunks_created: int


class KeywordSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1200)
    top_k: int = Field(default=TOP_K, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query must not be empty or whitespace.")
        return value


class KeywordSearchResult(BaseModel):
    document_id: str
    chunk_id: str
    document_title: str
    filename: str
    page_number: int | None = None
    text: str
    source_url: str | None = None
    document_type: str | None = None
    tax_period: str | None = None
    retrieval_method: str
    retrieval_score: float


class KeywordSearchResponse(BaseModel):
    query: str
    retrieval_method: str
    results: list[KeywordSearchResult]


class HybridSearchResult(BaseModel):
    document_id: str
    chunk_id: str
    document_title: str
    filename: str
    page_number: int | None = None
    text: str
    source_url: str | None = None
    document_type: str | None = None
    tax_period: str | None = None
    section_reference: str | None = None
    source_type: str = "unknown"
    authority: str | None = None
    official_domain_verified: bool = False
    document_metadata: dict = Field(default_factory=dict)
    rrf_score: float
    bm25_score: float | None = None
    semantic_score: float | None = None
    retrieval_methods: list[str]


class HybridSearchResponse(BaseModel):
    query: str
    retrieval_method: str
    result_count: int
    results: list[HybridSearchResult]
