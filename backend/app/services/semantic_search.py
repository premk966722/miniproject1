"""Sentence-Transformer embeddings and a persistent cosine-similarity FAISS index."""
import json
import logging
import threading
from pathlib import Path

import numpy as np

from .. import storage
from ..config import INDEX_DIR, SEMANTIC_EMBEDDING_MODEL, SEMANTIC_SIMILARITY_THRESHOLD, TOP_K

log = logging.getLogger("taxshield.semantic_search")
MAX_TOP_K = 20
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSearchError(RuntimeError):
    """Base error for semantic model or index failures."""


class EmbeddingModelUnavailable(SemanticSearchError):
    """The configured Sentence Transformer could not be loaded or used."""


def _faiss_module():
    try:
        import faiss
    except Exception as exc:
        raise EmbeddingModelUnavailable(
            "Semantic search dependencies are unavailable. Install sentence-transformers and faiss-cpu."
        ) from exc
    return faiss


def _metadata(row: dict, key: str):
    value = row.get(key)
    if value:
        return value
    for container_name in ("metadata", "document_metadata"):
        container = row.get(container_name)
        if isinstance(container, dict) and container.get(key):
            return container[key]
    return None


def _result(row: dict, score: float) -> dict:
    page_number = row.get("page_start")
    if page_number is None:
        page_number = row.get("page_end")
    return {
        "document_id": str(row.get("document_id") or ""),
        "chunk_id": str(row.get("id") or ""),
        "document_title": str(row.get("title") or row.get("document_title") or ""),
        "filename": str(row.get("filename") or ""),
        "page_number": page_number,
        "text": row.get("text") or "",
        "source_url": row.get("source_url"),
        "document_type": _metadata(row, "document_type"),
        "tax_period": _metadata(row, "tax_period"),
        "section_reference": row.get("section") or _metadata(row, "section_reference"),
        "source_type": row.get("source_type") or "unknown",
        "authority": row.get("authority"),
        "official_domain_verified": row.get("official_domain_verified") is True,
        "document_metadata": row.get("document_metadata") if isinstance(row.get("document_metadata"), dict) else {},
        "retrieval_method": "semantic",
        "retrieval_score": float(score),
    }


class SemanticSearchEngine:
    """Thread-safe semantic index that rebuilds from SQLite when its corpus changes."""

    def __init__(self, model_name: str = SEMANTIC_EMBEDDING_MODEL or DEFAULT_MODEL,
                 index_dir: Path = INDEX_DIR, top_k: int = TOP_K,
                 similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD):
        self.model_name = model_name or DEFAULT_MODEL
        self.index_dir = Path(index_dir)
        self.top_k = max(1, min(int(top_k), MAX_TOP_K))
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("Semantic cosine similarity threshold must be between -1 and 1.")
        self.similarity_threshold = float(similarity_threshold)
        self._lock = threading.RLock()
        self._model = None
        self._index = None
        self._version = None
        self._mapping: list[str] = []
        self._rows_by_id: dict[str, dict] = {}
        self._dimension: int | None = None

    @property
    def index_path(self) -> Path:
        return self.index_dir / "semantic.faiss"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "semantic_manifest.json"

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            # First use can download model files from Hugging Face and needs network access.
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            log.exception("Could not load semantic embedding model %s", self.model_name)
            raise EmbeddingModelUnavailable(
                f"Could not load semantic embedding model '{self.model_name}'. Check model access, internet, and available memory."
            ) from exc
        return self._model

    @staticmethod
    def _encode(model, texts: list[str]) -> np.ndarray:
        try:
            vectors = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
            vectors = np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            raise EmbeddingModelUnavailable("The semantic embedding model could not encode the text.") from exc
        if vectors.ndim == 1 and len(texts) == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
            raise EmbeddingModelUnavailable("The semantic model returned an invalid embedding shape.")
        if not np.isfinite(vectors).all():
            raise EmbeddingModelUnavailable("The semantic model returned non-finite embedding values.")
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def _load_persisted(self, faiss, version, rows_by_id: dict[str, dict], dimension: int) -> bool:
        if not self.index_path.is_file() or not self.manifest_path.is_file():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            expected_version = list(version)
            mapping = manifest["chunk_ids"]
            if (manifest.get("model_name") != self.model_name
                    or manifest.get("version") != expected_version
                    or manifest.get("dimension") != dimension
                    or not isinstance(mapping, list)
                    or len(mapping) != len(set(mapping))
                    or any(chunk_id not in rows_by_id for chunk_id in mapping)):
                return False
            index = faiss.read_index(str(self.index_path))
            if index.d != dimension or index.ntotal != len(mapping):
                return False
            self._index = index
            self._mapping = mapping
            self._dimension = dimension
            self._version = version
            return True
        except Exception:
            log.warning("Persisted semantic index is missing, outdated, or corrupt; rebuilding from SQLite.")
            return False

    def _persist(self, faiss, index, mapping: list[str], version, dimension: int) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_tmp = self.index_dir / "semantic.faiss.tmp"
        manifest_tmp = self.index_dir / "semantic_manifest.json.tmp"
        manifest = {"model_name": self.model_name, "dimension": dimension,
                    "version": list(version), "chunk_ids": mapping}
        try:
            faiss.write_index(index, str(index_tmp))
            manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            index_tmp.replace(self.index_path)
            manifest_tmp.replace(self.manifest_path)
        except Exception:
            for path in (index_tmp, manifest_tmp):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            # Keep search usable in memory; the next restart will rebuild if persistence failed.
            log.exception("Could not persist semantic index to %s", self.index_dir)

    def _ensure_index(self) -> None:
        version = storage.chunks_version()
        if version == self._version:
            return
        rows = []
        rows_by_id = {}
        for row in storage.list_chunks():
            text, chunk_id = row.get("text"), row.get("id")
            if not isinstance(text, str) or not text.strip() or not isinstance(chunk_id, str) or chunk_id in rows_by_id:
                continue
            rows.append(row)
            rows_by_id[chunk_id] = row
        if not rows:
            self._index = None
            self._mapping = []
            self._rows_by_id = {}
            self._dimension = None
            self._version = version
            return

        model = self._load_model()
        expected_dimension = model.get_sentence_embedding_dimension()
        if not isinstance(expected_dimension, int) or expected_dimension <= 0:
            raise EmbeddingModelUnavailable("The semantic model reported an invalid embedding dimension.")
        faiss = _faiss_module()
        self._rows_by_id = rows_by_id
        if self._load_persisted(faiss, version, rows_by_id, expected_dimension):
            return

        vectors = self._encode(model, [row["text"] for row in rows])
        if vectors.shape[1] != expected_dimension:
            raise EmbeddingModelUnavailable(
                f"Embedding dimension mismatch: model reports {expected_dimension}, encoder returned {vectors.shape[1]}."
            )
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(expected_dimension)
        index.add(vectors)
        mapping = [row["id"] for row in rows]
        self._index = index
        self._mapping = mapping
        self._dimension = expected_dimension
        self._version = version
        self._persist(faiss, index, mapping, version, expected_dimension)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 1200:
            raise ValueError("Query must contain 1 to 1200 non-whitespace characters.")
        limit = self.top_k if top_k is None else max(1, min(int(top_k), MAX_TOP_K))
        with self._lock:
            self._ensure_index()
            if self._index is None or not self._mapping:
                return []
            model = self._load_model()
            query_vector = self._encode(model, [query.strip()])
            if query_vector.shape[1] != self._dimension:
                raise EmbeddingModelUnavailable(
                    f"Query embedding dimension {query_vector.shape[1]} does not match index dimension {self._dimension}."
                )
            faiss = _faiss_module()
            faiss.normalize_L2(query_vector)
            scores, positions = self._index.search(query_vector, min(limit, len(self._mapping)))
            results = []
            seen = set()
            for score, position in zip(scores[0], positions[0]):
                position = int(position)
                score = float(score)
                if position < 0 or position >= len(self._mapping) or score < self.similarity_threshold:
                    continue
                chunk_id = self._mapping[position]
                if chunk_id in seen:
                    continue
                row = self._rows_by_id.get(chunk_id)
                if row is None:
                    continue
                seen.add(chunk_id)
                results.append(_result(row, score))
            return results


semantic_engine = SemanticSearchEngine(
    model_name=SEMANTIC_EMBEDDING_MODEL or DEFAULT_MODEL
)


def semantic_search(query: str, top_k: int = TOP_K) -> list[dict]:
    return semantic_engine.search(query, top_k=top_k)
