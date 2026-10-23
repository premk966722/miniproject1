"""BM25 + dense-vector retrieval with reciprocal-rank fusion and source metadata filters."""
import json
import re
import hashlib
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import EMBEDDING_MODEL, RRF_K
from .embeddings import embed_texts
from .storage import list_chunks

SEED_PATH = Path(__file__).resolve().parents[1] / "data" / "sources.json"


class HybridRetriever:
    def __init__(self, seed_path: Path = SEED_PATH):
        self.seed = json.loads(seed_path.read_text(encoding="utf-8"))
        self._seed_vectors: dict[str, list[float]] = {}

    @staticmethod
    def tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _corpus(self) -> list[dict]:
        corpus = []
        for d in self.seed:
            stable_id = "c_" + hashlib.sha256(d["id"].encode()).hexdigest()[:16]
            corpus.append({"id": stable_id, "document_id": d["id"], "text": d["text"], "title": d["title"],
                "filename": d["title"], "source_url": d["url"], "source_type": "government_reference",
                "authority": d["authority"], "official_domain_verified": True, "page_start": None, "page_end": None,
                "section": d["reference"], "metadata": {"document_type": "government guidance", "year_range": None},
                "document_metadata": {"checked": d.get("checked")}, "embedding": None})
        corpus.extend(list_chunks())
        return corpus

    @staticmethod
    def _matches_filter(doc: dict, filters: dict | None) -> bool:
        if not filters:
            return True
        meta = {**(doc.get("document_metadata") or {}), **(doc.get("metadata") or {})}
        for key, value in filters.items():
            if value in (None, "", "all"):
                continue
            candidates = [str(meta.get(key, "")), str(meta.get("year_range", "")), str(meta.get("assessment_year", "")), str(meta.get("financial_year", ""))]
            if key == "source" and value == "official" and not doc.get("official_domain_verified"):
                return False
            if key == "source" and value == "uploaded" and doc.get("source_type") not in ("user_upload", "official_url_pdf"):
                return False
            if key not in ("source",) and not any(str(value).lower() in c.lower() for c in candidates):
                return False
        return True

    @staticmethod
    def _bm25(query: str, docs: list[dict]) -> list[tuple[int, float]]:
        qterms = set(HybridRetriever.tokens(query))
        tokenized = [HybridRetriever.tokens(f"{d['title']} {d.get('section') or ''} {d['text']}") for d in docs]
        n = len(docs)
        if not n:
            return []
        avgdl = sum(map(len, tokenized)) / n or 1
        ranked = []
        for i, words in enumerate(tokenized):
            dl = len(words)
            score = 0.0
            for term in qterms:
                tf = words.count(term)
                if not tf:
                    continue
                df = sum(term in set(other) for other in tokenized)
                idf = np.log(1 + (n - df + .5) / (df + .5))
                score += float(idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl)))
            if score > 0:
                ranked.append((i, score))
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _rrf(*rankings: list[tuple[int, float]], constant: int = RRF_K) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _) in enumerate(ranking, 1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (constant + rank)
        return fused

    async def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[dict]:
        docs = [d for d in self._corpus() if self._matches_filter(d, filters)]
        if not docs:
            return []
        lexical = self._bm25(query, docs)
        texts = [f"{d['title']} {d.get('section') or ''} {d['text']}" for d in docs]
        qvecs = await embed_texts([query])
        dense_scores: dict[int, float] = {}
        dense_rank: list[tuple[int, float]] = []
        if qvecs and qvecs[0]:
            qvec = np.asarray(qvecs[0], dtype=float)
            vectors: list[list[float] | None] = []
            to_encode = []
            encode_positions = []
            for i, doc in enumerate(docs):
                stored = doc.get("embedding")
                if stored and doc.get("metadata", {}).get("embedding_model") == EMBEDDING_MODEL and len(stored) == len(qvec):
                    vectors.append(stored)
                else:
                    vectors.append(None)
                    to_encode.append(texts[i])
                    encode_positions.append(i)
            if to_encode:
                encoded = await embed_texts(to_encode)
                if encoded:
                    for i, vec in zip(encode_positions, encoded):
                        vectors[i] = vec
            for i, vec in enumerate(vectors):
                if vec is not None and len(vec) == len(qvec):
                    dense_scores[i] = float(cosine_similarity(qvec.reshape(1, -1), np.asarray(vec).reshape(1, -1))[0, 0])
            dense_rank = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
            dense_rank = [(i, s) for i, s in dense_rank if s >= 0.18]
        else:
            # Operational fallback for a no-key/no-embedding demo. This sparse vector
            # fallback is disclosed in /health and is not represented as dense semantics.
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            matrix = vectorizer.fit_transform(texts)
            qv = vectorizer.transform([query])
            scores = cosine_similarity(qv, matrix).ravel()
            dense_rank = sorted([(i, float(s)) for i, s in enumerate(scores) if s >= 0.12], key=lambda x: x[1], reverse=True)
        fused = self._rrf(lexical, dense_rank)
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        output = []
        for idx, score in ranked:
            lexical_score = next((s for i, s in lexical if i == idx), 0.0)
            dense_score = dense_scores.get(idx, 0.0)
            # A result must show actual relevance in at least one retriever.
            if lexical_score <= 0 and dense_score < 0.25:
                continue
            doc = docs[idx]
            output.append({**doc, "score": float(score), "lexical_score": float(lexical_score), "dense_score": float(dense_score)})
            if len(output) >= k:
                break
        # Remove near-identical overlapping chunks from the same page range.
        unique = []
        seen = set()
        for doc in output:
            key = (doc.get("document_id"), doc.get("page_start"), doc.get("page_end"), re.sub(r"\W+", " ", doc["text"].lower())[:180])
            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)
        return unique


retriever = HybridRetriever()
