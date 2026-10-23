"""Configurable dense embedding providers. Vectors are stored with their model name/dimension."""
import httpx

from .config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_PROVIDER, OLLAMA_BASE_URL


async def embed_texts(texts: list[str]) -> list[list[float]] | None:
    if not texts or EMBEDDING_PROVIDER == "none":
        return None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(35, connect=5)) as client:
            if EMBEDDING_PROVIDER in ("openai", "openai-compatible"):
                if not EMBEDDING_API_KEY:
                    return None
                res = await client.post(f"{EMBEDDING_BASE_URL}/embeddings", headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"}, json={"model": EMBEDDING_MODEL, "input": texts})
                res.raise_for_status()
                items = sorted(res.json()["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]
            if EMBEDDING_PROVIDER == "ollama":
                res = await client.post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": EMBEDDING_MODEL, "input": texts})
                res.raise_for_status()
                return res.json()["embeddings"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return None
    return None
