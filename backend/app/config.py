import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
APP_NAME = os.getenv("APP_NAME", "TAX-SHIELD")
APP_ENV = os.getenv("APP_ENV", "development")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")


def _project_path(value: str, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT / path


def _int_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {raw!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}; received {value}.")
    return value


UPLOAD_DIR = _project_path(os.getenv("UPLOAD_DIR", ""), DATA_DIR / "uploads")
INDEX_DIR = _project_path(os.getenv("INDEX_DIR", ""), DATA_DIR / "index")
TOP_K = _int_setting("TOP_K", 6, 1, 20)
RRF_K = _int_setting("RRF_K", 60, 1, 100000)
KEYWORD_CANDIDATES = _int_setting("KEYWORD_CANDIDATES", 20, 1, 100)
SEMANTIC_CANDIDATES = _int_setting("SEMANTIC_CANDIDATES", 20, 1, 100)
DB_PATH = Path(os.getenv("TAXSHIELD_DB", str(DATA_DIR / "taxshield.sqlite3")))
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_MB", "20")) * 1024 * 1024
MAX_CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "850"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "120"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "none").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
_embedding_model_setting = os.getenv("EMBEDDING_MODEL", "")
_remote_embedding_models = ("text-embedding-", "nomic-embed-text")
SEMANTIC_EMBEDDING_MODEL = (
    os.getenv("SEMANTIC_EMBEDDING_MODEL")
    or (_embedding_model_setting if _embedding_model_setting and not _embedding_model_setting.startswith(_remote_embedding_models)
        else "sentence-transformers/all-MiniLM-L6-v2")
)
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.20"))
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai-compatible").lower()
_default_llm_base_url = "https://api.groq.com/openai/v1" if LLM_PROVIDER == "groq" else "https://api.openai.com/v1"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _default_llm_base_url).rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# Optional lower-cost model used once if the primary model is rate limited.
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "")
# Keep provider waits below the browser's request timeout so the deterministic
# evidence fallback can still return a response.
LLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "20"))
