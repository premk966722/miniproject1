"""TAX-SHIELD FastAPI application entry point.

Run from the backend directory with:
    python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.router import router as api_router
from .api.documents import router as documents_router
from .api.search import router as search_router
from .config import APP_ENV, APP_NAME, FRONTEND_ORIGIN, INDEX_DIR, MAX_PDF_BYTES, UPLOAD_DIR
from .storage import initialize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("taxshield.api")
_requests: dict[str, deque] = defaultdict(deque)


class RequestBodyLimitMiddleware:
    """Bound API request bodies before multipart/JSON parsing consumes them."""

    def __init__(self, app, upload_limit: int, api_limit: int = 64 * 1024):
        self.app = app
        self.upload_limit = upload_limit + 1024 * 1024  # multipart headers and boundaries
        self.api_limit = api_limit

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        is_upload = scope.get("path") == "/api/documents/upload"
        limit = self.upload_limit if is_upload else self.api_limit
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                if int(raw_length) > limit:
                    await self._too_large(send, is_upload)
                    return
            except ValueError:
                await self._bad_length(send)
                return

        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise _RequestBodyTooLarge(is_upload)
            return message

        response_started = False

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge as exc:
            if not response_started:
                await self._too_large(send, exc.is_upload)

    @staticmethod
    async def _respond(send, status: int, message: str):
        body = ('{"detail":"' + message.replace('"', '\\"') + '"}').encode("utf-8")
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode("ascii"))]})
        await send({"type": "http.response.body", "body": body})

    async def _too_large(self, send, is_upload: bool):
        message = "PDF uploads are limited to 20 MB." if is_upload else "API request body exceeds the 64 KB limit."
        await self._respond(send, 413, message)

    async def _bad_length(self, send):
        await self._respond(send, 400, "Invalid Content-Length header.")


class _RequestBodyTooLarge(Exception):
    def __init__(self, is_upload: bool):
        self.is_upload = is_upload


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Prepare local data paths and fail visibly if persistent storage cannot start."""
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        initialize()
        log.info("%s started in %s mode", APP_NAME, APP_ENV)
    except Exception as exc:
        log.exception("TAX-SHIELD startup initialization failed")
        raise RuntimeError("TAX-SHIELD could not initialize its data store.") from exc
    yield
    log.info("%s stopped", APP_NAME)


app = FastAPI(
    title=APP_NAME,
    version="0.3.0",
    description="Evidence-grounded Indian income-tax research with source-linked responses.",
    summary="TAX-SHIELD API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys([FRONTEND_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173",
                                      "http://localhost:5174", "http://127.0.0.1:5174"])),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.add_middleware(RequestBodyLimitMiddleware, upload_limit=MAX_PDF_BYTES)


@app.middleware("http")
async def basic_rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        if request.url.path == "/api/documents/upload":
            try:
                if int(request.headers.get("content-length", "0")) > MAX_PDF_BYTES + 1024 * 1024:
                    return JSONResponse(status_code=413, content={"detail": f"PDF uploads are limited to {MAX_PDF_BYTES // (1024 * 1024)} MB."})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        for stale_ip, stale_history in list(_requests.items()):
            while stale_history and now - stale_history[0] > 60:
                stale_history.popleft()
            if not stale_history:
                _requests.pop(stale_ip, None)
        if ip not in _requests and len(_requests) >= 4096:
            _requests.pop(next(iter(_requests)), None)
        history = _requests[ip]
        if len(history) >= 60:
            return JSONResponse(status_code=429, content={"detail": "Request limit reached. Please wait one minute and try again."})
        history.append(now)
    return await call_next(request)


@app.get("/", tags=["System"], summary="TAX-SHIELD API root")
def root():
    return {"status": "healthy", "application": APP_NAME}


app.include_router(api_router)
app.include_router(documents_router)
app.include_router(search_router)
