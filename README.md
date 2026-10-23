# TAX-SHIELD

Evidence-grounded Indian income-tax document research. The app supports PDF upload and extraction, page-aware chunks, SQLite persistence, BM25 and Sentence Transformer + FAISS retrieval, reciprocal-rank fusion, citation ID validation, and a PDF research flow in the existing React UI. `/api/chat` uses the current hybrid retrieval pipeline; the backward-compatible `/api/ask` route keeps the earlier retriever.

The built-in catalogue is a small set of public government guidance references. It is not a complete or automatically updated copy of Indian tax law. Uploaded PDFs are labeled user-provided and are never marked as government-verified just because a user supplies a URL.

## Install (Windows PowerShell)

Use Python 3.12 and Node.js 20 or newer. The current VS Code environment reports Python 3.15.1 and has no project `.venv`; the FAISS CPU package currently has no compatible distribution for that interpreter. Install Python 3.12 for the full semantic retrieval runtime.

```powershell
cd "C:\Users\premk\OneDrive\Desktop\mini project 5"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
cd frontend
npm.cmd install
```

## Start the app

Terminal 1 (backend):

```powershell
cd "C:\Users\premk\OneDrive\Desktop\mini project 5\backend"
..\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd "C:\Users\premk\OneDrive\Desktop\mini project 5\frontend"
npm.cmd run dev
```

Open `http://localhost:5173`. API root: `http://127.0.0.1:8000/`; health: `http://127.0.0.1:8000/api/health`; docs: `http://127.0.0.1:8000/docs`. SQLite database is created at `backend/data/taxshield.sqlite3` on first API start. If port 8000 is occupied, select another port in the Uvicorn command and set `VITE_API_BASE_URL` in `frontend/.env` to match.

## Dense retrieval configuration

The app combines true dense vector similarity with BM25 and RRF when an embedding provider is configured. Without one, the health endpoint reports `bm25+tfidf-fallback+rrf`; TF-IDF is a sparse lexical fallback, not dense semantic search.

For Ollama, install/run Ollama and pull an embedding model, then edit `backend/.env`:

```dotenv
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

For an OpenAI-compatible embedding service:

```dotenv
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your-key
```

Embeddings are saved with model name and dimensions. Changing models causes old vectors to be ignored and re-embedded on retrieval; re-uploading PDFs after a model change is recommended for faster searches.

Answer generation supports Groq (`LLM_PROVIDER=groq`, `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_API_KEY=...`, `LLM_MODEL=...`), Ollama, and other OpenAI-compatible endpoints. Configure the provider in `backend/.env`; when evidence exists, an unset or unavailable provider returns a clear API error rather than silently substituting an extractive answer. With no retrieved evidence, the API returns an insufficient-evidence response without calling a provider. Secrets belong only in `backend/.env`, which is git-ignored. Groq's chat endpoint is OpenAI-compatible; check Groq's current model list before selecting `LLM_MODEL`.

## Upload and ask

Start both services, click **Add PDF** or **Choose PDF**, and select a text-searchable official Act, Rules, circular, notification, or government guidance PDF. Upload limit: 20 MB. Scanned image-only or encrypted PDFs need OCR/unlocking before upload. The success message reports extracted pages and chunks. Uploaded records are marked `USER UPLOAD · NOT VERIFIED`; a user-uploaded file is not certified based on its contents or filename.

Try: **“Which provisions in the uploaded document discuss the applicable assessment year?”** The answer includes the exact evidence excerpt, source title, page/section when extractable, and a stable citation ID. If no passage clears the retrieval evidence check, TAX-SHIELD returns an insufficient-evidence response instead of making up an answer.

## API

- `POST /api/documents/upload` — multipart PDF upload, optional user-supplied source URL.
- `GET /api/documents` — seeded references and ingested documents, with verification labels.
- `POST /api/chat` — question, optional tax period and top_k; hybrid-retrieves passages, requests an evidence-grounded LLM answer, and validates each citation against retrieved chunk IDs.
- `POST /api/ask` — backward-compatible Stage 1 alias.
- `GET /api/health` — corpus counts and active retrieval mode.
- `GET /api/sources` — built-in government reference catalogue.
- `POST /api/search/keyword` — BM25 search over ingested PDF passages.
- `POST /api/search/semantic` — Sentence Transformer and FAISS semantic search over ingested PDF passages.
- `POST /api/search/hybrid` — BM25 and semantic candidate ranks fused with Reciprocal Rank Fusion (RRF).

For `/api/chat`, try `{"question":"What deductions are available under Section 80C?","tax_period":"FY 2025-26","top_k":5}`. The response includes `status`, `insufficient_evidence`, `citations`, and `retrieval_metadata`. Citation excerpts, page numbers, URLs, and provenance labels come from retrieved records; uploaded PDFs remain labelled as unverified uploads. Retrieved passages are treated as untrusted data, and factual answer claims must cite supplied chunk IDs.

## Semantic retrieval

The default semantic checkpoint is `sentence-transformers/all-MiniLM-L6-v2`. The first semantic search with uploaded chunks downloads the model from Hugging Face, so it can take time and needs internet access. Vectors are normalized and scored with cosine similarity; the default `SEMANTIC_SIMILARITY_THRESHOLD=0.20` excludes passages with lower cosine scores. The FAISS index and its chunk ID manifest are saved under `backend/data/index` and rebuilt when the SQLite chunk corpus changes.

Use `POST /api/search/semantic` in Swagger at `http://127.0.0.1:8000/docs`. Example query: `What tax deductions are available for eligible investments?`. Unit tests use deterministic mock encoders and do not download the model.

Hybrid retrieval requests `KEYWORD_CANDIDATES` BM25 hits and `SEMANTIC_CANDIDATES` semantic hits, then combines their one-based ranks with `1 / (RRF_K + rank)`. Raw BM25 and cosine scores are returned separately for inspection; they do not affect the fused rank.

Run tests from the backend folder:

```powershell
cd "C:\Users\premk\OneDrive\Desktop\mini project 5\backend"
..\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.pytest_tmp
```

Security boundaries include PDF extension/MIME/signature checks, a 20 MB cap, safe filenames, text-only extraction, HTTP(S)-only user URLs, a basic per-process rate limit, input length limits, bounded LLM timeouts, no secret exposure in errors, and a system prompt that treats retrieved documents as untrusted quoted evidence.

## Frontend (React + TypeScript)

The Vite frontend reads its API address from `frontend/.env` (`VITE_API_BASE_URL`). Copy `frontend/.env.example` to `.env` if the backend is not running at `http://127.0.0.1:8000`. The interface keeps conversation history only for the current browser session, lists actual ingested PDFs, and displays citation passages and provenance returned by `/api/chat`.

Start the backend in Terminal 1:

```powershell
cd "C:\Users\premk\OneDrive\Desktop\mini project 5\backend"
..\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in Terminal 2:

```powershell
cd "C:\Users\premk\OneDrive\Desktop\mini project 5\frontend"
npm.cmd install
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
npm.cmd run dev
```

Open `http://localhost:5173`. Use **Add PDF** to upload a searchable tax document (maximum 20 MB), then use **Document library** to review its title, filename, pages, chunks, ingestion status, and provenance. Uploading does not certify government origin. Ask a question and open the evidence panel to inspect retrieved passages; source links open only for absolute HTTP or HTTPS URLs. If retrieval or answer generation fails, the UI shows the backend error and does not invent a response.

Frontend checks:

```powershell
cd "C:\Users\premk\OneDrive\Desktop\mini project 5\frontend"
npm.cmd test
npm.cmd run build
```

## Step 10 QA and demo

Automated checks cover backend services and API routes plus frontend components. The backend end-to-end test creates a two-page synthetic PDF marked **QA TEST FIXTURE ONLY**, uploads it through the API, checks SQLite persistence and page metadata, calls BM25, semantic, and hybrid routes, then checks that the answer citation resolves to the retrieved chunk. Semantic vectors/index operations and LLM generation are deterministic test doubles in that test. This verifies application wiring and citation ID validation; it does **not** measure real embedding quality, legal accuracy, or a live provider's behavior.

Run the checks:

```powershell
cd "C:\Users\premk\OneDrive\Desktop\mini project 5\backend"
..\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.pytest_tmp
cd ..\frontend
npm.cmd test -- --reporter=dot
npm.cmd run build
```

Demo sequence:

1. Configure a working Sentence Transformers model and FAISS installation, plus an available LLM provider in `backend/.env`; start both services using the commands above.
2. Open `http://localhost:5173`, add a searchable official Income Tax Department PDF, and confirm the library labels it as a user upload unless its origin was verified through the source catalogue.
3. Ask a question answerable from a passage in that PDF. Expand the evidence card and compare its title, page, and excerpt with the source PDF.
4. In Swagger (`http://127.0.0.1:8000/docs`), inspect `/api/search/keyword`, `/api/search/semantic`, `/api/search/hybrid`, and `/api/chat` for ranked evidence, RRF methods/scores, and citation chunk IDs.
5. Repeat with a question absent from the PDF and check that the response reports insufficient evidence.

Do not use the synthetic QA PDF as tax authority. There is no curated legal question/answer relevance benchmark in this repository, and no live model/provider benchmark is claimed. For a defensible quality evaluation, add a reviewed dataset of official document IDs, exact source URLs, effective dates, query text, relevant chunk/page IDs, and expected answer points. Measure Recall@k and MRR for retrieval, citation validity (cited chunk IDs present in retrieved evidence), and human-reviewed answer support/abstention separately. Record model versions, corpus snapshot, and evaluation date with every run. The only current evaluation result is the synthetic wiring test's assertions, not a legal RAG quality score.

## QA limitations observed in this environment

The workspace has no uploaded PDF corpus, and the active Python 3.15 runtime does not have `faiss` or `sentence_transformers` installed. Therefore real semantic retrieval and live `/api/chat` generation were not exercised here. Use Python 3.12, install `backend/requirements.txt`, configure model/provider credentials, and upload a real source PDF for that demonstration. `/api/health` reports dense-retrieval package and LLM configuration readiness without exposing secrets. Request bodies are bounded (20 MB PDF plus multipart overhead; 64 KB for other API requests), in addition to the upload service's PDF checks.
