## Swiggy Annual Report RAG – ML Intern Assignment

This project is a **Retrieval-Augmented Generation (RAG)** application built on top of the **Swiggy Annual Report FY 2023–24**. It lets you ask natural‑language questions and get grounded, non‑hallucinated answers strictly from the report.

The app consists of:

- **Backend**: Python + FastAPI + LangChain + FAISS
- **Frontend**: A beautiful, glassmorphism‑style single‑page chat UI (vanilla HTML/CSS/JS)

---

### 1. Swiggy Annual Report Source

- **Document**: Swiggy Annual Report FY 2023–24  
- **Format**: PDF  
- **Local file used in this project**: the user‑provided PDF at
  `c:\Users\Arjun\AppData\Roaming\Cursor\User\workspaceStorage\54687ddf8445e44e3e634479567ba45d\pdfs\884561ce-b462-4f2a-b081-f767686e5af9\Annual-Report-FY-2023-24 (3)_compressed (1).pdf`

> **Note**: When you submit this assignment, add the official public URL of the Swiggy FY 2023–24 Annual Report here (from Swiggy’s investor relations / financials page).

---

### 2. Project Structure

- **`backend/`**
  - `app.py` – FastAPI app exposing `/query` and `/health` endpoints; ensures the vector index exists at startup.
  - `rag_pipeline.py` – Core RAG logic: loads the PDF, chunks text, creates embeddings, builds/loads FAISS, and runs LLM‑powered QA.
  - `ingest.py` – One‑time ingestion script to build the FAISS index from the PDF.
- **`frontend/`**
  - `index.html` – Chat‑style interface for asking questions.
  - `style.css` – Modern, glassmorphism‑inspired styling.
  - `app.js` – Calls the FastAPI backend and renders messages + supporting context.
- **`requirements.txt`** – Python dependencies for the backend.

The FAISS index and metadata are stored under `storage/` (created automatically).

---

### 3. Backend – Setup & Run

#### 3.1. Create a virtual environment

```bash
cd swiggy
python -m venv .venv
.venv\Scripts\activate  # on Windows PowerShell
```

#### 3.2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3.3. Configure environment variables

Create a `.env` file in the project root (same folder as `requirements.txt`) with:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
# Optional override if your PDF is in a different location:
# SWIGGY_REPORT_PATH=C:\path\to\Annual-Report-FY-2023-24.pdf
```

By default, the code uses the user‑provided PDF path shown in section 1. If your report is elsewhere, set `SWIGGY_REPORT_PATH`.

#### 3.4. Build the vector index

This step loads the Swiggy report, splits it into chunks, embeds them, and saves a FAISS index to disk:

```bash
python -m backend.ingest
```

You only need to re‑run this if you change the PDF or chunking parameters.

#### 3.5. Run the FastAPI server

```bash
uvicorn backend.app:app --reload --port 8000
```

A health check is available at `http://localhost:8000/health`.

---

### 4. Frontend – Beautiful Q&A UI

The frontend is a single‑page, chat‑style interface with:

- **Glassmorphism card** inside a dark gradient background.
- **Floating gradient accents** and soft shadows.
- **Chat bubbles** for user and assistant, with avatars.
- **Context pills** showing which pages of the report were used.
- **Keyboard‑friendly input** (`Enter` to send, `Shift+Enter` for new line).

#### 4.1. Open the UI

1. Make sure the backend is running on `http://localhost:8000`.
2. Open `frontend/index.html` in your browser (double‑click it or use “Open with browser”).

The page will connect to the backend at `http://localhost:8000/query` and you can start asking questions.

---

### 5. RAG Design Details

- **Document Processing**
  - Uses `PyPDFLoader` from `langchain_community` to load the Swiggy report.
  - Splits text with `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`) for semantically meaningful chunks.
- **Embeddings & Vector Store**
  - Uses Gemini embeddings via `GoogleGenerativeAIEmbeddings` (`text-embedding-004`).
  - Stores vectors in a local **FAISS** index under `storage/swiggy_faiss`.
- **Retrieval-Augmented Generation**
  - Retrieves top‑k (k=5) most relevant chunks.
  - Passes them, along with the user question, into a Gemini chat model (`gemini-1.5-flash`) via a custom prompt.
  - The prompt **forces grounding**: if the answer is not in the context, the model must respond with “I don’t know based on the Swiggy Annual Report FY 2023–24.”
- **Interface**
  - Frontend sends POST requests with `{ "question": "..." }` to `/query`.
  - Backend responds with:
    - `answer`: final answer string.
    - `contexts`: list of snippets with page numbers and source metadata.
  - UI renders the answer and shows which pages were used as supporting evidence.

---

### 6. How This Meets the Assignment Requirements

- **Document Processing**
  - Loads the Swiggy Annual Report PDF from disk.
  - Splits into meaningful chunks and keeps metadata (page numbers, source).
- **Embedding & Vector Store**
  - Generates embeddings using an API‑based Gemini embedding model.
  - Stores embeddings in FAISS and supports semantic similarity search.
- **RAG**
  - Retrieves the most relevant chunks for each query.
  - Uses a Gemini LLM to generate answers strictly from retrieved context.
  - Explicitly avoids hallucinations via the prompt and recommended answer format.
- **Question Answering Interface**
  - Simple, **beautiful** chat UI (no framework required).
  - Shows both **final answer** and **supporting context** (page pills) for transparency.

---

### 7. Extending or Customizing

- Swap OpenAI with another LLM/embedding provider by editing `rag_pipeline.py`.
- Tune `chunk_size`, `chunk_overlap`, or `k` in retrieval to optimize quality.
- Enhance the UI further with charts or multi‑tab views (e.g., “Financials”, “Business Highlights”, “Risk Factors”) while reusing the same `/query` endpoint.

