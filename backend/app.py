import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .rag_pipeline import answer_query, build_vector_store


class QueryRequest(BaseModel):
    question: str


app = FastAPI(title="Swiggy Annual Report RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    
@app.on_event("startup")
async def ensure_index():
    """
    Ensure that the FAISS index is built when the server starts.
    If it already exists, the loader in rag_pipeline will just use it.
    """
    build_vector_store()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/query")
async def query_report(payload: QueryRequest):
    answer, contexts = answer_query(payload.question)
    return {
        "answer": answer,
        "contexts": contexts,
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

