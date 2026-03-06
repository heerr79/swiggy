import os
from functools import lru_cache
from typing import List, Tuple

import requests
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# ----------------------------------
# PDF LOCATION
# ----------------------------------

def get_report_path() -> str:
    env_path = os.getenv("SWIGGY_REPORT_PATH")

    if env_path and os.path.exists(env_path):
        return env_path

    # fallback default
    return r"c:\Users\Arjun\AppData\Roaming\Cursor\User\workspaceStorage\54687ddf8445e44e3e634479567ba45d\pdfs\884561ce-b462-4f2a-b081-f767686e5af9\Annual-Report-FY-2023-24 (3)_compressed (1).pdf"


# ----------------------------------
# STORAGE DIRECTORY
# ----------------------------------

def get_storage_dir():
    storage = os.path.join(BASE_DIR, "storage")
    os.makedirs(storage, exist_ok=True)
    return storage


# ----------------------------------
# BUILD VECTOR STORE
# ----------------------------------

def build_vector_store():

    pdf_path = get_report_path()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    storage_dir = get_storage_dir()

    vector_store.save_local(
        os.path.join(storage_dir, "swiggy_faiss")
    )

    return vector_store


# ----------------------------------
# LOAD VECTOR STORE
# ----------------------------------

@lru_cache(maxsize=1)
def load_vector_store():

    storage_dir = get_storage_dir()
    index_path = os.path.join(storage_dir, "swiggy_faiss")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except:
        return build_vector_store()


# ----------------------------------
# GEMINI API CALL
# ----------------------------------

def call_gemini(prompt: str):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return "GOOGLE_API_KEY is missing. Please add it to your .env file."

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    headers = {
        "Content-Type": "application/json"
    }

    params = {
        "key": api_key
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0
        }
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if "text" in part:
                    return part["text"]

        return "No response generated."

    except Exception as e:
        return f"LLM error: {str(e)}"


# ----------------------------------
# MAIN QUERY FUNCTION
# ----------------------------------

def answer_query(question: str) -> Tuple[str, List[dict]]:

    vector_store = load_vector_store()

    docs = vector_store.similarity_search(question, k=5)

    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI assistant answering questions about Swiggy's Annual Report FY 2023-24.

Answer ONLY using the context below.

If the answer is not present, respond exactly:
"I don’t know based on the Swiggy Annual Report FY 2023-24."

Context:
{context_text}

Question:
{question}

Answer:
"""

    answer = call_gemini(prompt)

    contexts = []

    for doc in docs:

        metadata = doc.metadata or {}

        contexts.append({
            "page": metadata.get("page"),
            "source": metadata.get("source"),
            "snippet": doc.page_content
        })

    return answer, contexts