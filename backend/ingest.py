"""
One-time (or occasional) ingestion script.

Usage:
    python -m backend.ingest
"""

from .rag_pipeline import build_vector_store


def main():
    build_vector_store()


if __name__ == "__main__":
    main()

