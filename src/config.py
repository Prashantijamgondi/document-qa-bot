from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
TEMP_DIR = BASE_DIR / ".tmp_uploads"
COLLECTION_NAME = "document_knowledge_base"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    embedding_model: str = "models/gemini-embedding-001"
    generation_model: str = "gemini-2.5-flash"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    min_query_length: int = 3


def get_settings() -> Settings:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file before running the app."
        )
    return Settings(gemini_api_key=api_key)