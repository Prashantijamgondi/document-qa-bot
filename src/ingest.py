from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from docx import Document
from pypdf import PdfReader
from tqdm import tqdm
from chromadb.config import Settings
from src.config import COLLECTION_NAME, DB_DIR, TEMP_DIR, get_settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def ensure_runtime_dirs() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_pdf_pages(file_path: str | Path) -> list[dict]:
    extracted_data: list[dict] = []
    file_path = Path(file_path)
    file_name = file_path.name

    reader = PdfReader(str(file_path))
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        clean = clean_text(text)
        if clean:
            extracted_data.append(
                {
                    "text": clean,
                    "metadata": {
                        "source": file_name,
                        "page": index + 1,
                        "file_type": "pdf",
                    },
                }
            )
    return extracted_data


def extract_docx_content(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    file_name = file_path.name
    document = Document(str(file_path))

    paragraphs = [clean_text(p.text) for p in document.paragraphs if p.text.strip()]
    joined_text = "\n".join(paragraphs).strip()
    if not joined_text:
        return []

    return [
        {
            "text": joined_text,
            "metadata": {
                "source": file_name,
                "page": 1,
                "file_type": "docx",
            },
        }
    ]


def extract_txt_content(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    file_name = file_path.name
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    clean = clean_text(text)
    if not clean:
        return []

    return [
        {
            "text": clean,
            "metadata": {
                "source": file_name,
                "page": 1,
                "file_type": "txt",
            },
        }
    ]


def extract_document(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_pages(file_path)
    if suffix == ".docx":
        return extract_docx_content(file_path)
    if suffix == ".txt":
        return extract_txt_content(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def smart_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, tuple[int, int]]]:
    chunks: list[tuple[str, tuple[int, int]]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        if end < text_length:
            split_candidates = [chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind("\n"), chunk.rfind(" ")]
            best_split = max(split_candidates)
            if best_split > chunk_size // 2:
                end = start + best_split + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append((chunk, (start, end)))

        if end >= text_length:
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def chunk_extracted_pages(
    pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[dict]:
    chunks: list[dict] = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        page_chunks = smart_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_index, (chunk_text, (start, end)) in enumerate(page_chunks, start=1):
            chunk_id = f"{metadata['source']}::p{metadata['page']}::c{chunk_index}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_range": f"{start}-{end}",
                    },
                }
            )

    return chunks


def get_embedding_function():
    settings = get_settings()
    return GoogleGenerativeAiEmbeddingFunction(
        api_key=settings.gemini_api_key,
        model_name=settings.embedding_model,
    )


def get_collection():
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    embedding_fn = get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def save_to_vector_db(chunks: list[dict]) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    existing = set(collection.get(include=[]).get("ids", []))
    new_rows = [i for i, item_id in enumerate(ids) if item_id not in existing]
    if not new_rows:
        return 0

    collection.add(
        ids=[ids[i] for i in new_rows],
        documents=[documents[i] for i in new_rows],
        metadatas=[metadatas[i] for i in new_rows],
    )
    return len(new_rows)


def ingest_paths(file_paths: Iterable[str | Path], reset_db: bool = False) -> dict:
    ensure_runtime_dirs()
    settings = get_settings()

    if reset_db:
        reset_collection()

    all_pages: list[dict] = []
    processed_files: list[str] = []

    valid_paths = [Path(path) for path in file_paths if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS]
    for file_path in tqdm(valid_paths, desc="Processing documents"):
        pages = extract_document(file_path)
        if pages:
            all_pages.extend(pages)
            processed_files.append(file_path.name)

    chunks = chunk_extracted_pages(
        all_pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    indexed_count = save_to_vector_db(chunks)

    return {
        "processed_files": processed_files,
        "pages_extracted": len(all_pages),
        "chunks_created": len(chunks),
        "chunks_indexed": indexed_count,
    }


def save_uploaded_files(uploaded_files) -> list[Path]:
    ensure_runtime_dirs()
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        destination = TEMP_DIR / uploaded_file.name
        with open(destination, "wb") as file_obj:
            file_obj.write(uploaded_file.getbuffer())
        saved_paths.append(destination)

    return saved_paths


def clear_temp_uploads() -> None:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
