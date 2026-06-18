# from __future__ import annotations

# import os
# from typing import Any

# import chromadb
# from google import genai
# from google.genai import types
# from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

# from src.config import COLLECTION_NAME, DB_DIR, get_settings
# from chromadb.config import Settings

# def configure_gemini() -> None:
#     settings = get_settings()
#     genai.configure(api_key=settings.gemini_api_key)

# def get_collection():
#     settings = get_settings()
#     client = chromadb.PersistentClient(
#         path=str(DB_DIR),
#         settings=Settings(anonymized_telemetry=False)
#     )
#     embedding_fn = GoogleGenerativeAiEmbeddingFunction(
#         api_key=settings.gemini_api_key,
#         model_name=settings.embedding_model,
#     )
#     return client.get_or_create_collection(
#         name=COLLECTION_NAME,
#         embedding_function=embedding_fn,
#         metadata={"hnsw:space": "cosine"},
#     )


# def format_context(results: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]]:
#     documents = results.get("documents", [[]])[0]
#     metadatas = results.get("metadatas", [[]])[0]
#     distances = results.get("distances", [[]])[0] if results.get("distances") else []

#     context_blocks: list[str] = []
#     citations: list[str] = []
#     retrieved_chunks: list[dict[str, Any]] = []

#     for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
#         if not meta:
#             continue
#         distance = distances[idx] if idx < len(distances) else None
#         source_name = meta.get("source", "unknown")
#         page_num = meta.get("page", "?")
#         chunk_idx = meta.get("chunk_index", "?")
#         citation_str = f"{source_name}, Page {page_num}, Chunk {chunk_idx}"
#         context_blocks.append(
#             f"[Source: {source_name}, Page: {page_num}, Chunk: {chunk_idx}]\n{doc}"
#         )
#         citations.append(citation_str)
#         retrieved_chunks.append(
#             {
#                 "source": source_name,
#                 "page": page_num,
#                 "chunk_index": chunk_idx,
#                 "distance": distance,
#                 "text": doc,
#             }
#         )

#     return "\n\n---\n\n".join(context_blocks), citations, retrieved_chunks

# def get_genai_client():
#     settings = get_settings()
#     return genai.Client(api_key=settings.gemini_api_key)

# def query_rag_pipeline(user_query: str, k: int | None = None) -> dict[str, Any]:
#     settings = get_settings()
#     if len(user_query.strip()) < settings.min_query_length:
#         raise ValueError("Please enter a more specific question.")

#     collection = get_collection()

#     results = collection.query(
#         query_texts=[user_query],
#         n_results=k or settings.retrieval_k,
#         include=["documents", "metadatas", "distances"],
#     )

#     context_payload, citations, retrieved_chunks = format_context(results)

#     if not context_payload.strip():
#         return {
#             "answer": "I am sorry, but the provided documents do not contain the answer to your question.",
#             "citations": [],
#             "retrieved_chunks": [],
#         }

#     prompt = f"""
# You are a precise document Q&A assistant.
# Use ONLY the provided context to answer the user's question.
# If the answer is not present in the context, reply exactly with:
# I am sorry, but the provided documents do not contain the answer to your question.
# Keep the answer concise and factual.
# Cite sources inline in this format: (filename, Page X).
# Do not use outside knowledge.

# CONTEXT:
# {context_payload}

# USER QUESTION:
# {user_query}

# GROUNDED ANSWER:
# """.strip()

#     client = get_genai_client()
#     response = client.models.generate_content(
#         model=settings.generation_model,
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             temperature=0.2,
#             max_output_tokens=500,
#         ),
#     )

#     answer = (response.text or "").strip()
#     if not answer:
#         answer = "I am sorry, but the provided documents do not contain the answer to your question."

#     return {
#         "answer": answer,
#         "citations": citations,
#         "retrieved_chunks": retrieved_chunks,
#     }

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from google import genai
from google.genai import types

from src.config import COLLECTION_NAME, DB_DIR, get_settings


def get_genai_client():
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


def get_collection():
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_query(text: str) -> list[float]:
    settings = get_settings()
    client = get_genai_client()
    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
    )
    return response.embeddings[0].values


def format_context(results: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]]:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    context_blocks = []
    citations = []
    retrieved_chunks = []

    for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
        if not meta:
            continue
        distance = distances[idx] if idx < len(distances) else None
        source_name = meta.get("source", "unknown")
        page_num = meta.get("page", "?")
        chunk_idx = meta.get("chunk_index", "?")
        file_size = meta.get("file_size_bytes")
        citation_str = f"{source_name}, Page {page_num}, Chunk {chunk_idx}"

        context_blocks.append(
            f"[Source: {source_name}, Page: {page_num}, Chunk: {chunk_idx}, FileSizeBytes: {file_size}]\n{doc}"
        )
        citations.append(citation_str)
        retrieved_chunks.append(
            {
                "source": source_name,
                "page": page_num,
                "chunk_index": chunk_idx,
                "distance": distance,
                "text": doc,
            }
        )

    return "\n\n---\n\n".join(context_blocks), citations, retrieved_chunks


def query_rag_pipeline(user_query: str, k: int | None = None) -> dict[str, Any]:
    settings = get_settings()
    if len(user_query.strip()) < settings.min_query_length:
        raise ValueError("Please enter a more specific question.")

    collection = get_collection()
    query_embedding = embed_query(user_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k or settings.retrieval_k,
        include=["documents", "metadatas", "distances"],
    )

    context_payload, citations, retrieved_chunks = format_context(results)

    if not context_payload.strip():
        return {
            "answer": "I am sorry, but the provided documents do not contain the answer to your question.",
            "citations": [],
            "retrieved_chunks": [],
        }

    prompt = f"""
You are a precise document Q&A assistant.
Use ONLY the provided context to answer the user's question.
If the answer is not present in the context, reply exactly with:
I am sorry, but the provided documents do not contain the answer to your question.
Keep the answer concise and factual.
Cite sources inline in this format: (filename, Page X).
Do not use outside knowledge.

CONTEXT:
{context_payload}

USER QUESTION:
{user_query}

GROUNDED ANSWER:
""".strip()

    client = get_genai_client()
    response = client.models.generate_content(
        model=settings.generation_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=500,
        ),
    )

    answer = (response.text or "").strip()
    if not answer:
        answer = "I am sorry, but the provided documents do not contain the answer to your question."

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": retrieved_chunks,
    }