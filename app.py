from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingest import clear_temp_uploads, ingest_paths, save_uploaded_files
from src.query import query_rag_pipeline

load_dotenv()

st.set_page_config(page_title="Document Q&A Bot", page_icon="📄", layout="wide")

st.title("📄 Document Q&A Bot with RAG")
st.caption("Upload PDF, DOCX, or TXT files, index them into ChromaDB, and ask grounded questions with citations.")

with st.sidebar:
    st.header("Setup")
    st.markdown(
        "1. Add `GEMINI_API_KEY` to your `.env` file.\n"
        "2. Upload files.\n"
        "3. Click **Index Documents**.\n"
        "4. Ask questions from your uploaded knowledge base."
    )
    reset_db = st.checkbox("Reset vector database before indexing", value=False)

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="You can upload multiple files together."
)

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Index Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one supported file.")
        else:
            try:
                saved_paths = save_uploaded_files(uploaded_files)
                with st.spinner("Extracting, chunking, embedding, and saving documents..."):
                    summary = ingest_paths(saved_paths, reset_db=reset_db)
                st.success("Documents indexed successfully.")
                st.json(summary)
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
            finally:
                clear_temp_uploads()

with col2:
    question = st.text_input("Ask a question about the indexed documents")
    if st.button("Get Answer", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving context and generating answer..."):
                    result = query_rag_pipeline(question)
                st.subheader("Answer")
                st.write(result["answer"])

                if result["retrieved_chunks"]:
                    st.subheader("Retrieved Context")
                    for chunk in result["retrieved_chunks"]:
                        label = f"{chunk['source']} | Page {chunk['page']} | Chunk {chunk['chunk_index']}"
                        with st.expander(label):
                            st.write(chunk["text"])
                            if chunk["distance"] is not None:
                                st.caption(f"Distance score: {chunk['distance']}")
            except Exception as exc:
                st.error(f"Query failed: {exc}")

st.markdown("---")
st.markdown("Built with Streamlit, ChromaDB, PyPDF, python-docx, and Gemini.")
