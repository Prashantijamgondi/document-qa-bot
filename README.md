# Document Q&A Bot with RAG

A production-ready Python RAG project built for an AI Engineering internship assignment. The app ingests private documents, stores embeddings in a persistent local ChromaDB database, retrieves relevant chunks for a user query, and generates grounded answers with inline citations.

## Features

- Supports PDF, DOCX, and TXT documents
- Uses ChromaDB as a local persistent vector database
- Uses Gemini embeddings and Gemini text generation
- Provides grounded answers using only retrieved context
- Shows source references with filename and page number
- Includes both CLI and Streamlit interfaces
- Keeps ingestion and querying separate for clean architecture

## Tech Stack

- Python 3.11+
- Streamlit
- ChromaDB
- PyPDF
- python-docx
- google-generativeai
- python-dotenv
- tqdm

## Project Structure

```text
document-qa-bot/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── data/
├── db/
├── app.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── ingest.py
    ├── query.py
    └── main.py
```

## How It Works

1. Documents are loaded from uploads or the `data/` folder.
2. PDF pages and document text are extracted with metadata.
3. Long text is broken into overlapping chunks.
4. Chunks are embedded using Gemini embeddings.
5. Embeddings are stored in persistent ChromaDB.
6. On each user query, the system retrieves the top matching chunks.
7. Gemini generates an answer using only retrieved context.
8. The response includes source citations like `(file.pdf, Page 2)`.

## Local Setup

### 1. Create the project folder

```bash
mkdir document-qa-bot
cd document-qa-bot
```

### 2. Create and activate a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the environment file

Copy `.env.example` to `.env` and add your API key:

```env
GEMINI_API_KEY=your_real_api_key_here
```

### 5. Add documents

Put sample `.pdf`, `.docx`, or `.txt` files inside the `data/` folder, or upload them from the Streamlit UI.

## Run the Project

### Option A: CLI mode

```bash
python -m src.main
```

### Option B: Streamlit mode

```bash
streamlit run app.py
```

## Deployment on Render

1. Push this project to a public GitHub repository.
2. Go to [Render](https://render.com/) and create a new **Web Service** from your repo.
3. Use these settings:
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add an environment variable named `GEMINI_API_KEY` in the Render dashboard.
5. Deploy and test the live app.

