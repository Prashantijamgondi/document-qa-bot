from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.ingest import ingest_paths
from src.query import query_rag_pipeline


def run_cli() -> None:
    load_dotenv()
    data_dir = Path("data")
    files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.docx")) + list(data_dir.glob("*.txt"))

    if not files:
        print("No supported files found inside the data/ folder.")
        return

    summary = ingest_paths(files, reset_db=False)
    print("Ingestion complete:")
    print(summary)

    print("\nAsk questions about your documents. Type 'exit' to stop.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        try:
            result = query_rag_pipeline(question)
            print(f"\nAnswer:\n{result['answer']}\n")
            if result["citations"]:
                print("Sources:")
                for item in result["citations"]:
                    print(f"- {item}")
                print()
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    run_cli()
