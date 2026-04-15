from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    return soup.get_text("\n", strip=True)


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def iter_documents(source_dir: Path) -> Iterable[Document]:
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            text = load_text(path)
        elif suffix in {".html", ".htm"}:
            text = load_html(path)
        else:
            text = load_pdf(path)

        if text.strip():
            yield Document(page_content=text, metadata={"source": str(path)})


def save_chunks(chunks: list[Document], chunks_file: Path) -> None:
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with chunks_file.open("w", encoding="utf-8") as f:
        for doc in chunks:
            f.write(
                json.dumps({"text": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents and build FAISS index.")
    parser.add_argument("--source", default="data", help="Directory containing source documents.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    chunk_size = args.chunk_size or cfg["chunking"]["chunk_size"]
    overlap = args.overlap or cfg["chunking"]["overlap"]

    files_in_source = [p for p in source_dir.rglob("*") if p.is_file()]
    docs = list(iter_documents(source_dir))
    if not docs:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"No supported documents found in {source_dir}.\n"
            f"Found {len(files_in_source)} file(s), but none matched supported extensions: {supported_list}\n"
            "Add files like .md/.txt/.pdf under the source directory and run ingest again."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    embeddings = OllamaEmbeddings(
        model=cfg["models"]["embedding"],
        base_url=cfg["models"]["base_url"],
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir = Path(cfg["storage"]["index_dir"])
    chunks_file = Path(cfg["storage"]["chunks_file"])
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    save_chunks(chunks, chunks_file)

    print(f"Loaded documents: {len(docs)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Saved FAISS index to: {index_dir}")
    print(f"Saved chunks to: {chunks_file}")


if __name__ == "__main__":
    main()
