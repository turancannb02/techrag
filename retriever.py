from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Workaround for OpenMP duplication crashes on some local macOS setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class RetrievalResult:
    text: str
    metadata: dict
    score: float


class HybridRetriever:
    def __init__(self, config_path: str = "config.yaml") -> None:
        cfg = load_config(config_path)
        self.cfg = cfg

        embeddings = OllamaEmbeddings(
            model=cfg["models"]["embedding"],
            base_url=cfg["models"]["base_url"],
        )
        self.vectorstore = FAISS.load_local(
            cfg["storage"]["index_dir"],
            embeddings,
            allow_dangerous_deserialization=True,
        )

        self.chunks = self._load_chunks(Path(cfg["storage"]["chunks_file"]))
        if not self.chunks:
            raise ValueError("No chunks loaded. Run ingest.py first.")

        tokenized = [self._tokenize(item["text"]) for item in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _load_chunks(self, path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"Missing chunks file: {path}")
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _semantic_scores(self, query: str, fetch_k: int) -> dict[int, float]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
        semantic: dict[int, float] = {}
        for doc, distance in docs_and_scores:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is None:
                continue
            semantic[chunk_id] = max(semantic.get(chunk_id, 0.0), 1.0 / (1.0 + float(distance)))
        return semantic

    def _bm25_scores(self, query: str) -> dict[int, float]:
        scores = self.bm25.get_scores(self._tokenize(query))
        max_score = max(scores) if len(scores) else 0.0
        if max_score <= 0:
            return {}
        return {idx: float(score) / float(max_score) for idx, score in enumerate(scores) if score > 0}

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self.cfg["retrieval"]["top_k"]
        bm25_weight = float(self.cfg["retrieval"].get("hybrid_weight", 0.6))
        semantic_weight = 1.0 - bm25_weight

        semantic = self._semantic_scores(query, fetch_k=max(top_k * 4, 20))
        bm25 = self._bm25_scores(query)

        merged: dict[int, float] = {}
        for chunk_id, s in semantic.items():
            merged[chunk_id] = merged.get(chunk_id, 0.0) + semantic_weight * s
        for chunk_id, s in bm25.items():
            merged[chunk_id] = merged.get(chunk_id, 0.0) + bm25_weight * s

        best = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results: list[RetrievalResult] = []
        for chunk_id, score in best:
            row = self.chunks[chunk_id]
            results.append(RetrievalResult(text=row["text"], metadata=row["metadata"], score=score))
        return results

    @staticmethod
    def to_documents(results: list[RetrievalResult]) -> list[Document]:
        return [Document(page_content=r.text, metadata=r.metadata) for r in results]
