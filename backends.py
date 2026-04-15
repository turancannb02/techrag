from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Load local .env for local runs; HF Spaces injects env vars directly.
load_dotenv()


def get_embeddings(cfg: dict):
    models = cfg.get("models", {})
    backend = models.get("embedding_backend", "ollama").lower()

    if backend == "ollama":
        return OllamaEmbeddings(
            model=models["embedding"],
            base_url=models.get("base_url", "http://localhost:11434"),
        )

    if backend in {"hf", "huggingface"}:
        return HuggingFaceEmbeddings(model_name=models["embedding"])

    raise ValueError(
        f"Unsupported embedding backend '{backend}'. "
        "Use 'ollama' or 'huggingface'."
    )


def get_llm(cfg: dict):
    models = cfg.get("models", {})
    backend = models.get("llm_backend", "ollama").lower()

    if backend == "ollama":
        return ChatOllama(
            model=models["llm"],
            base_url=models.get("base_url", "http://localhost:11434"),
            temperature=0.1,
        )

    if backend == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Define it in environment variables (or .env for local runs)."
            )
        return ChatGroq(
            model=models["llm"],
            api_key=api_key,
            temperature=0.1,
        )

    raise ValueError(
        f"Unsupported llm backend '{backend}'. "
        "Use 'ollama' or 'groq'."
    )
