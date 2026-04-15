from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator

import yaml

from backends import get_llm
from retriever import HybridRetriever


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Answer:
    query: str
    answer: str
    sources: list[dict]


@dataclass
class PreparedQuery:
    query: str
    prompt: str
    sources: list[dict]
    timings: dict[str, float]


class TechRAG:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = load_config(config_path)
        self.retriever = HybridRetriever(config_path=config_path)
        self.llm = get_llm(self.cfg)

    def _build_prompt(self, query: str, contexts: list[dict]) -> str:
        context_block = "\n\n".join(
            f"[{i}] source={item['metadata'].get('source', 'unknown')}\n{item['text']}"
            for i, item in enumerate(contexts, start=1)
        )
        return (
            "You are an assistant for technical documentation QA. "
            "Answer using only the provided context. If context is insufficient, say so clearly. "
            "Cite sources as [1], [2], etc.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context_block}\n\n"
            "Answer:"
        )

    def ask(self, query: str, top_k: int | None = None) -> Answer:
        answer, _ = self.ask_with_timings(query, top_k=top_k)
        return answer

    def _raise_llm_error(self, exc: Exception) -> None:
        models = self.cfg.get("models", {})
        llm_backend = models.get("llm_backend", "ollama").lower()
        msg = str(exc)
        if llm_backend == "ollama" and "model" in msg and "not found" in msg:
            llm_name = models["llm"]
            raise RuntimeError(
                f"Ollama model '{llm_name}' is not available locally. "
                f"Run: ollama pull {llm_name}"
            ) from exc
        if llm_backend == "groq" and "api" in msg.lower() and "key" in msg.lower():
            raise RuntimeError("Groq authentication failed. Check GROQ_API_KEY.") from exc

    def prepare_query(self, query: str, top_k: int | None = None) -> PreparedQuery:
        t0 = perf_counter()
        results = self.retriever.search(query, top_k=top_k)
        t1 = perf_counter()
        contexts = [{"text": r.text, "metadata": r.metadata, "score": r.score} for r in results]
        prompt = self._build_prompt(query, contexts)
        t2 = perf_counter()

        sources = [
            {
                "source": c["metadata"].get("source", "unknown"),
                "chunk_id": c["metadata"].get("chunk_id"),
                "score": round(float(c["score"]), 4),
            }
            for c in contexts
        ]
        t3 = perf_counter()

        timings = {
            "retrieval_ms": round((t1 - t0) * 1000, 2),
            "prompt_build_ms": round((t2 - t1) * 1000, 2),
            "postprocess_ms": round((t3 - t2) * 1000, 2),
        }
        return PreparedQuery(query=query, prompt=prompt, sources=sources, timings=timings)

    def stream_from_prompt(self, prompt: str) -> Iterator[str]:
        try:
            for chunk in self.llm.stream(prompt):
                text = chunk.content
                if isinstance(text, str) and text:
                    yield text
        except Exception as exc:
            self._raise_llm_error(exc)
            raise

    def ask_with_timings(self, query: str, top_k: int | None = None) -> tuple[Answer, dict[str, float]]:
        prepared = self.prepare_query(query, top_k=top_k)
        t0 = perf_counter()
        try:
            reply = self.llm.invoke(prepared.prompt)
        except Exception as exc:
            self._raise_llm_error(exc)
            raise
        t1 = perf_counter()

        timings = dict(prepared.timings)
        timings["llm_inference_ms"] = round((t1 - t0) * 1000, 2)
        timings["total_ms"] = round(sum(timings.values()), 2)

        return Answer(query=query, answer=reply.content, sources=prepared.sources), timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single TechRAG query.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--show-timings", action="store_true", help="Print per-stage timing metrics.")
    parser.add_argument("--stream", action="store_true", help="Stream tokens while generating the answer.")
    args = parser.parse_args()

    rag = TechRAG(config_path=args.config)
    if args.stream:
        prepared = rag.prepare_query(args.query, top_k=args.top_k)
        t0 = perf_counter()
        print("\nAnswer:\n")
        chunks: list[str] = []
        for token in rag.stream_from_prompt(prepared.prompt):
            print(token, end="", flush=True)
            chunks.append(token)
        print("")
        t1 = perf_counter()

        answer = Answer(query=args.query, answer="".join(chunks), sources=prepared.sources)
        timings = dict(prepared.timings)
        timings["llm_inference_ms"] = round((t1 - t0) * 1000, 2)
        timings["total_ms"] = round(sum(timings.values()), 2)
    else:
        answer, timings = rag.ask_with_timings(args.query, top_k=args.top_k)

    if not args.stream:
        print("\nAnswer:\n")
        print(answer.answer)
    print("\nSources:")
    for src in answer.sources:
        print(f"- {src['source']} (chunk={src['chunk_id']}, score={src['score']})")
    if args.show_timings:
        print("\nTimings (ms):")
        print(f"- retrieval: {timings['retrieval_ms']}")
        print(f"- prompt_build: {timings['prompt_build_ms']}")
        print(f"- llm_inference: {timings['llm_inference_ms']}")
        print(f"- postprocess: {timings['postprocess_ms']}")
        print(f"- total: {timings['total_ms']}")


if __name__ == "__main__":
    main()
