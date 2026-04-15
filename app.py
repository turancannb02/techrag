from __future__ import annotations

from pathlib import Path
from time import perf_counter

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chain import TechRAG
from ingest import iter_documents, load_config, save_chunks

try:
    from backends import get_embeddings as resolve_embeddings
except ImportError:
    from langchain_ollama import OllamaEmbeddings

    def resolve_embeddings(cfg: dict):
        return OllamaEmbeddings(
            model=cfg["models"]["embedding"],
            base_url=cfg["models"]["base_url"],
        )


@st.cache_resource
def get_rag() -> TechRAG:
    ensure_index_ready()
    return TechRAG()


def ensure_index_ready(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    index_dir = Path(cfg["storage"]["index_dir"])
    chunks_file = Path(cfg["storage"]["chunks_file"])
    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"

    if faiss_file.exists() and pkl_file.exists() and chunks_file.exists():
        return

    source_candidates = [Path("data"), Path("demo_data")]
    source_dir = next((p for p in source_candidates if p.exists()), None)
    if source_dir is None:
        raise RuntimeError("No source directory found. Expected `data/` or `demo_data/`.")

    docs = list(iter_documents(source_dir))
    if not docs and source_dir.name != "demo_data":
        fallback = Path("demo_data")
        if fallback.exists():
            docs = list(iter_documents(fallback))
            source_dir = fallback
    if not docs:
        raise RuntimeError("No supported documents found to build index.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["overlap"],
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    embeddings = resolve_embeddings(cfg)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    save_chunks(chunks, chunks_file)

    st.info(f"Index was missing and has been built from `{source_dir}`.")


def render_sources(sources: list[dict]) -> None:
    st.subheader("Sources")
    if not sources:
        st.info("No sources returned.")
        return
    for src in sources:
        st.markdown(
            f"- `{src.get('source', 'unknown')}` "
            f"(chunk={src.get('chunk_id')}, score={src.get('score')})"
        )


def render_timings(timings: dict[str, float] | None) -> None:
    if not timings:
        return
    st.subheader("Timings (ms)")
    cols = st.columns(5)
    keys = ["retrieval_ms", "prompt_build_ms", "llm_inference_ms", "postprocess_ms", "total_ms"]
    for col, key in zip(cols, keys):
        col.metric(key.replace("_ms", ""), f"{timings.get(key, 0):.2f}")


def main() -> None:
    st.set_page_config(page_title="TechRAG", page_icon="📚", layout="wide")
    st.title("TechRAG")
    st.caption("Local RAG over technical documents with hybrid retrieval (FAISS + BM25).")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5, step=1)
        stream = st.toggle("Stream answer", value=True)

    query = st.text_area(
        "Question",
        value="What are the key waveform candidates for 6G?",
        height=100,
        placeholder="Ask about your technical corpus...",
    )

    ask = st.button("Ask", type="primary")
    if not ask:
        return

    if not query.strip():
        st.warning("Enter a question first.")
        return

    rag = get_rag()

    try:
        if stream:
            prepared = rag.prepare_query(query=query, top_k=top_k)
            st.subheader("Answer")
            answer_box = st.empty()
            full_text = ""
            t0 = perf_counter()
            for token in rag.stream_from_prompt(prepared.prompt):
                full_text += token
                answer_box.markdown(full_text)
            t1 = perf_counter()

            timings = dict(prepared.timings)
            timings["llm_inference_ms"] = round((t1 - t0) * 1000, 2)
            timings["total_ms"] = round(sum(timings.values()), 2)
            render_sources(prepared.sources)
            render_timings(timings)
        else:
            answer, timings = rag.ask_with_timings(query=query, top_k=top_k)
            st.subheader("Answer")
            st.markdown(answer.answer)
            render_sources(answer.sources)
            render_timings(timings)
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
