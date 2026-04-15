from __future__ import annotations

from time import perf_counter

import streamlit as st

from chain import TechRAG


@st.cache_resource
def get_rag() -> TechRAG:
    return TechRAG()


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
