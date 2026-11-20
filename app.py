import streamlit as st

import config
from rag import answer, get_retriever

st.set_page_config(page_title="রাজর্ষি চ্যাটবট", page_icon="📖")
st.title("রাজর্ষি চ্যাটবট")
st.caption(f"রবীন্দ্রনাথ ঠাকুরের \"রাজর্ষি\" উপন্যাস থেকে উত্তর দেয়, প্রতিটি উত্তরে পরিচ্ছেদের উল্লেখসহ। উৎস: [Wikisource]({config.page_url(config.BOOK_TITLE)})")

with st.sidebar:
    st.subheader("Settings")
    k = st.slider("Retrieved chunks (k)", 2, 10, config.TOP_K)
    min_score = st.slider("Minimum similarity to answer", 0.0, 1.0, config.MIN_SCORE, 0.01)
    st.markdown(
        f"Embedding: `{config.EMBED_MODEL}`  \n"
        f"Chunks: {config.CHUNK_SIZE} chars, overlap {config.CHUNK_OVERLAP}  \n"
        f"Vector store: FAISS  \n"
        f"LLM: `{config.LLM_PROVIDER}` / `{config.LLM_MODEL or config.DEFAULT_MODELS.get(config.LLM_PROVIDER)}`"
    )
    st.markdown("Try:  \n- রাজা গোবিন্দমাণিক্য কোন রাজ্যের রাজা ছিলেন?  \n- হাসির ভাইয়ের নাম কী?  \n- রাজা মন্দিরে কী নিষেধ করলেন?")
    if st.button("Clear chat"):
        st.session_state.messages = []


@st.cache_resource(show_spinner="Loading embedding model and FAISS index...")
def retriever_for(k: int):
    return get_retriever(k=k)


def render_sources(msg: dict):
    if msg.get("sources"):
        st.markdown("সূত্র:")
        for s in msg["sources"]:
            st.markdown(f"- [রাজর্ষি, {s['chapter']}]({s['url']}) (মুদ্রিত পৃষ্ঠা {s['pages']})")
    if msg.get("chunks"):
        with st.expander("Retrieved passages"):
            for c in msg["chunks"]:
                st.markdown(f"{c['chapter']}, score {c['score']:.3f}")
                st.text(c["text"])


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg)

if question := st.chat_input("রাজর্ষি সম্পর্কে প্রশ্ন করুন"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("খুঁজছি..."):
            try:
                result = answer(question, retriever_for(k), min_score)
                content = result.text
                sources = result.sources
                chunks = [{"chapter": d.metadata["chapter"], "score": d.metadata["score"], "text": d.page_content} for d in result.retrieved]
            except Exception as e:
                content, sources, chunks = f"Error: {e}", [], []
        msg = {"role": "assistant", "content": content, "sources": sources, "chunks": chunks}
        st.markdown(content)
        render_sources(msg)
    st.session_state.messages.append(msg)
