import re
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

import config
from chunking import normalize
from embeddings import get_embeddings

NO_ANSWER = "দুঃখিত, এই প্রশ্নের উত্তর বইটিতে পাওয়া যায়নি।"

SYSTEM_PROMPT = f"""তুমি রবীন্দ্রনাথ ঠাকুরের উপন্যাস "রাজর্ষি" বিষয়ে প্রশ্নের উত্তর দাও।
নিয়ম:
1. শুধু নিচের CONTEXT অংশে দেওয়া বইয়ের উদ্ধৃতি থেকে উত্তর দেবে। নিজের জানা তথ্য, অনুমান বা বইয়ের বাইরের কিছু যোগ করবে না।
2. আগে সবগুলো উদ্ধৃতি পড়বে, তারপর যে উদ্ধৃতিতে প্রশ্নের সরাসরি উত্তর আছে সেটি ব্যবহার করবে। শুধু যা জিজ্ঞাসা করা হয়েছে তার উত্তর দেবে; প্রশ্নে চাওয়া হয়নি এমন বাড়তি ঘটনা বা তথ্য লিখবে না। কে কাকে কী বলল বা করল তা উদ্ধৃতির সঙ্গে মিলিয়ে নেবে।
3. প্রতিটি তথ্যের পরে যে উদ্ধৃতি থেকে নেওয়া তার নম্বর বন্ধনীতে দেবে, যেমন [1] বা [2][3]।
4. কোনো উদ্ধৃতিতেই উত্তর না থাকলে কোনো নম্বর ছাড়া হুবহু এই বাক্যটি লিখবে: {NO_ANSWER}
5. উত্তর বাংলায়, সংক্ষেপে ও স্পষ্টভাবে দেবে (১ থেকে ৪ বাক্য)। সাধারণ লেখা দেবে, বোল্ড বা অন্য মার্কডাউন ব্যবহার করবে না।"""

USER_PROMPT = """CONTEXT:
{context}

প্রশ্ন: {question}"""


class ScoredRetriever(BaseRetriever):
    """FAISS similarity search that keeps the cosine score in each document's metadata.

    With window > 0 each hit is widened with its neighbouring chunks from the same chapter,
    so the LLM sees the whole scene around the matching passage.
    """

    vectorstore: FAISS
    k: int = config.TOP_K
    window: int = config.WINDOW

    _by_id: dict | None = PrivateAttr(default=None)

    def _neighbours(self) -> dict:
        if self._by_id is None:
            self._by_id = {d.metadata["chunk_id"]: d for d in self.vectorstore.docstore._dict.values()}
        return self._by_id

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        results = self.vectorstore.similarity_search_with_score(normalize(query), k=self.k)
        passages = []
        for doc, score in results:
            chapter, idx = doc.metadata["chunk_id"].split("-")
            lo, hi = int(idx) - self.window, int(idx) + self.window
            for p in passages:
                if p["chapter"] == chapter and lo <= p["hi"] + 1 and hi >= p["lo"] - 1:
                    p["lo"], p["hi"] = min(lo, p["lo"]), max(hi, p["hi"])
                    break
            else:
                meta = dict(doc.metadata, score=round(float(score), 4))
                passages.append({"chapter": chapter, "lo": lo, "hi": hi, "meta": meta})

        by_id = self._neighbours()
        docs = []
        for p in passages:
            parts = [by_id[c].page_content for j in range(p["lo"], p["hi"] + 1) if (c := f"{p['chapter']}-{j:03d}") in by_id]
            docs.append(Document(page_content="\n".join(parts), metadata=p["meta"]))
        return docs


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[dict] = field(default_factory=list)
    retrieved: list[Document] = field(default_factory=list)


@lru_cache(maxsize=4)
def load_store(model: str = config.EMBED_MODEL, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> FAISS:
    path = config.index_dir(model, size, overlap)
    if not path.exists():
        raise FileNotFoundError(f"Index not found at {path}. Run: python build_index.py")
    return FAISS.load_local(str(path), get_embeddings(model), allow_dangerous_deserialization=True)


def get_retriever(k: int = config.TOP_K, window: int = config.WINDOW, **kw) -> ScoredRetriever:
    return ScoredRetriever(vectorstore=load_store(**kw), k=k, window=window)


def get_llm():
    import os
    provider = config.LLM_PROVIDER
    model = config.LLM_MODEL or config.DEFAULT_MODELS.get(provider, "")
    key_var = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    if key_var and not os.getenv(key_var):
        raise RuntimeError(f"{key_var} is not set. Copy .env.example to .env and add your key.")
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=2048)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


@lru_cache(maxsize=1)
def get_chain():
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
    return prompt | get_llm() | StrOutputParser()


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"[{i}] ({d.metadata['chapter']})\n{d.page_content}")
    return "\n\n".join(parts)


def sources_for(indices: list[int], docs: list[Document]) -> list[dict]:
    sources, seen = [], set()
    for n in indices:
        m = docs[n - 1].metadata
        if m["chapter"] not in seen:
            seen.add(m["chapter"])
            sources.append({"chapter": m["chapter"], "chapter_no": m["chapter_no"], "pages": m["pages"], "url": m["url"]})
    return sources


def cite(text: str, docs: list[Document]) -> tuple[str, list[dict]]:
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(docs)})
    sources = sources_for(used, docs)

    def relabel(match):
        n = int(match.group(1))
        if 1 <= n <= len(docs):
            return f"[{docs[n - 1].metadata['chapter']}]"
        return ""

    text = re.sub(r"\[(\d+)\]", relabel, text)
    text = re.sub(r"(\[[^\]]+\])(\1)+", r"\1", text)
    return text.strip(), sources


def answer(question: str, retriever: ScoredRetriever | None = None, min_score: float = config.MIN_SCORE) -> Answer:
    retriever = retriever or get_retriever()
    docs = retriever.invoke(question)
    if not docs or docs[0].metadata["score"] < min_score:
        return Answer(NO_ANSWER, False, [], docs)

    raw = get_chain().invoke({"context": format_context(docs), "question": question}).strip()
    if NO_ANSWER in raw or "পাওয়া যায়নি" in raw[:80]:
        return Answer(NO_ANSWER, False, [], docs)
    text, sources = cite(raw, docs)
    if not sources:
        sources = sources_for(list(range(1, len(docs) + 1)), docs)
        text += "\n\n(মডেল নির্দিষ্ট উদ্ধৃতি দেয়নি; নিচের পরিচ্ছেদগুলো থেকে উত্তর তৈরি হয়েছে।)"
    return Answer(text, True, sources, docs)


def format_sources(sources: list[dict]) -> str:
    lines = []
    for s in sources:
        lines.append(f"রাজর্ষি, {s['chapter']} (মুদ্রিত পৃষ্ঠা {s['pages']}): {s['url']}")
    return "\n".join(lines)
