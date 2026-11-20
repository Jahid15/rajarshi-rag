from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

import config


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class PrefixedEmbeddings(Embeddings):
    """E5 models expect 'query: ' and 'passage: ' prefixes."""

    def __init__(self, base: Embeddings, query_prefix: str, doc_prefix: str):
        self.base = base
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts):
        return self.base.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text):
        return self.base.embed_query(self.query_prefix + text)


def get_embeddings(model: str = config.EMBED_MODEL) -> Embeddings:
    base = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": _device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    if "e5" in model.lower():
        return PrefixedEmbeddings(base, "query: ", "passage: ")
    return base
