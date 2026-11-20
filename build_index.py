import argparse
import json
import time

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

import config
from chunking import make_chunks
from embeddings import get_embeddings


def build(model: str, size: int, overlap: int) -> None:
    out = config.index_dir(model, size, overlap)
    chunks = make_chunks(size, overlap)
    print(f"{len(chunks)} chunks (size {size}, overlap {overlap}), embedding with {model}")
    start = time.time()
    store = FAISS.from_documents(chunks, get_embeddings(model), distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))
    meta = {"model": model, "chunk_size": size, "chunk_overlap": overlap, "chunks": len(chunks)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Saved FAISS index to {out} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.EMBED_MODEL)
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    a = ap.parse_args()
    build(a.model, a.chunk_size, a.chunk_overlap)
