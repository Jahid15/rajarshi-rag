"""Bonus: compare embedding models and chunking strategies with a retrieval hit-rate score."""

import json
import statistics
import sys

import config
from build_index import build
from rag import get_retriever

CONFIGS = [
    ("BAAI/bge-m3", 600, 120),
    ("BAAI/bge-m3", 300, 60),
    ("BAAI/bge-m3", 1000, 200),
    ("intfloat/multilingual-e5-base", 600, 120),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 600, 120),
]
KS = (1, 3, 5)


def evaluate(model, size, overlap, questions, negatives):
    if not (config.index_dir(model, size, overlap) / "index.faiss").exists():
        build(model, size, overlap)
    retriever = get_retriever(k=max(KS), model=model, size=size, overlap=overlap)
    hits = {k: 0 for k in KS}
    rr = []
    pos_scores = []
    for q in questions:
        docs = retriever.invoke(q["question"])
        chapters = [d.metadata["chapter_no"] for d in docs]
        pos_scores.append(docs[0].metadata["score"])
        rank = next((i + 1 for i, c in enumerate(chapters) if c in q["chapters"]), None)
        rr.append(1 / rank if rank else 0)
        for k in KS:
            hits[k] += rank is not None and rank <= k
    neg_scores = [retriever.invoke(q)[0].metadata["score"] for q in negatives]
    n = len(questions)
    return {
        "model": model,
        "chunk_size": size,
        "chunk_overlap": overlap,
        **{f"hit@{k}": round(hits[k] / n, 3) for k in KS},
        "mrr": round(sum(rr) / n, 3),
        "mean_top_score_in_book": round(statistics.mean(pos_scores), 3),
        "mean_top_score_not_in_book": round(statistics.mean(neg_scores), 3),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads((config.ROOT / "eval" / "eval_questions.json").read_text(encoding="utf-8"))
    questions, negatives = data["questions"], data["not_in_book"]
    rows = []
    for model, size, overlap in CONFIGS:
        print(f"Evaluating {model} {size}/{overlap}")
        row = evaluate(model, size, overlap, questions, negatives)
        print(row)
        rows.append(row)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "retrieval_comparison.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    lines = [
        f"# Retrieval comparison ({len(questions)} questions with a known gold chapter)",
        "",
        "A hit means a chunk from the correct chapter appears in the top k results.",
        "",
        "| Embedding model | Chunk size / overlap | Hit@1 | Hit@3 | Hit@5 | MRR | Top score (in book) | Top score (not in book) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model'].split('/')[-1]} | {r['chunk_size']} / {r['chunk_overlap']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | "
            f"{r['hit@5']:.2f} | {r['mrr']:.2f} | {r['mean_top_score_in_book']:.3f} | {r['mean_top_score_not_in_book']:.3f} |"
        )
    (config.RESULTS_DIR / "retrieval_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
