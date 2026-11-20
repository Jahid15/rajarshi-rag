"""Run the 10 test questions through the full pipeline and save a report."""

import argparse
import json
import sys

import config
from rag import answer, get_retriever


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM and only check retrieval")
    args = ap.parse_args()

    tests = json.loads((config.ROOT / "tests" / "test_questions.json").read_text(encoding="utf-8"))
    retriever = get_retriever()
    lines = [f"# Test results ({'retrieval only' if args.retrieval_only else 'full pipeline'})", ""]
    passed = 0

    for t in tests:
        docs = retriever.invoke(t["question"])
        top = docs[0].metadata["score"]
        retrieved = [d.metadata["chapter_no"] for d in docs]
        lines += [f"## Q{t['id']}. {t['question']}", "", f"Expected: {t['expected_answer']}", ""]

        if args.retrieval_only:
            if t["in_book"]:
                ok = any(c in t["chapters"] for c in retrieved) and top >= config.MIN_SCORE
            else:
                ok = True
                gate = "stopped by score gate" if top < config.MIN_SCORE else "passed to LLM, which must refuse"
                lines += [f"No-answer question: {gate}", ""]
            lines += [f"Top score: {top:.3f}, retrieved chapters: {retrieved}", ""]
        else:
            res = answer(t["question"], retriever)
            cited = [s["chapter_no"] for s in res.sources]
            if t["in_book"]:
                ok = res.found and any(c in t["chapters"] for c in cited)
            else:
                ok = not res.found
            lines += [f"Chatbot: {res.text}", "", f"Cited: {', '.join(s['chapter'] for s in res.sources) or 'none'}", ""]
        lines += [f"Result: {'PASS' if ok else 'CHECK'}", ""]
        passed += ok
        print(f"Q{t['id']}: {'PASS' if ok else 'CHECK'} (top score {top:.3f})")

    lines.insert(1, f"\n{passed}/{len(tests)} passed. PASS means the answer cites a correct chapter, or correctly says the answer is not in the book.")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    name = "test_results_retrieval.md" if args.retrieval_only else "test_results.md"
    (config.RESULTS_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{passed}/{len(tests)} passed, report saved to results/{name}")


if __name__ == "__main__":
    main()
