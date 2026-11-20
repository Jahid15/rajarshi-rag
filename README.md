# রাজর্ষি RAG Chatbot

A chatbot that answers questions about one Bangla book, রাজর্ষি by রবীন্দ্রনাথ ঠাকুর, using a RAG pipeline. Every answer comes only from the book and cites the chapter (পরিচ্ছেদ) it came from. If the answer is not in the book, the bot says so.

## Demo video

https://drive.google.com/file/d/1mhwBmfT_LVnrvClhJaIqWJuNJ_JRsyY1/view?usp=sharing

## Book

- Title: রাজর্ষি (1961 edition), রবীন্দ্রনাথ ঠাকুর
- Source: https://bn.wikisource.org/wiki/রাজর্ষি_(১৯৬১)
- Size: 44 chapters plus উপসংহার (45 subpages), about 221,000 characters of prose

## Pipeline

```
Wikisource index page
   -> ingest.py        crawl all 45 chapter subpages, strip headers/page numbers, save data/book.json
   -> chunking.py      Unicode normalisation + Bangla-aware recursive splitting (600 chars, 120 overlap)
   -> embeddings.py    BAAI/bge-m3 (multilingual, normalised vectors)
   -> build_index.py   FAISS index (inner product = cosine similarity)
   -> rag.py           LangChain retriever (top 8, widened with neighbouring chunks) -> score gate -> prompt -> LLM -> chapter citations
   -> app.py           Streamlit chat interface
```

## How to run

Python 3.10 to 3.12 is recommended.

```bash
git clone https://github.com/Jahid15/rajarshi-rag.git
cd rajarshi-rag
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

cp .env.example .env            # then add one API key and set LLM_PROVIDER
python ingest.py                # optional: data/book.json is already in the repo
python build_index.py           # downloads bge-m3 (about 2.3 GB) and builds the FAISS index
streamlit run app.py
```

LLM providers (set `LLM_PROVIDER` in `.env`):

| Provider | Key in .env | Default model | Note |
|---|---|---|---|
| gemini | GOOGLE_API_KEY | gemini-2.5-flash | free tier, good Bangla |
| groq | GROQ_API_KEY | llama-3.3-70b-versatile | free tier, fast |
| anthropic | ANTHROPIC_API_KEY | claude-haiku-5-5 | paid, cheap and fast, used for the results in this repo |

Any other model of the same provider can be used with `LLM_MODEL`.

Other scripts:

```bash
python run_tests.py                  # 10 test questions through the full pipeline, writes results/test_results.md
python run_tests.py --retrieval-only # same questions without the LLM, writes results/test_results_retrieval.md
python compare.py                    # bonus: hit-rate comparison of embedding models and chunk sizes
```

## Ingestion

`ingest.py` uses the MediaWiki API. It reads the book's index page, collects every link that starts with `রাজর্ষি (১৯৬১)/` in reading order (45 subpages), then downloads each chapter. From each page it removes the Wikisource header and navigation, page-number markers, styles and the repeated chapter heading, and keeps the paragraph text. It also keeps the printed page range of each chapter (for example ৫-৭), which is shown in the citations. Raw HTML is cached in `data/raw/` so the crawl only runs once, and it waits whenever Wikimedia answers 429 (rate limit).

## Preprocessing and chunking

Preprocessing (`chunking.normalize`):
- Unicode NFC normalisation, so the same letter always has the same code points
- remove zero-width characters (ZWJ, ZWNJ, zero-width space, BOM, soft hyphen) that Wikisource text contains but user questions do not
- replace non-breaking spaces, turn `|` into the Bangla danda `।`, collapse extra spaces and blank lines

Chunking: LangChain `RecursiveCharacterTextSplitter`, chunk size 600 characters, overlap 120 characters, with separators in this order: paragraph, line, `।`, `?`, `!`, `;`, `,`, space. The separator is kept at the end of the chunk, so chunks end at the end of a Bangla sentence instead of in the middle of a word.

Why this choice:
- 600 Bangla characters is roughly 4 to 6 sentences, about one short scene or a few lines of dialogue. That is enough context for the LLM to answer, and still small enough that one chunk is about one thing.
- The 120 character overlap (20%) keeps a sentence that falls on a boundary readable in at least one chunk.
- Chunks never cross a chapter boundary, so every chunk belongs to exactly one পরিচ্ছেদ and the citation is always exact.
- The comparison below shows 600/120 gives the best Hit@1 and MRR. 300/60 has slightly higher Hit@5 but its chunks are often too short for the LLM to answer from, and 1000/200 loses retrieval accuracy.

Result: 528 chunks, average about 440 characters.

## Embedding model

Model: `BAAI/bge-m3`.

Why:
- It is trained on more than 100 languages including Bengali, and is one of the strongest open models on multilingual retrieval benchmarks (MIRACL, which includes Bengali).
- It supports long inputs (8192 tokens), so a 600 character Bangla chunk is never cut off.
- It runs locally and for free on CPU, no API needed.
- It was clearly the best in my own test on this book (see the bonus section): Hit@5 0.87 compared with 0.84 for multilingual-e5-base and 0.16 for paraphrase-multilingual-MiniLM-L12-v2.

A default English model such as all-MiniLM-L6-v2 cannot read Bangla at all. Even the multilingual MiniLM fails here, which shows why the model choice matters.

## Vector store and retriever

- Vector store: FAISS (`langchain_community.vectorstores.FAISS`) with `MAX_INNER_PRODUCT`. The embeddings are normalised, so the score is the cosine similarity.
- Retriever: `ScoredRetriever` in `rag.py`, a LangChain `BaseRetriever` subclass. It finds the top 8 chunks, keeps each similarity score in the metadata (shown in the app and used for the no-answer check), and widens each hit with the 2 chunks before and after it from the same chapter. Overlapping windows in the same chapter are merged into one passage, so no text is repeated.
- Chain: LangChain LCEL, `ChatPromptTemplate | chat model | StrOutputParser`.

Why widen the hits: in a novel the sentence that best matches a question is often next to the sentence that answers it. For "জয়সিংহ কীভাবে মারা যান?" the best match is the aftermath of his death, and the stabbing itself is two chunks earlier. With plain top 5 chunks the LLM said the cause of death was not given. With widened passages it answers correctly. Small chunks are good for finding the right place, and the wider passage gives the LLM the full scene.

## Citations and no-answer handling

Citations:
- The retrieved chunks are given to the LLM as numbered passages `[1]` to `[5]`, each labelled with its chapter.
- The prompt makes the LLM answer only from those passages and put the passage number after each fact.
- `rag.cite` turns the numbers into chapter names in the answer text and builds the source list: chapter name, printed page range and the Wikisource link for that chapter.
- If the LLM forgets to cite, the bot lists all retrieved chapters and says that no specific citation was given, instead of pointing at one chapter that may be wrong.

No-answer handling has two layers:
1. Score gate: if the best chunk has cosine similarity below 0.45, the bot answers "দুঃখিত, এই প্রশ্নের উত্তর বইটিতে পাওয়া যায়নি।" without calling the LLM. On my evaluation set the lowest top score of a real question was 0.491, while off-topic questions scored between 0.395 and 0.481, so 0.45 never blocks a real question and stops half of the off-topic ones early.
2. Prompt rule: when the question is close to the book but the fact is not there (for example, the year the temple was built), the LLM must reply with exactly that sentence. The code detects it and shows no sources.

## Chat interface

`streamlit run app.py` opens a chat page at http://localhost:8501. Each answer shows the cited chapters with links to Wikisource, and an expander with the retrieved passages and their similarity scores. The sidebar can change k and the score threshold.

## 10 test questions

The questions with expected answers and gold chapters are in `tests/test_questions.json`.

| # | Question | Expected answer | Chapter |
|---|---|---|---|
| 1 | গোবিন্দমাণিক্য কোন রাজ্যের রাজা ছিলেন? | ত্রিপুরার মহারাজা | প্রথম পরিচ্ছেদ (আরও কয়েকটিতে) |
| 2 | মন্দিরের সিঁড়িতে রক্তের দাগ দেখে হাসি রাজাকে কী প্রশ্ন করেছিল? | "এত রক্ত কেন!" | দ্বিতীয় পরিচ্ছেদ |
| 3 | হাসি ও তাতার কাকার নাম কী? | কেদারেশ্বর | দ্বিতীয় পরিচ্ছেদ |
| 4 | রাজা হওয়ার জন্য রঘুপতি নক্ষত্ররায়কে প্রথমে কী আনতে বলেছিলেন? | মায়ের জন্য গোবিন্দমাণিক্যের রক্ত (রাজরক্ত) | পঞ্চম পরিচ্ছেদ |
| 5 | জয়সিংহ কীভাবে মারা যান? | চতুর্দশ দেবতার পূজার রাতে প্রতিমার সামনে নিজের বুকে ছুরি বসিয়ে "রাজরক্ত" দেন | পঞ্চদশ পরিচ্ছেদ |
| 6 | বিচারে রঘুপতিকে কী শাস্তি দেওয়া হয়েছিল? | আট বছরের নির্বাসন (নক্ষত্ররায়েরও) | অষ্টাদশ পরিচ্ছেদ |
| 7 | নক্ষত্ররায় রাজা হয়ে কী নাম গ্রহণ করেন? | ছত্রমাণিক্য | অষ্টাত্রিংশ পরিচ্ছেদ |
| 8 | গোবিন্দমাণিক্যের দুর্গে আসা ফকিরের সঙ্গের তিনটি বালক আসলে কারা ছিল? | শাহ সুজার তিন কন্যা, ছদ্মবেশে | চতুশ্চত্বারিংশ পরিচ্ছেদ, উপসংহার |
| 9 | রবীন্দ্রনাথ ঠাকুর কোন সালে নোবেল পুরস্কার পেয়েছিলেন? | বইয়ে নেই, বট বলবে উত্তর পাওয়া যায়নি | none |
| 10 | ভুবনেশ্বরী মন্দিরটি কোন সালে নির্মিত হয়েছিল? | বইয়ে নেই, বট বলবে উত্তর পাওয়া যায়নি | none |

Question 9 is stopped by the score gate. Question 10 is about the book's own temple, so it passes the gate and tests that the LLM refuses to guess.

Results with Claude Haiku 5.5 (`results/test_results.md`): 10/10. All 8 answerable questions are answered correctly with the right chapter cited, and both questions that are not in the book get "দুঃখিত, এই প্রশ্নের উত্তর বইটিতে পাওয়া যায়নি।" with no sources. PASS is checked automatically from the cited chapters; I also read every answer against the book.

Retrieval check without the LLM: 10/10 (`results/test_results_retrieval.md`).

What I changed after the first full run (8/10): with plain top 5 chunks, the bot said the answer to Q8 was not in the book (the epilogue chunk was ranked 8th) and could not say how Jaisingh died (Q5). Retrieving 8 chunks and widening them with neighbours fixed both. Q4 was reworded because the first version ("রঘুপতি নক্ষত্ররায়কে কী করতে বলেছিলেন?") has several true answers in different chapters.

## Bonus: comparing embedding models and chunking

`compare.py` builds a FAISS index for each setting and scores it on `eval/eval_questions.json`: 31 questions spread over the whole book, each with its gold chapter(s), plus 6 questions that are not in the book. A hit means a chunk from a gold chapter is in the top k. MRR is the mean of 1/rank of the first correct chunk.

| Embedding model | Chunk size / overlap | Hit@1 | Hit@3 | Hit@5 | MRR | Top score (in book) | Top score (not in book) |
|---|---|---|---|---|---|---|---|
| bge-m3 | 600 / 120 | 0.65 | 0.84 | 0.87 | 0.74 | 0.605 | 0.442 |
| bge-m3 | 300 / 60 | 0.61 | 0.87 | 0.90 | 0.73 | 0.614 | 0.436 |
| bge-m3 | 1000 / 200 | 0.65 | 0.77 | 0.84 | 0.71 | 0.603 | 0.442 |
| multilingual-e5-base | 600 / 120 | 0.58 | 0.77 | 0.84 | 0.69 | 0.858 | 0.780 |
| paraphrase-multilingual-MiniLM-L12-v2 | 600 / 120 | 0.10 | 0.10 | 0.16 | 0.11 | 0.884 | 0.806 |

What this shows:
- Embedding model matters most. bge-m3 beats multilingual-e5-base on every metric, and the multilingual MiniLM is close to random on Bangla prose.
- bge-m3 also separates real and off-topic questions much better (0.605 vs 0.442 average top score, a gap of 0.16). For e5 the gap is only 0.08 and for MiniLM 0.08, so a score threshold would not work well with them.
- For chunk size, 600/120 has the best Hit@1 and MRR. 300/60 finds the right chapter a little more often in the top 5 but gives the LLM less context per chunk. 1000/200 is worse because long chunks mix several events.

Full numbers: `results/retrieval_comparison.md` and `results/retrieval_comparison.json`.

## Project structure

```
config.py            paths, book info, model and chunk settings (overridable in .env)
ingest.py            crawl and clean all chapter subpages
chunking.py          normalisation and chunking
embeddings.py        bge-m3 / e5 embeddings through LangChain
build_index.py       build and save the FAISS index
rag.py               retriever, prompt, LLM, citations, no-answer logic
app.py               Streamlit chat interface
run_tests.py         run the 10 test questions
compare.py           bonus hit-rate comparison
data/book.json       the cleaned book text (45 chapters)
tests/               10 test questions with expected answers
eval/                31 + 6 questions for the retrieval comparison
results/             test and comparison reports
```
