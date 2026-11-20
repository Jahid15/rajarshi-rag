# Retrieval comparison (31 questions with a known gold chapter)

A hit means a chunk from the correct chapter appears in the top k results.

| Embedding model | Chunk size / overlap | Hit@1 | Hit@3 | Hit@5 | MRR | Top score (in book) | Top score (not in book) |
|---|---|---|---|---|---|---|---|
| bge-m3 | 600 / 120 | 0.65 | 0.84 | 0.87 | 0.74 | 0.605 | 0.442 |
| bge-m3 | 300 / 60 | 0.61 | 0.87 | 0.90 | 0.73 | 0.614 | 0.436 |
| bge-m3 | 1000 / 200 | 0.65 | 0.77 | 0.84 | 0.71 | 0.603 | 0.442 |
| multilingual-e5-base | 600 / 120 | 0.58 | 0.77 | 0.84 | 0.69 | 0.858 | 0.780 |
| paraphrase-multilingual-MiniLM-L12-v2 | 600 / 120 | 0.10 | 0.10 | 0.16 | 0.11 | 0.884 | 0.806 |
