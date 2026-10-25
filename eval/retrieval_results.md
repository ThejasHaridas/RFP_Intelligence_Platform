| config | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| bm25-only | 0.633 | 0.933 | 0.933 | 1.0 | 0.764 | 0.825 |
| dense-only | 0.7 | 0.867 | 0.933 | 0.967 | 0.793 | 0.82 |
| hybrid (RRF) | 0.7 | 0.933 | 0.967 | 1.0 | 0.819 | 0.858 |
| hybrid + query expansion | 0.7 | 0.933 | 0.967 | 1.0 | 0.818 | 0.856 |
| hybrid + expansion + rerank | 0.8 | 0.967 | 1.0 | 1.0 | 0.873 | 0.899 |

_30 questions · embedder `st:BAAI/bge-small-en-v1.5` · reranker `cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2` · 335 chunks_
