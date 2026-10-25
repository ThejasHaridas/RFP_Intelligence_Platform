"""Retrieval evaluation: Recall@k, MRR@10 and nDCG@10 across search configurations.

A hit is relevant when it comes from the expected file (substring match, empty =
any file of the bid) and its text contains one of the expected answer phrases
(whitespace/case-insensitive). Phrase-based labels stay valid when chunking
parameters change, unlike chunk-id labels.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .search.engine import SearchEngine, SearchHit


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def is_relevant(hit: SearchHit, item: dict) -> bool:
    if item.get("file") and item["file"].lower() not in hit.chunk.file_name.lower():
        return False
    text = _norm(hit.chunk.text)
    return any(_norm(p) in text for p in item["phrases"])


@dataclass
class Config:
    name: str
    mode: str = "hybrid"
    expand: bool = False
    rerank: bool = False


DEFAULT_CONFIGS = [
    Config("bm25-only", mode="bm25"),
    Config("dense-only", mode="dense"),
    Config("hybrid (RRF)", mode="hybrid"),
    Config("hybrid + query expansion", mode="hybrid", expand=True),
    Config("hybrid + expansion + rerank", mode="hybrid", expand=True, rerank=True),
]


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def evaluate(engine: SearchEngine, questions: list[dict], configs: list[Config] | None = None,
             ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, Any]:
    configs = configs or DEFAULT_CONFIGS
    results: dict[str, Any] = {"configs": [], "per_question": {}}
    for cfg in configs:
        if cfg.rerank and engine.reranker.name == "none":
            results["configs"].append({"config": cfg.name, "skipped": "no reranker available"})
            continue
        recall = {k: 0.0 for k in ks}
        mrr = ndcg = 0.0
        for item in questions:
            hits = engine.search(item["question"], top_k=max(ks), filters={"bid_id": item["bid_id"]},
                                 mode=cfg.mode, expand=cfg.expand, rerank=cfg.rerank)  # type: ignore[arg-type]
            rels = [is_relevant(h, item) for h in hits]
            first = next((i for i, r in enumerate(rels) if r), None)
            for k in ks:
                recall[k] += 1.0 if first is not None and first < k else 0.0
            mrr += 1.0 / (first + 1) if first is not None else 0.0
            dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rels[:10]) if r)
            idcg = sum(1 / math.log2(i + 2) for i in range(min(10, max(1, sum(rels)))))
            ndcg += dcg / idcg if idcg else 0.0
            results["per_question"].setdefault(item["id"], {})[cfg.name] = first + 1 if first is not None else None
        n = len(questions)
        row = {"config": cfg.name, **{f"Recall@{k}": round(recall[k] / n, 3) for k in ks},
               "MRR@10": round(mrr / n, 3), "nDCG@10": round(ndcg / n, 3)}
        results["configs"].append(row)
    results["meta"] = {"questions": len(questions), "embedder": engine.index.embedder.name,
                       "reranker": engine.reranker.name, "chunks": len(engine.index.chunks)}
    return results


def to_markdown(results: dict[str, Any]) -> str:
    rows = [r for r in results["configs"] if "skipped" not in r]
    if not rows:
        return "no results"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    for r in results["configs"]:
        if "skipped" in r:
            lines.append(f"\n_{r['config']}: skipped ({r['skipped']})_")
    m = results.get("meta", {})
    lines.append(f"\n_{m.get('questions')} questions · embedder `{m.get('embedder')}` · "
                 f"reranker `{m.get('reranker')}` · {m.get('chunks')} chunks_")
    return "\n".join(lines)
