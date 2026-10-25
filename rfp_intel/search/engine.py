"""Hybrid search engine: BM25 + dense vectors, fused with Reciprocal Rank Fusion,
optional query expansion, metadata filtering and re-ranking. Every hit carries a
citation (file, page, chunk id) and the chunk text.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..ingestion.models import Chunk
from .embeddings import tokenize
from .index import SearchIndex
from .query import expand_query
from .rerank import Reranker, get_reranker

log = logging.getLogger("rfp_intel.search")

FilterValue = Union[str, int, list[Any], None]
Mode = Literal["hybrid", "bm25", "dense"]


class SearchFilters(BaseModel):
    bid_id: FilterValue = None
    doc_type: FilterValue = None
    addendum_number: FilterValue = None
    file_name: FilterValue = None
    page_number: FilterValue = None

    def match(self, c: Chunk) -> bool:
        for field, want in self.model_dump(exclude_none=True).items():
            have = getattr(c, field)
            allowed = want if isinstance(want, list) else [want]
            if field == "file_name":
                if not any(str(a).lower() in (have or "").lower() for a in allowed):
                    return False
            elif have not in allowed and str(have) not in [str(a) for a in allowed]:
                return False
        return True


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    rank: int = 0
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def citation(self) -> dict:
        return self.chunk.citation()

    def to_dict(self) -> dict:
        return {"rank": self.rank, "score": round(self.score, 4), "citation": self.citation,
                "section": self.chunk.section, "text": self.chunk.text,
                "scores": {k: round(v, 4) for k, v in self.scores.items()}}


class SearchEngine:
    def __init__(self, index: SearchIndex | None = None, settings: Settings | None = None,
                 reranker: Reranker | None = None) -> None:
        self.settings = settings or get_settings()
        self.index = index or SearchIndex(self.settings)
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = get_reranker(self.settings)
        return self._reranker

    # ------------------------------------------------------------------ primitives
    def _mask(self, filters: SearchFilters | None) -> np.ndarray:
        n = len(self.index.chunks)
        if filters is None or not filters.model_dump(exclude_none=True):
            return np.ones(n, dtype=bool)
        return np.fromiter((filters.match(c) for c in self.index.chunks), dtype=bool, count=n)

    def _bm25_rank(self, query: str, mask: np.ndarray, pool: int) -> list[tuple[int, float]]:
        if self.index.bm25 is None:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        scores = np.asarray(self.index.bm25.get_scores(toks))
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:pool]
        return [(int(i), float(scores[i])) for i in order if np.isfinite(scores[i]) and scores[i] > 0]

    def _dense_rank(self, query: str, mask: np.ndarray, pool: int) -> list[tuple[int, float]]:
        if not len(self.index.vectors):
            return []
        q = self.index.embedder.encode([query], is_query=True)[0]
        if self.index.faiss is not None and mask.all():
            sims, ids = self.index.faiss.search(q[None, :], min(pool, len(self.index.vectors)))
            return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0]
        sims = self.index.vectors @ q
        sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:pool]
        return [(int(i), float(sims[i])) for i in order if np.isfinite(sims[i])]

    # ------------------------------------------------------------------ public API
    def search(self, query: str, top_k: int = 5, filters: SearchFilters | dict | None = None,
               mode: Mode = "hybrid", expand: Optional[bool] = None, rerank: Optional[bool] = None,
               extra_queries: list[str] | None = None) -> list[SearchHit]:
        if isinstance(filters, dict):
            filters = SearchFilters(**filters)
        s = self.settings
        expand = s.query_expansion if expand is None else expand
        rerank = (s.reranker_backend != "none") if rerank is None else rerank
        mask = self._mask(filters)
        if not mask.any():
            return []
        variants = expand_query(query) if expand else [query]
        variants += [q for q in (extra_queries or []) if q and q not in variants]

        fused: dict[int, float] = {}
        parts: dict[int, dict[str, float]] = {}
        for vi, variant in enumerate(variants):
            weight = 1.0 if vi == 0 else 0.8
            lists = []
            if mode in ("hybrid", "bm25"):
                lists.append(("bm25", self._bm25_rank(variant, mask, s.candidate_pool)))
            if mode in ("hybrid", "dense"):
                lists.append(("dense", self._dense_rank(variant, mask, s.candidate_pool)))
            for name, ranked in lists:
                for rank, (idx, raw) in enumerate(ranked, start=1):
                    fused[idx] = fused.get(idx, 0.0) + weight / (s.rrf_k + rank)
                    p = parts.setdefault(idx, {})
                    if vi == 0:
                        p[f"{name}_rank"] = float(rank)
                        p[name] = raw
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        hits = [SearchHit(chunk=self.index.chunks[i], score=sc, scores={**parts.get(i, {}), "rrf": sc})
                for i, sc in ordered[: max(top_k, s.rerank_top_n)]]
        if rerank and hits:
            try:
                hits = self.reranker.rerank(query, hits)
                for h in hits:
                    h.score = h.scores.get("rerank", h.score)
            except Exception as exc:  # reranker failure must not break search
                log.warning("rerank failed: %s", exc)
        hits = hits[:top_k]
        for r, h in enumerate(hits, start=1):
            h.rank = r
        return hits
