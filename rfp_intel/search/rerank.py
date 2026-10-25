"""Re-rankers: cross-encoder (default) or LLM-based; ``none`` keeps fused order."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ..config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover
    from .engine import SearchHit

log = logging.getLogger("rfp_intel.search")


class Reranker:
    name = "none"

    def rerank(self, query: str, hits: list["SearchHit"]) -> list["SearchHit"]:
        return hits


class CrossEncoderReranker(Reranker):
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, max_length=512)
        self.name = f"cross-encoder:{model}"

    def rerank(self, query, hits):
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.embed_text()) for h in hits], show_progress_bar=False)
        for h, s in zip(hits, scores):
            h.scores["rerank"] = float(s)
        return sorted(hits, key=lambda h: h.scores["rerank"], reverse=True)


class _Relevance(BaseModel):
    scores: list[int] = Field(description="relevance 0-10 for each passage, in input order")


class LLMReranker(Reranker):
    """Scores all candidates in one call with the fast model (listwise pointwise-hybrid)."""

    name = "llm"

    def __init__(self, llm) -> None:
        self.llm = llm

    def rerank(self, query, hits):
        import asyncio

        if not hits:
            return hits
        passages = "\n\n".join(f"[{i}] {h.chunk.text[:700]}" for i, h in enumerate(hits))
        prompt = f"Query: {query}\n\nPassages:\n{passages}\n\nReturn one 0-10 relevance score per passage."
        try:
            coro = self.llm.structured("Rate how well each passage answers the query.", prompt, _Relevance,
                                       agent="reranker", fast=True)
            try:
                asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(1) as ex:
                    res = ex.submit(asyncio.run, coro).result()
            except RuntimeError:
                res = asyncio.run(coro)
        except Exception as exc:
            log.warning("LLM rerank failed (%s); keeping fused order", exc)
            return hits
        for h, s in zip(hits, res.scores + [0] * len(hits)):
            h.scores["rerank"] = float(s) + h.scores.get("rrf", 0)  # tie-break by fusion score
        return sorted(hits, key=lambda h: h.scores["rerank"], reverse=True)


_CACHE: dict[str, Reranker] = {}


def get_reranker(settings: Settings | None = None, backend: str | None = None, llm=None) -> Reranker:
    settings = settings or get_settings()
    backend = (backend or settings.reranker_backend).lower()
    if backend in _CACHE:
        return _CACHE[backend]
    rr: Reranker = Reranker()
    try:
        if backend == "cross-encoder":
            rr = CrossEncoderReranker(settings.reranker_model)
        elif backend == "llm":
            from ..llm.client import get_llm

            rr = LLMReranker(llm or get_llm(settings))
    except Exception as exc:
        log.warning("reranker %s unavailable (%s); results keep hybrid fusion order", backend, exc)
        rr = Reranker()
    _CACHE[backend] = rr
    return rr
