"""Retrieval / Search agent: query rewriting + multi-query hybrid search with
metadata filters, re-ranking and evidence de-duplication. Agents never see whole
documents, only the passages this agent returns."""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from ..ingestion.models import Chunk
from ..search.query import REWRITE_SYSTEM, QueryRewrite
from .base import Agent
from .schemas import Evidence


class RetrievalAgent(Agent):
    name = "retrieval"

    async def rewrite(self, question: str) -> list[str]:
        """LLM query rewriting (fast model). Falls back to [] on any failure."""
        if self.ctx.llm is None:
            return []
        try:
            out = await self.ctx.llm.structured(REWRITE_SYSTEM, question, QueryRewrite, agent=self.name,
                                                tracer=self.tracer, fast=True)
            return [q for q in out.queries + ([" ".join(out.keywords)] if out.keywords else []) if q][:4]
        except Exception as exc:
            self.tracer.emit(self.name, "rewrite_failed", error=str(exc))
            return []

    async def search(self, query: str, bid_id: Optional[str], top_k: int,
                     doc_types: Optional[list[str]] = None, rerank: Optional[bool] = None) -> list[Evidence]:
        filters: dict = {}
        if bid_id:
            filters["bid_id"] = bid_id
        if doc_types:
            filters["doc_type"] = list(doc_types)
        start = time.perf_counter()
        hits = await asyncio.to_thread(self.ctx.engine.search, query, top_k, filters or None, "hybrid", None, rerank)
        self.tracer.emit(self.name, "tool_call", tool="search_engine.search",
                         input={"query": query, "filters": filters, "top_k": top_k},
                         output=[{"chunk_id": h.chunk.chunk_id, "file": h.chunk.file_name, "page": h.chunk.page_number,
                                  "score": round(h.score, 4)} for h in hits],
                         latency_ms=round((time.perf_counter() - start) * 1000, 1))
        return [_to_evidence(h.chunk, h.score, query) for h in hits]

    async def gather(self, queries: list[str], bid_id: Optional[str], top_k: int,
                     doc_types: Optional[list[str]] = None, per_query_k: int = 6) -> list[Evidence]:
        """Run several queries (and an optional doc-type-filtered pass), fuse by
        reciprocal rank and return the top ``top_k`` unique passages."""
        tasks = [self.search(q, bid_id, per_query_k) for q in queries]
        if doc_types:
            tasks += [self.search(q, bid_id, max(2, per_query_k // 2), doc_types) for q in queries[:2]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        fused: dict[str, tuple[float, Evidence]] = {}
        for res in results:
            if isinstance(res, Exception):
                self.tracer.emit(self.name, "search_failed", error=str(res))
                continue
            for rank, ev in enumerate(res, start=1):
                score = 1.0 / (60 + rank)
                prev = fused.get(ev.chunk_id)
                fused[ev.chunk_id] = (score + (prev[0] if prev else 0.0), prev[1] if prev else ev)
        ranked = sorted(fused.values(), key=lambda t: t[0], reverse=True)[:top_k]
        out = []
        for score, ev in ranked:
            ev.score = round(score, 5)
            out.append(ev)
        return out

    def document_chunks(self, bid_id: str, file_name: str, limit: int = 25) -> list[Evidence]:
        """All chunks of one document in reading order (used for short docs like addenda)."""
        chunks = [c for c in self.ctx.engine.index.chunks if c.bid_id == bid_id and c.file_name == file_name]
        chunks.sort(key=lambda c: (c.page_number, int(c.chunk_id.rsplit("::", 1)[-1])))
        self.tracer.emit(self.name, "tool_call", tool="index.document_chunks",
                         input={"bid_id": bid_id, "file": file_name}, output={"chunks": len(chunks[:limit])})
        return [_to_evidence(c, 0.0, f"document:{file_name}") for c in chunks[:limit]]


def _to_evidence(c: Chunk, score: float, query: str) -> Evidence:
    return Evidence(eid="", chunk_id=c.chunk_id, file=c.file_name, page=c.page_number, doc_type=c.doc_type,
                    addendum_number=c.addendum_number, doc_date=c.doc_date, section=c.section, text=c.text,
                    score=score, query=query)


def number_evidence(pools: list[list[Evidence]], prefix: str = "E") -> tuple[list[Evidence], dict[str, list[str]]]:
    """Merge evidence lists into one numbered pool. Returns (pool, chunk_id -> eids)."""
    pool: list[Evidence] = []
    by_chunk: dict[str, str] = {}
    for evs in pools:
        for ev in evs:
            if ev.chunk_id in by_chunk:
                continue
            eid = f"{prefix}{len(pool) + 1}"
            by_chunk[ev.chunk_id] = eid
            pool.append(ev.model_copy(update={"eid": eid}))
    return pool, {k: [v] for k, v in by_chunk.items()}
