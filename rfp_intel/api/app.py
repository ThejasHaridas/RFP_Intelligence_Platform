"""REST API.

    POST /index            {"folder": "./data/Bid1", "bid_id": "Bid1", "force": false}
    GET  /search?q=...&bid_id=Bid1&doc_type=rfp&top_k=5&mode=hybrid
    POST /ask              {"question": "...", "bid_ids": ["Bid1"]}
    POST /extract          {"folder": "./data/Bid1"} or {"bid_id": "Bid1"}
    GET  /bids             catalog of indexed bids
    GET  /health
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ..agents.orchestrator import Orchestrator

app = FastAPI(title="RFP Intelligence Platform", version="1.0.0")


@lru_cache(maxsize=1)
def orchestrator() -> Orchestrator:
    return Orchestrator()


class IndexRequest(BaseModel):
    folder: str
    bid_id: Optional[str] = None
    force: bool = False


class AskRequest(BaseModel):
    question: str
    bid_ids: Optional[list[str]] = None
    top_k: int = 6


class ExtractRequest(BaseModel):
    folder: Optional[str] = None
    bid_id: Optional[str] = None
    force_reindex: bool = False


@app.get("/health")
def health() -> dict[str, Any]:
    o = orchestrator()
    return {"status": "ok", "bids": o.engine.index.bid_ids(), "chunks": len(o.engine.index.chunks),
            "embedder": o.engine.index.embedder.name, "reranker": o.engine.reranker.name,
            "llm": o.llm.provider if o.llm else None}


@app.get("/bids")
def bids() -> list[dict[str, Any]]:
    return orchestrator().engine.index.catalog()


@app.post("/index")
def index(req: IndexRequest) -> dict[str, Any]:
    folder = Path(req.folder)
    if not folder.is_dir():
        raise HTTPException(404, f"folder not found: {folder}")
    return orchestrator().engine.index.index_folder(folder, req.bid_id, req.force).model_dump()


@app.get("/search")
def search(q: str, bid_id: Optional[str] = None, doc_type: Optional[list[str]] = Query(None),
           addendum_number: Optional[int] = None, file_name: Optional[str] = None, top_k: int = 5,
           mode: str = "hybrid", expand: bool = True, rerank: bool = True) -> dict[str, Any]:
    if mode not in ("hybrid", "bm25", "dense"):
        raise HTTPException(400, "mode must be hybrid, bm25 or dense")
    filters = {k: v for k, v in {"bid_id": bid_id, "doc_type": doc_type, "addendum_number": addendum_number,
                                 "file_name": file_name}.items() if v is not None}
    hits = orchestrator().engine.search(q, top_k=top_k, filters=filters or None, mode=mode,  # type: ignore[arg-type]
                                        expand=expand, rerank=rerank)
    return {"query": q, "filters": filters, "results": [h.to_dict() for h in hits]}


@app.post("/ask")
async def ask(req: AskRequest) -> dict[str, Any]:
    o = orchestrator()
    if o.llm is None:
        raise HTTPException(503, "no LLM configured")
    return await o.answer(req.question, req.bid_ids, req.top_k)


@app.post("/extract")
async def extract(req: ExtractRequest) -> dict[str, Any]:
    o = orchestrator()
    if o.llm is None:
        raise HTTPException(503, "no LLM configured")
    if not req.folder and not req.bid_id:
        raise HTTPException(400, "give folder or bid_id")
    return await o.extract_bid(folder=Path(req.folder) if req.folder else None, bid_id=req.bid_id,
                               force_reindex=req.force_reindex)
