"""Persistent, incrementally-updatable search index.

Layout on disk (``RFP_INDEX_DIR``)::

    manifest.json                 # embedder name/dim + list of bid shards
    bids/<bid_id>/chunks.jsonl    # chunk text + metadata
    bids/<bid_id>/vectors.npy     # L2-normalised embeddings (row-aligned with chunks)
    bids/<bid_id>/docs.json       # per-document metadata, warnings, key/values, fingerprint

Each bid is a shard: indexing a new bid folder embeds only that folder; a bid
whose file fingerprints are unchanged is skipped. The in-memory dense index
(FAISS inner-product) and BM25 index are assembled from shards on load; both
are cheap to rebuild compared with embedding.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..ingestion.loader import load_bid_folder
from ..ingestion.models import Chunk, ParsedDocument
from .chunker import chunk_documents
from .embeddings import Embedder, get_embedder, tokenize

log = logging.getLogger("rfp_intel.search")


class IndexReport(BaseModel):
    bid_id: str
    status: str  # indexed | unchanged | failed
    documents: int = 0
    chunks: int = 0
    warnings: list[str] = Field(default_factory=list)
    doc_types: dict[str, str] = Field(default_factory=dict)


def fingerprint(docs: list[ParsedDocument]) -> str:
    return hashlib.sha1(json.dumps(sorted((d.file_name, d.sha1) for d in docs)).encode()).hexdigest()


def folder_fingerprint(folder: Path) -> str:
    from ..ingestion.loader import SUPPORTED, _sha1

    items = sorted((p.name, _sha1(p)) for p in Path(folder).rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("."))
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()


class SearchIndex:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = Path(self.settings.index_dir)
        self.embedder = embedder or get_embedder(self.settings)
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.docs: dict[str, list[dict[str, Any]]] = {}
        self._bm25 = None
        self._faiss = None
        self._lock = threading.RLock()
        self.load()

    # ---------------------------------------------------------------- persistence
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _shard(self, bid_id: str) -> Path:
        return self.root / "bids" / bid_id

    def load(self) -> None:
        with self._lock:
            self.chunks, vecs, self.docs = [], [], {}
            if self.manifest_path.exists():
                manifest = json.loads(self.manifest_path.read_text())
                if manifest.get("embedder") != self.embedder.name:
                    log.warning("index built with %s but current embedder is %s; re-index needed",
                                manifest.get("embedder"), self.embedder.name)
                    return
            shards = sorted((self.root / "bids").glob("*/chunks.jsonl")) if (self.root / "bids").exists() else []
            for cfile in shards:
                shard = cfile.parent
                chunks = [Chunk.model_validate_json(l) for l in cfile.read_text(encoding="utf-8").splitlines() if l]
                v = np.load(shard / "vectors.npy")
                if len(chunks) != len(v):
                    log.error("shard %s is corrupt (chunks/vectors mismatch); skipping", shard.name)
                    continue
                self.chunks.extend(chunks)
                vecs.append(v)
                meta = json.loads((shard / "docs.json").read_text()) if (shard / "docs.json").exists() else {}
                self.docs[shard.name] = meta.get("documents", [])
            self.vectors = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, self.embedder.dim), np.float32)
            self._rebuild_memory_indexes()

    def _rebuild_memory_indexes(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([tokenize(c.embed_text()) for c in self.chunks]) if self.chunks else None
        self._faiss = None
        try:
            import faiss

            if len(self.vectors):
                idx = faiss.IndexFlatIP(self.vectors.shape[1])
                idx.add(self.vectors)
                self._faiss = idx
        except Exception:  # faiss optional; numpy fallback in engine
            self._faiss = None

    def _write_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        bids = sorted(p.name for p in (self.root / "bids").glob("*") if p.is_dir()) if (self.root / "bids").exists() else []
        self.manifest_path.write_text(json.dumps({"embedder": self.embedder.name, "dim": self.embedder.dim,
                                                  "bids": bids}, indent=2))

    # ---------------------------------------------------------------- indexing
    def stored_fingerprint(self, bid_id: str) -> str | None:
        p = self._shard(bid_id) / "docs.json"
        return json.loads(p.read_text()).get("fingerprint") if p.exists() else None

    def index_folder(self, folder: Path, bid_id: str | None = None, force: bool = False) -> IndexReport:
        folder = Path(folder)
        bid_id = bid_id or folder.name
        if not force and self.stored_fingerprint(bid_id) == folder_fingerprint(folder) and \
                any(c.bid_id == bid_id for c in self.chunks):
            return IndexReport(bid_id=bid_id, status="unchanged",
                               documents=len(self.docs.get(bid_id, [])),
                               chunks=sum(c.bid_id == bid_id for c in self.chunks))
        docs = load_bid_folder(folder, bid_id, ocr=self.settings.ocr_enabled)
        return self.index_documents(bid_id, docs)

    def index_documents(self, bid_id: str, docs: list[ParsedDocument]) -> IndexReport:
        s = self.settings
        chunks = chunk_documents(docs, target=s.chunk_target_chars, max_chars=s.chunk_max_chars,
                                 overlap=s.chunk_overlap_chars)
        warnings = [f"{d.file_name}: {w}" for d in docs for w in d.warnings]
        if not chunks:
            return IndexReport(bid_id=bid_id, status="failed", documents=len(docs), warnings=warnings + ["no chunks"])
        vectors = self.embedder.encode([c.embed_text() for c in chunks])
        shard = self._shard(bid_id)
        tmp = shard.with_name(shard.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "chunks.jsonl").write_text("\n".join(c.model_dump_json() for c in chunks), encoding="utf-8")
        np.save(tmp / "vectors.npy", vectors)
        doc_meta = [d.model_dump(exclude={"blocks"}) for d in docs]
        (tmp / "docs.json").write_text(json.dumps({"bid_id": bid_id, "fingerprint": fingerprint(docs),
                                                   "documents": doc_meta}, indent=2), encoding="utf-8")
        with self._lock:  # atomic swap of the shard directory
            shutil.rmtree(shard, ignore_errors=True)
            tmp.rename(shard)
            self._write_manifest()
            self.load()
        return IndexReport(bid_id=bid_id, status="indexed", documents=len(docs), chunks=len(chunks),
                           warnings=warnings, doc_types={d.file_name: d.doc_type for d in docs})

    def remove_bid(self, bid_id: str) -> bool:
        with self._lock:
            shard = self._shard(bid_id)
            if not shard.exists():
                return False
            shutil.rmtree(shard)
            self._write_manifest()
            self.load()
            return True

    # ---------------------------------------------------------------- catalog
    def bid_ids(self) -> list[str]:
        return sorted(self.docs)

    def catalog(self) -> list[dict[str, Any]]:
        """Short description of each indexed bid (used for routing questions to bids)."""
        out = []
        for bid_id, docs in sorted(self.docs.items()):
            kv: dict[str, str] = {}
            for d in docs:
                kv.update(d.get("key_values") or {})
            titles = [d.get("title") for d in docs if d.get("title")]
            out.append({
                "bid_id": bid_id,
                "title": kv.get("Title") or (titles[0] if titles else None),
                "solicitation_number": kv.get("Solicitation Number"),
                "issuer": kv.get("Issuing Organization"),
                "files": [{"file": d["file_name"], "doc_type": d["doc_type"], "addendum_number": d.get("addendum_number"),
                           "doc_date": d.get("doc_date"), "pages": d.get("num_pages"), "warnings": d.get("warnings", [])}
                          for d in docs],
            })
        return out

    @property
    def bm25(self):
        return self._bm25

    @property
    def faiss(self):
        return self._faiss
