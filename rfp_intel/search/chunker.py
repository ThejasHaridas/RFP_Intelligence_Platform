"""Section- and table-aware chunking for RFP documents.

Strategy (see README "Chunking"):
* headings start a new chunk and are carried as ``section`` metadata (and
  prepended to the embedding text), so each chunk knows what part of the RFP it is;
* tables and key/value groups are kept whole (split by rows with the header row
  repeated only if they exceed the max size) because spec / bid-info tables lose
  meaning when cut mid-row;
* prose is packed paragraph-by-paragraph to ~1200 chars (~250-300 tokens) with a
  ~200 char sentence-aligned overlap inside the same section;
* chunks never span pages, so every chunk has an exact page citation.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

from ..ingestion.models import Block, Chunk, ParsedDocument

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'•])")


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for sent in _SENT_SPLIT.split(text):
        while len(sent) > max_chars:  # pathological run-on text
            head, sent = sent[:max_chars], sent[max_chars:]
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(head)
        if len(cur) + len(sent) + 1 > max_chars and cur:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    m = re.search(r"(?<=[.!?;\n])\s*", tail)  # start overlap at a sentence boundary
    return tail[m.end():].strip() if m and m.end() < len(tail) else tail.strip()


def _split_table(text: str, max_chars: int) -> list[str]:
    rows = text.split("\n")
    if len(text) <= max_chars or len(rows) < 3:
        return _split_long(text, max_chars) if len(text) > max_chars * 2 else [text]
    header, out, cur = rows[0], [], [rows[0]]
    for row in rows[1:]:
        if sum(len(r) + 1 for r in cur) + len(row) > max_chars and len(cur) > 1:
            out.append("\n".join(cur))
            cur = [header]
        cur.append(row)
    if len(cur) > 1:
        out.append("\n".join(cur))
    return out


def chunk_document(doc: ParsedDocument, target: int = 1200, max_chars: int = 1800,
                   overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    file_key = hashlib.sha1((doc.sha1 or doc.file_name).encode()).hexdigest()[:8]

    def emit(text: str, page: int, section: str | None, kind: str) -> None:
        text = text.strip()
        if len(text) < 3:
            return
        chunks.append(Chunk(
            chunk_id=f"{doc.bid_id}::{file_key}::p{page}::{len(chunks)}",
            bid_id=doc.bid_id, file_name=doc.file_name, doc_type=doc.doc_type,
            addendum_number=doc.addendum_number, doc_date=doc.doc_date,
            page_number=page, section=section, kind=kind, text=text))  # type: ignore[arg-type]

    buf: list[str] = []
    buf_page: int | None = None
    buf_section: str | None = None
    buf_kind = "text"

    def flush(carry: bool = False) -> str:
        nonlocal buf
        text = "\n".join(buf).strip()
        if text and buf_page is not None:
            emit(text, buf_page, buf_section, buf_kind)
        buf = []
        return _tail_overlap(text, overlap) if carry else ""

    for block in _merge_tiny(doc.blocks):
        if block.kind == "heading":
            flush()
            buf_section = block.section or block.text
            buf_page = block.page
            continue
        if block.kind == "table":
            flush()
            for part in _split_table(block.text, max_chars):
                emit(part, block.page, block.section, "table")
            continue
        new_group = block.page != buf_page or block.section != buf_section or \
            (block.kind == "kv") != (buf_kind == "kv")
        if new_group:
            flush()
            buf_page, buf_section, buf_kind = block.page, block.section, block.kind
        for para in (p for p in block.text.split("\n") if p.strip()):
            for piece in _split_long(para, max_chars):
                size = sum(len(x) + 1 for x in buf)
                if buf and size + len(piece) > target:
                    carry = flush(carry=buf_kind == "text")
                    buf = [carry] if carry else []
                buf.append(piece)
    flush()
    return chunks


def _merge_tiny(blocks: Iterable[Block], min_chars: int = 60) -> list[Block]:
    """Attach very small text blocks to the following text block on the same page."""
    out: list[Block] = []
    pending: Block | None = None
    for b in blocks:
        if pending is not None:
            if b.kind == "text" and b.page == pending.page:
                b = b.model_copy(update={"text": pending.text + "\n" + b.text})
            else:
                out.append(pending)
            pending = None
        if b.kind == "text" and len(b.text) < min_chars:
            pending = b
            continue
        out.append(b)
    if pending is not None:
        out.append(pending)
    return out


def chunk_documents(docs: Iterable[ParsedDocument], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, **kw))
    return out
