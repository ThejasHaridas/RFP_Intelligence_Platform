"""Load a bid folder into ParsedDocuments (one per file), with graceful failure handling."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .html_parser import parse_html
from .metadata import addendum_number, classify_doc_type, parse_date, pdf_metadata_date
from .models import ParsedDocument
from .pdf_parser import parse_pdf

log = logging.getLogger("rfp_intel.ingestion")

SUPPORTED = {".pdf", ".html", ".htm", ".txt", ".md"}


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def parse_file(path: Path, bid_id: str, ocr: bool = True) -> ParsedDocument:
    suffix = path.suffix.lower()
    doc = ParsedDocument(bid_id=bid_id, file_name=path.name, file_path=str(path), doc_type="other", sha1=_sha1(path))
    try:
        if suffix == ".pdf":
            blocks, n, meta, warnings = parse_pdf(path, ocr=ocr)
            doc.blocks, doc.num_pages, doc.warnings = blocks, n, warnings
            first = doc.page_text(1) or (blocks[0].text if blocks else "")
            doc.doc_type = classify_doc_type(path.name, first, is_html=False)
            doc.title = (meta.get("Title") or "").strip() or next((b.text for b in blocks if b.kind == "heading"), None)
            # addenda often quote *new* dates in their body, so prefer the file's own date there
            if doc.doc_type == "addendum":
                doc.doc_date = pdf_metadata_date(meta) or parse_date(first[:1500])
            else:
                doc.doc_date = parse_date(first[:1500]) or pdf_metadata_date(meta)
        elif suffix in (".html", ".htm"):
            blocks, kv, title, warnings = parse_html(path)
            doc.blocks, doc.key_values, doc.title, doc.warnings = blocks, kv, title, warnings
            doc.num_pages = 1
            doc.doc_type = "bid_page"
            date_src = next((v for k, v in kv.items() if k.lower() in ("publication", "issue date", "published", "posted")), "")
            doc.doc_date = parse_date(date_src) if date_src else None
        else:
            from .cleaning import clean_text
            from .models import Block

            text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            doc.blocks = [Block(text=text, page=1)] if text else []
            doc.num_pages = 1
            doc.doc_type = classify_doc_type(path.name, text, is_html=False)
        if doc.doc_type == "addendum":
            doc.addendum_number = addendum_number(path.name, doc.page_text(1))
        if not doc.blocks:
            doc.warnings.append("no text extracted")
    except Exception as exc:  # unreadable / corrupt file -> logged, pipeline continues
        log.exception("failed to parse %s", path)
        doc.warnings.append(f"unreadable file: {type(exc).__name__}: {exc}")
    for w in doc.warnings:
        log.warning("%s/%s: %s", bid_id, path.name, w)
    return doc


def load_bid_folder(folder: Path, bid_id: str | None = None, ocr: bool = True) -> list[ParsedDocument]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"bid folder not found: {folder}")
    bid_id = bid_id or folder.name
    docs = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED and not path.name.startswith("."):
            docs.append(parse_file(path, bid_id, ocr=ocr))
        elif path.is_file():
            log.info("skipping unsupported file %s", path)
    return docs
