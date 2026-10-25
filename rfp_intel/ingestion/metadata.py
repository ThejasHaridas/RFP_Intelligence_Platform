"""Document-level metadata inference (doc type, addendum number, document date)."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from .models import DocType

_ADDENDUM_NUM = re.compile(r"\baddend(?:um|a)\b\s*(?:no\.?|number|#)?\s*(\d{1,2})", re.I)
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_DATE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "mdy"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "ymd"),
    (re.compile(rf"\b(\d{{1,2}})-({_MONTHS})[a-z]*-(\d{{4}})\b", re.I), "d-mon-y"),
    (re.compile(rf"\b({_MONTHS})[a-z]*\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I), "mon d y"),
]


def classify_doc_type(file_name: str, first_page_text: str, is_html: bool) -> DocType:
    """Rule-based classifier using the file name first and then first-page content.

    Rules are generic RFP vocabulary (not tied to the sample bids).
    """
    name = file_name.lower()
    head = first_page_text[:1500].lower()
    if is_html:
        return "bid_page"
    if re.search(r"addend|amendment", name) or re.search(r"^\s*(addendum|amendment)\b", head) or \
            re.search(r"\baddendum\s+(no\.?|number|#)\s*\d", head[:400]):
        return "addendum"
    if "affidavit" in name or re.search(r"\baffidavit\b", head[:300]):
        return "affidavit"
    if re.search(r"spec|datasheet|data sheet|quote|configuration", name) or \
            re.search(r"\b(sku|part number|specifications?)\b", head[:300]):
        return "specs"
    if re.search(r"\b(w-?9|form|certificate|questionnaire)\b", name):
        return "form"
    if re.search(r"rfp|rfq|ifb|bid|solicitation|proposal|porfp|itb|final", name) or \
            re.search(r"request for (proposal|quot|bid)|invitation (to|for) bid|solicitation", head):
        return "rfp"
    return "other"


def addendum_number(file_name: str, first_page_text: str) -> Optional[int]:
    for text in (file_name, first_page_text[:600]):
        m = _ADDENDUM_NUM.search(text)
        if m:
            return int(m.group(1))
    return None


def parse_date(text: str) -> Optional[str]:
    """Return the first date found in ``text`` as ISO yyyy-mm-dd."""
    best: tuple[int, str] | None = None
    for rx, kind in _DATE_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        try:
            if kind == "mdy":
                d = datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))
            elif kind == "ymd":
                d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif kind == "d-mon-y":
                d = datetime.strptime(f"{m.group(1)} {m.group(2)[:3]} {m.group(3)}", "%d %b %Y")
            else:
                d = datetime.strptime(f"{m.group(1)[:3]} {m.group(2)} {m.group(3)}", "%b %d %Y")
        except ValueError:
            continue
        if best is None or m.start() < best[0]:
            best = (m.start(), d.strftime("%Y-%m-%d"))
    return best[1] if best else None


def pdf_metadata_date(meta: dict) -> Optional[str]:
    for key in ("creationDate", "modDate", "CreationDate", "ModDate"):
        raw = (meta or {}).get(key) or ""
        m = re.search(r"(\d{4})(\d{2})(\d{2})", raw)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None
