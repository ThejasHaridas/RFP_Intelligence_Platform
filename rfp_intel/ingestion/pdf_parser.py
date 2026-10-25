"""PDF parsing: text lines + headings (pdfplumber), data tables, OCR fallback (PyMuPDF render).

Per page we:
1. detect real data tables (>= 2 rows with >= 2 filled cells) and render them as
   pipe-delimited rows so specification / bid-information tables stay intact;
2. extract the remaining text line by line, tagging headings (bold / numbered /
   upper-case short lines) so the chunker can split on sections;
3. remove running headers/footers that repeat across pages;
4. OCR pages that have no extractable text (scanned or vector-outlined pages).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .cleaning import clean_lines, find_repeated_lines, is_page_number, line_signature, normalize_chars
from .models import Block

log = logging.getLogger("rfp_intel.ingestion")

_NUMBERED_HEADING = re.compile(r"^(section\s+)?\d+(\.\d+){0,3}\.?\s+[A-Z][A-Za-z]")
_MIN_PAGE_CHARS = 25


def _is_heading(text: str, bold_ratio: float) -> bool:
    t = text.strip()
    if not t or len(t) > 90 or t.endswith((".", ",", ";")) or t.startswith(("•", "-", "(")):
        return False
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 3:
        return False
    digits = sum(c.isdigit() for c in t)
    if digits / len(t) > 0.15 or t.count(",") > 1:
        return False  # spec lines like "256 GB, M.2 2230, ..." are not headings
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return bold_ratio > 0.8 or (upper_ratio > 0.85 and len(letters) >= 5) or bool(_NUMBERED_HEADING.match(t) and len(t) < 70)


def _clean_cell(cell: Any) -> str:
    return re.sub(r"\s+", " ", normalize_chars(str(cell))).strip() if cell is not None else ""


def _table_rows(raw_rows: list[list[Any]]) -> list[list[str]]:
    rows = []
    for r in raw_rows:
        cells = [_clean_cell(c) for c in r]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    return rows


def _is_data_table(rows: list[list[str]]) -> bool:
    multi = [r for r in rows if len(r) >= 2]
    if len(multi) < 2:
        return False
    avg = sum(len(c) for r in multi for c in r) / max(1, sum(len(r) for r in multi))
    return avg < 250  # long-paragraph cells are layout boxes, not tables


def _render_table(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(r) for r in rows)


def _ocr_page(pdf_path: Path, page_index: int) -> str:
    """OCR a page with RapidOCR (ONNX, no system deps). Returns '' if unavailable."""
    try:
        import numpy as np
        import pymupdf
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return ""
    global _OCR
    try:
        _OCR  # type: ignore[used-before-def]
    except NameError:
        _OCR = RapidOCR()
    with pymupdf.open(pdf_path) as doc:
        pix = doc[page_index].get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
    result, _ = _OCR(img)
    if not result:
        return ""
    # result: [box, text, score]; order top-to-bottom, left-to-right, group into lines
    items = sorted(((b[0][1], b[0][0], t) for b, t, s in result if float(s) > 0.5), key=lambda x: (round(x[0] / 15), x[1]))
    lines: list[str] = []
    last_y = None
    for y, _, text in items:
        if last_y is not None and abs(y - last_y) < 15 and lines:
            lines[-1] += " " + text
        else:
            lines.append(text)
        last_y = y
    return "\n".join(lines)


def parse_pdf(path: Path, ocr: bool = True) -> tuple[list[Block], int, dict, list[str]]:
    """Return (blocks, num_pages, pdf_metadata, warnings)."""
    import pdfplumber

    warnings: list[str] = []
    page_lines: list[list[tuple[str, float]]] = []  # (text, bold_ratio) per line
    page_tables: list[list[str]] = []
    meta: dict = {}
    # PyMuPDF is ~50x faster at telling whether a page has a text layer at all;
    # pages without one (scans, outlined glyphs) skip the slow pdfplumber layout pass.
    try:
        import pymupdf

        with pymupdf.open(path) as mdoc:
            has_text = [len(p.get_text().strip()) >= _MIN_PAGE_CHARS for p in mdoc]
    except Exception:
        has_text = []
    with pdfplumber.open(path) as pdf:
        meta = dict(pdf.metadata or {})
        num_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            tables_txt: list[str] = []
            text_page = page
            if has_text and i < len(has_text) and not has_text[i]:
                ocr_text = _ocr_page(path, i) if ocr else ""
                if ocr_text.strip():
                    warnings.append(f"page {i + 1}: no text layer, used OCR")
                else:
                    warnings.append(f"page {i + 1}: empty page (no text layer{'' if ocr else ', OCR disabled'})")
                page_lines.append([(normalize_chars(l), 0.0) for l in ocr_text.splitlines()])
                page_tables.append([])
                continue
            try:
                for t in page.find_tables():
                    rows = _table_rows(t.extract())
                    if _is_data_table(rows):
                        tables_txt.append(_render_table(rows))
                        try:
                            text_page = text_page.outside_bbox(t.bbox)
                        except Exception:
                            pass
            except Exception as exc:  # table detection must never kill the page
                warnings.append(f"page {i + 1}: table extraction failed ({exc})")
            lines: list[tuple[str, float]] = []
            try:
                for ln in text_page.extract_text_lines(return_chars=True, strip=True, y_tolerance=6):
                    chars = ln.get("chars") or []
                    bold = sum("bold" in (c.get("fontname") or "").lower() for c in chars) / max(1, len(chars))
                    lines.append((normalize_chars(ln["text"]), bold))
            except Exception as exc:
                warnings.append(f"page {i + 1}: text extraction failed ({exc})")
            if sum(len(t) for t, _ in lines) + sum(len(t) for t in tables_txt) < _MIN_PAGE_CHARS:
                ocr_text = _ocr_page(path, i) if ocr else ""
                if ocr_text.strip():
                    warnings.append(f"page {i + 1}: no text layer, used OCR")
                    lines = [(normalize_chars(l), 0.0) for l in ocr_text.splitlines()]
                else:
                    warnings.append(f"page {i + 1}: empty page (no text layer{'' if ocr else ', OCR disabled'})")
            page_lines.append(lines)
            page_tables.append(tables_txt)

    repeated = find_repeated_lines([[t for t, _ in lines] for lines in page_lines])
    seen_repeated: set[str] = set()
    blocks: list[Block] = []
    section: str | None = None
    for pno, (lines, tables) in enumerate(zip(page_lines, page_tables), start=1):
        buf: list[str] = []

        def flush() -> None:
            text = "\n".join(clean_lines(buf)).strip()
            if text:
                blocks.append(Block(text=text, kind="text", page=pno, section=section))
            buf.clear()

        for text, bold in lines:
            sig = line_signature(text)
            if sig in repeated:
                if sig in seen_repeated:
                    continue  # running header/footer: keep only first occurrence
                seen_repeated.add(sig)
            if is_page_number(text):
                continue
            if _is_heading(text, bold):
                flush()
                # consecutive heading lines (wrapped headings) merge into one
                if blocks and blocks[-1].kind == "heading" and blocks[-1].page == pno and not buf \
                        and len(blocks[-1].text) + len(text) < 90 and not re.match(r"\s*\d", text):
                    blocks[-1].text += " " + text.strip()
                    section = blocks[-1].text
                    continue
                section = text.strip()
                blocks.append(Block(text=section, kind="heading", page=pno, section=section))
                continue
            buf.append(text)
        flush()
        for t in tables:
            blocks.append(Block(text=t, kind="table", page=pno, section=section))
    return blocks, num_pages, meta, warnings
