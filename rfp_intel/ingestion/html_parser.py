"""HTML bid-page parsing (portal pages such as BidNet, BidSync, state portals).

Generic strategy, not tied to one portal:
* drop scripts, navigation, headers/footers, cookie banners and menus;
* extract label/value pairs from common markup (``*label*`` + sibling body,
  ``<dt>/<dd>``, two-column ``<table>`` rows, ``<th>/<td>``);
* keep remaining content blocks grouped by their nearest heading.
"""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from .cleaning import clean_text, normalize_chars
from .models import Block

_NOISE = re.compile(r"cookie|consent|nav|menu|header|footer|breadcrumb|sidebar|modal|toolbar|share|social|skip|print-options|banner", re.I)
_LABEL_SUFFIXES = ("label", "field-name", "fieldname", "key", "term")


def _is_label_class(token: str) -> bool:
    t = token.lower()
    return t.endswith(_LABEL_SUFFIXES) and not t.startswith(("no-", "no_", "has-", "with-"))


def _txt(el: Tag) -> str:
    return re.sub(r"\s+", " ", normalize_chars(el.get_text(" ", strip=True))).strip()


def _is_noise(el: Tag) -> bool:
    if el.name in ("nav", "header", "footer", "aside", "button", "select"):
        return True
    attrs = " ".join([el.get("id") or "", " ".join(el.get("class") or []), el.get("role") or ""])
    return bool(attrs.strip()) and bool(_NOISE.search(attrs))


def _is_menu(text: str) -> bool:
    lines = [l for l in text.split("\n") if l.strip()]
    short = [l for l in lines if len(l.split()) <= 2]
    return len(lines) >= 3 and len(short) >= len(lines) - 1


def parse_html(path: Path) -> tuple[list[Block], dict[str, str], str | None, list[str]]:
    """Return (blocks, key_values, page_title, warnings)."""
    warnings: list[str] = []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    title = _txt(soup.title) if soup.title else None
    for t in soup(["script", "style", "noscript", "svg", "iframe", "template"]):
        t.decompose()
    total = max(1, len((soup.body or soup).get_text(" ", strip=True)))
    for el in list(soup.find_all(True)):
        # never drop a wrapper that holds most of the page (e.g. a <form> around the notice)
        if el.parent is not None and _is_noise(el) and len(el.get_text(" ", strip=True)) < 0.3 * total:
            el.decompose()

    root = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    kv: dict[str, str] = {}
    blocks: list[Block] = []
    section: str | None = None
    consumed: set[int] = set()

    def add_kv(label: str, value: str) -> None:
        label, value = label.strip().rstrip(":"), value.strip()
        if label and value and len(label) < 80 and label not in kv:
            kv[label] = value

    for el in root.find_all(True):
        if id(el) in consumed or any(id(p) in consumed for p in el.parents):
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            section = _txt(el) or section
            continue
        cls = " ".join(el.get("class") or [])
        # label + body pattern (e.g. <span class="field-label">X</span><div class="field-body">Y</div>)
        if el.name in ("span", "label", "div", "strong", "b") and any(_is_label_class(tok) for tok in cls.split()):
            body = el.find_next_sibling()
            if body is not None:
                label, value = _txt(el), _txt(body)
                if label and value:
                    add_kv(label, value)
                    blocks.append(Block(text=f"{label}: {value}", kind="kv", page=1, section=section))
                elif value:  # unlabeled field (e.g. extra contact lines) -> plain text under section
                    blocks.append(Block(text=value, kind="text", page=1, section=section))
                consumed.add(id(body))
                consumed.add(id(el))
                continue
        if el.name == "dl":
            for dt in el.find_all("dt"):
                dd = dt.find_next_sibling("dd")
                if dd is not None:
                    add_kv(_txt(dt), _txt(dd))
                    blocks.append(Block(text=f"{_txt(dt)}: {_txt(dd)}", kind="kv", page=1, section=section))
            consumed.add(id(el))
            continue
        if el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [_txt(c) for c in tr.find_all(["th", "td"])]
                cells = [c for c in cells if c]
                if len(cells) == 2:
                    add_kv(cells[0], cells[1])
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(Block(text="\n".join(rows), kind="table", page=1, section=section))
            consumed.add(id(el))
            continue
        if el.name in ("p", "li", "pre", "blockquote") or (el.name == "div" and not el.find(True)):
            text = _txt(el)
            if len(text) > 2 and text.lower() not in ("see more", "print", "follow"):
                blocks.append(Block(text=clean_text(text), kind="text", page=1, section=section))
            consumed.add(id(el))

    # merge short consecutive text lines under the same section (contact blocks, tabs)
    merged: list[Block] = []
    for b in blocks:
        if merged and b.kind == "text" and merged[-1].kind == "text" and merged[-1].section == b.section \
                and len(merged[-1].text) + len(b.text) < 400:
            merged[-1].text += "\n" + b.text
        else:
            merged.append(b)
    # drop menu/tab residue: blocks made only of 1-2 word lines ("Notice / Categories / Documents")
    blocks = [b for b in merged if not (b.kind == "text" and _is_menu(b.text))]
    # de-duplicate identical consecutive blocks (portal pages repeat content for responsive layouts)
    deduped: list[Block] = []
    seen: set[str] = set()
    for b in blocks:
        key = b.text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)
    if not deduped:
        text = clean_text(soup.get_text("\n", strip=True))
        if text:
            deduped.append(Block(text=text, kind="text", page=1))
            warnings.append("no structured content found; used full-page text")
        else:
            warnings.append("empty HTML page")
    return deduped, kv, title, warnings
