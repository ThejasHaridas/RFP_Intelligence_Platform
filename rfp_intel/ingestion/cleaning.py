"""Text cleaning: header/footer removal, de-hyphenation, line joining, whitespace."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_BULLET = re.compile(r"^\s*([•\-–▪◦●○■□*]|\(?[a-zA-Z0-9]{1,3}[.)]|\d+(\.\d+)+)\s+")
_SENT_END = re.compile(r"[.:;?!]\s*$")
_LEADER = re.compile(r"[.…_·]{4,}|[-–—=]{6,}")
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', " ": " ", "​": "", "": "•"}


def normalize_chars(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for a, b in _QUOTES.items():
        text = text.replace(a, b)
    return text


def normalize_ws(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def line_signature(line: str) -> str:
    """Signature used to detect running headers/footers (digits collapsed)."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def find_repeated_lines(pages: list[list[str]], edge: int = 4, min_ratio: float = 0.5) -> set[str]:
    """Lines that appear in the top/bottom ``edge`` lines of >= ``min_ratio`` of pages."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        edges = [l for l in lines if l.strip()]
        sigs = {line_signature(l) for l in edges[:edge] + edges[-edge:]}
        counts.update(s for s in sigs if s)
    threshold = max(2, int(len(pages) * min_ratio))
    return {s for s, c in counts.items() if c >= threshold}


_PAGE_NUM = re.compile(r"^(page\s*)?#(\s*(of|/|\|)\s*#)?$", re.I)


def is_page_number(line: str) -> bool:
    return bool(_PAGE_NUM.match(line_signature(line)))


def clean_lines(lines: list[str]) -> list[str]:
    """Join wrapped lines into paragraphs and fix hyphenation; keep bullets/items."""
    out: list[str] = []
    for raw in lines:
        line = _LEADER.sub(" ", raw).rstrip()
        if not line.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        line = line.strip()
        if out and out[-1]:
            prev = out[-1]
            if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha() and line[:1].islower():
                out[-1] = prev[:-1] + line  # de-hyphenate "submis-\nsion"
                continue
            continuation = (line[:1].islower() or line[:1] in "(,&") and not _BULLET.match(line)
            if continuation and not _SENT_END.search(prev):
                out[-1] = prev + " " + line
                continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return out


def clean_text(text: str) -> str:
    return normalize_ws("\n".join(clean_lines(normalize_chars(text).splitlines())))
