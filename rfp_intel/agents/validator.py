"""Validator / Critic agent.

Deterministic checks first (cheap, explainable), then an LLM critic that reads
each value next to the exact passages it cites. A field fails when a rule fails
or the critic says it is unsupported; the orchestrator re-runs retrieval and
extraction for failed fields with the critic's suggested query.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..ingestion.metadata import _DATE_PATTERNS
from ..llm.client import LLMError
from .base import Agent
from .fields import FIELD_BY_NAME
from .schemas import CriticLLM, FieldResult, FieldValidation, RunState, ValidationIssue

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_PHONE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_ISO_DT = re.compile(r"^\d{4}-\d{2}-\d{2}(\s+\d{1,2}:\d{2})?(\s+[A-Z]{2,5})?")
_STOP = set("the and for with from that this are was will shall must any all per not only into upon "
            "within days years year required requirement".split())


def find_dates(text: str) -> set[str]:
    """All dates in text, normalised to YYYY-MM-DD."""
    from datetime import datetime

    out: set[str] = set()
    for rx, kind in _DATE_PATTERNS:
        for m in rx.finditer(text):
            try:
                if kind == "mdy":
                    d = datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))
                elif kind == "ymd":
                    d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                elif kind == "d-mon-y":
                    d = datetime.strptime(f"{m.group(1)} {m.group(2)[:3]} {m.group(3)}", "%d %b %Y")
                else:
                    d = datetime.strptime(f"{m.group(1)[:3]} {m.group(2)} {m.group(3)}", "%b %d %Y")
                out.add(d.strftime("%Y-%m-%d"))
            except ValueError:
                continue
    return out


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9\-./@]*", text.lower()) if len(t) > 2 and t not in _STOP}


def _flat(v) -> str:
    return "; ".join(v) if isinstance(v, list) else (v or "")


def rule_checks(name: str, result: FieldResult, cited_text: str, known_chunks: set[str]) -> list[ValidationIssue]:
    spec = FIELD_BY_NAME[name]
    issues: list[ValidationIssue] = []

    def add(check: str, ok: bool, detail: str = "") -> None:
        issues.append(ValidationIssue(field=name, check=check, passed=ok, detail=detail))

    if result.status == "not_found":
        add("not_found_has_reason", bool(result.notes), "not-found fields must say why")
        return issues
    if result.status == "failed":
        add("extraction_ok", False, result.notes)
        return issues
    value = _flat(result.value)
    add("has_citation", bool(result.sources), "every value needs at least one source")
    unknown = [s.chunk_id for s in result.sources if s.chunk_id and s.chunk_id not in known_chunks]
    add("citation_exists", not unknown, f"unknown chunk ids: {unknown}" if unknown else "")
    if spec.fmt == "summary" or not cited_text:
        return issues
    ct = cited_text.lower()
    ct_compact = re.sub(r"[\s#]", "", ct)
    # identifiers / numbers in the value must appear in the cited text (dates checked separately)
    value_wo_dates = value
    for rx, _ in _DATE_PATTERNS:
        value_wo_dates = rx.sub(" ", value_wo_dates)
    if spec.fmt != "datetime":
        idents = [t for t in re.findall(r"[A-Za-z0-9#][A-Za-z0-9\-./#]*\d[A-Za-z0-9\-./]*", value_wo_dates) if len(t) >= 4]
        missing = [t for t in idents if re.sub(r"[\s#]", "", t.lower()).rstrip(".") not in ct_compact]
        add("identifiers_grounded", not missing, f"not in cited text: {missing[:5]}" if missing else "")
    vd = find_dates(value)
    if vd:
        cd = find_dates(cited_text)
        add("dates_grounded", vd <= cd or bool(vd & cd), f"value dates {sorted(vd)} vs cited {sorted(cd)}")
    vt = _tokens(value_wo_dates)
    if len(vt) >= 3 and spec.allow_none_value != value:
        overlap = len(vt & _tokens(cited_text)) / len(vt)
        add("text_overlap", overlap >= 0.35, f"{overlap:.0%} of value terms found in cited passages")
    if spec.fmt == "datetime":
        add("format_datetime", bool(_ISO_DT.match(value)), "expected 'YYYY-MM-DD HH:MM TZ'")
    if spec.fmt == "contact":
        add("format_contact", bool(_EMAIL.search(value) or _PHONE.search(value)), "expected an email or phone")
    return issues


class ValidatorAgent(Agent):
    name = "validator"

    def cited_text(self, state: RunState, result: FieldResult) -> str:
        chunks = {c.chunk_id: c for c in self.ctx.engine.index.chunks}
        return "\n".join(chunks[s.chunk_id].text for s in result.sources if s.chunk_id in chunks)

    async def validate(self, state: RunState, fields: Iterable[str]) -> dict[str, FieldValidation]:
        known = {c.chunk_id for c in self.ctx.engine.index.chunks}
        fields = list(fields)
        out: dict[str, FieldValidation] = {}
        texts: dict[str, str] = {}
        for name in fields:
            r = state.drafts[name]
            texts[name] = self.cited_text(state, r)
            issues = rule_checks(name, r, texts[name], known)
            out[name] = FieldValidation(field=name, passed=all(i.passed for i in issues), issues=issues)

        # LLM critic over fields that passed the rules and have a value
        to_critique = [n for n in fields if out[n].passed and state.drafts[n].status == "found"
                       and FIELD_BY_NAME[n].fmt != "summary"]
        if to_critique and self.ctx.llm is not None:
            blocks = []
            for n in to_critique:
                r = state.drafts[n]
                blocks.append(f"### {n}\nDefinition: {FIELD_BY_NAME[n].description}\nValue: {_flat(r.value)}\n"
                              f"Notes: {r.notes}\nCited passages:\n{texts[n][:2500]}")
            prompt = ("Check each extracted field against ONLY its cited passages. A field is supported if the "
                      "passages state the value (paraphrase allowed) and it answers the field definition. Values "
                      "updated by an addendum are supported if the addendum passage states the new value. Flag "
                      "values that are invented, wrong, incomplete for the definition, or cite irrelevant "
                      "passages; give a better search query for those.\n\n" + "\n\n".join(blocks))
            try:
                critic = await self.ctx.llm.structured(
                    "You are a strict fact-checking critic for extracted RFP data.", prompt, CriticLLM,
                    agent=self.name, tracer=self.tracer)
                for v in critic.verdicts:
                    if v.field in out:
                        out[v.field].issues.append(ValidationIssue(field=v.field, check="critic_supported",
                                                                   passed=v.supported, detail=v.issue))
                        if not v.supported:
                            out[v.field].passed = False
                            out[v.field].retry_query = v.retry_query
            except (LLMError, RuntimeError) as exc:
                state.errors.append(f"critic unavailable: {exc}")  # rules-only validation still applies

        for name, val in out.items():
            r = state.drafts[name]
            rule_ratio = sum(i.passed for i in val.issues) / max(1, len(val.issues))
            if r.status == "found":
                critic_ok = all(i.passed for i in val.issues if i.check == "critic_supported")
                r.confidence = round(max(0.05, min(0.99, r.confidence * (0.6 + 0.4 * rule_ratio) * (1.0 if critic_ok else 0.6))), 2)
        self.tracer.emit(self.name, "validation", results={n: {"passed": v.passed,
                         "failed_checks": [i.check for i in v.issues if not i.passed]} for n, v in out.items()})
        return out
