"""Extraction specialists. One instance per field group (Dates & Logistics,
Commercial & Legal, Product & Specs); each fills its fields from retrieved
evidence only and returns values with citations."""
from __future__ import annotations

import json
import re

from ..llm.client import LLMError
from .base import GROUNDING_RULES, Agent
from .fields import GROUP_TITLES, FieldSpec
from .retrieval import number_evidence
from .schemas import NOT_FOUND, Evidence, ExtractionLLM, FieldResult, Source

_NULLISH = {"", "null", "none found", "n/a", "not found", "not found in documents", "unknown"}


def normalize_value(value):
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip() and str(v).strip().lower() not in _NULLISH]
        return items or None
    v = str(value).strip()
    return None if v.lower() in _NULLISH else v


def build_extraction_prompt(bid_id: str, specs: list[FieldSpec], pool: list[Evidence],
                            field_hints: dict[str, list[str]], feedback: dict[str, str] | None = None) -> str:
    lines = [f"Bid folder: {bid_id}", "", "FIELDS TO EXTRACT:"]
    for s in specs:
        extra = f" Instructions: {s.instructions}" if s.instructions else ""
        none = f" If the documents explicitly say it does not apply, use \"{s.allow_none_value}\"." if s.allow_none_value else ""
        hint = f" Most relevant passages: {', '.join(field_hints.get(s.name, [])[:8])}." if field_hints.get(s.name) else ""
        fb = f" PREVIOUS ATTEMPT REJECTED: {feedback[s.name]}" if feedback and s.name in feedback else ""
        lines.append(f"- {s.name}: {s.description}{none}{extra}{hint}{fb}")
    lines += ["", "EVIDENCE PASSAGES:"]
    lines += [ev.render() for ev in pool]
    lines += ["", f"Return one entry per field (exactly these names: {json.dumps([s.name for s in specs])})."]
    return "\n".join(lines)


class ExtractionAgent(Agent):
    def __init__(self, ctx, group: str) -> None:
        super().__init__(ctx)
        self.group = group
        self.name = f"extraction.{group}"

    def system_prompt(self) -> str:
        title = GROUP_TITLES.get(self.group, "Extraction specialist")
        return (f"You are the {title} in an RFP analysis team. You fill structured bid fields from "
                f"retrieved passages of government bid documents.\n{GROUNDING_RULES}\n"
                "Confidence: 0.9+ when a passage states the value verbatim, 0.6-0.8 when it requires light "
                "interpretation or combining passages, below 0.5 when uncertain.")

    async def extract(self, bid_id: str, specs: list[FieldSpec], evidence: dict[str, list[Evidence]],
                      feedback: dict[str, str] | None = None) -> dict[str, FieldResult]:
        pool, chunk_to_eid = number_evidence([evidence.get(s.name, []) for s in specs])
        hints = {s.name: [chunk_to_eid[e.chunk_id][0] for e in evidence.get(s.name, []) if e.chunk_id in chunk_to_eid]
                 for s in specs}
        by_eid = {e.eid: e for e in pool}
        results: dict[str, FieldResult] = {}
        if not pool:  # empty retrieval -> explicit not-found, no LLM call
            for s in specs:
                results[s.name] = FieldResult(value=None, confidence=0.3, notes=f"{NOT_FOUND} (no evidence retrieved)",
                                              status="not_found")
            return results
        prompt = build_extraction_prompt(bid_id, specs, pool, hints, feedback)
        try:
            out = await self.require_llm().structured(self.system_prompt(), prompt, ExtractionLLM,
                                                      agent=self.name, tracer=self.tracer)
        except (LLMError, RuntimeError) as exc:
            for s in specs:
                results[s.name] = FieldResult(notes=f"extraction failed: {exc}", status="failed")
            return results

        names = {s.name.lower(): s.name for s in specs}
        for item in out.fields:
            name = names.get(item.field.strip().lower())
            if not name:
                continue
            value = normalize_value(item.value)
            eids = [e.strip().strip("[]") for e in item.evidence_ids]
            cited = [by_eid[e] for e in eids if e in by_eid]
            sources = _dedupe_sources(cited)
            if value is None:
                results[name] = FieldResult(value=None, sources=sources, confidence=item.confidence,
                                            notes=item.notes or NOT_FOUND, status="not_found")
            elif not sources:  # guardrail: a value without a valid citation is not accepted
                results[name] = FieldResult(value=value, confidence=min(item.confidence, 0.3),
                                            notes=f"rejected: no valid citation ({item.notes})", status="failed")
            else:
                results[name] = FieldResult(value=value, sources=sources, confidence=item.confidence,
                                            notes=item.notes, status="found")
        for s in specs:  # model skipped a field
            results.setdefault(s.name, FieldResult(notes="model returned no entry for this field", status="failed"))
        return results


def _dedupe_sources(evs: list[Evidence]) -> list[Source]:
    seen, out = set(), []
    for e in evs:
        key = (e.file, e.page, e.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(Source(file=e.file, page=e.page, chunk_id=e.chunk_id, quote=_snippet(e.text)))
    return out


def _snippet(text: str, n: int = 240) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[:n] + "…"
