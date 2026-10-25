"""Q&A / Report agent: answers questions with citations, writes the bid summary,
and renders the final JSON record and a human-readable Markdown report."""
from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

from ..llm.client import LLMError
from .base import GROUNDING_RULES, Agent
from .fields import FIELDS
from .retrieval import RetrievalAgent, number_evidence
from .schemas import NOT_FOUND, AnswerLLM, FieldResult, QueryPlanLLM, RunState, Source, SummaryLLM, SubQuery

PLANNER_SYSTEM = (
    "You plan retrieval for questions about government bids. Given the catalog of indexed bids and a question, "
    "decide which bid(s) it is about and write 1-4 focused search sub-queries (one per bid for comparisons; "
    "use doc_types ['addendum'] for questions about addendum changes). Use bid ids exactly as in the catalog. "
    "If the question names no bid and the catalog has several, include all relevant bids."
)

ANSWER_SYSTEM = (
    "You answer questions about bid/RFP documents for a sales team, using ONLY the numbered evidence passages. "
    "Cite every claim inline with evidence ids like [E2]. If the evidence does not contain the answer, set "
    "found=false and answer exactly 'Not found in documents'. For comparisons, answer per bid. Mention when an "
    "addendum changed the original value.\n" + GROUNDING_RULES
)


class ReportAgent(Agent):
    name = "qa_report"

    # ---------------------------------------------------------------- Q&A mode
    async def plan_question(self, state: RunState, question: str, bid_ids: Optional[list[str]]) -> QueryPlanLLM:
        catalog = [{k: c.get(k) for k in ("bid_id", "title", "solicitation_number", "issuer")} | {
            "files": [f["file"] for f in c.get("files", [])]} for c in state.catalog]
        if bid_ids:
            catalog = [c for c in catalog if c["bid_id"] in bid_ids] or catalog
        if len(catalog) == 1 or self.ctx.llm is None:
            only = [c["bid_id"] for c in catalog]
            return QueryPlanLLM(intent="lookup", bids=only,
                                sub_queries=[SubQuery(bid_id=b, query=question) for b in only] or [SubQuery(query=question)])
        try:
            plan = await self.ctx.llm.structured(PLANNER_SYSTEM, f"CATALOG: {catalog}\n\nQUESTION: {question}",
                                                 QueryPlanLLM, agent=self.name + ".planner", tracer=self.tracer, fast=True)
            valid = {c["bid_id"] for c in catalog}
            plan.sub_queries = [q for q in plan.sub_queries if q.bid_id in valid or q.bid_id is None] or \
                [SubQuery(bid_id=b, query=question) for b in valid]
            return plan
        except (LLMError, RuntimeError) as exc:
            state.errors.append(f"planner failed: {exc}")
            return QueryPlanLLM(intent="other", bids=[c["bid_id"] for c in catalog],
                                sub_queries=[SubQuery(bid_id=c["bid_id"], query=question) for c in catalog])

    async def answer(self, state: RunState, question: str, bid_ids: Optional[list[str]] = None,
                     top_k: int = 6) -> dict[str, Any]:
        retriever = RetrievalAgent(self.ctx)
        plan = await self.plan_question(state, question, bid_ids)
        state.send(self.name, "retrieval", "sub_queries", plan=plan.model_dump())
        rewrites = await retriever.rewrite(question)

        async def run(sq: SubQuery):
            queries = [sq.query] + ([question] if sq.query != question else []) + rewrites[:2]
            return await retriever.gather(queries, sq.bid_id, top_k, sq.doc_types or None)

        pools = await asyncio.gather(*(run(sq) for sq in plan.sub_queries))
        for sq, evs in zip(plan.sub_queries, pools):
            state.evidence[f"{sq.bid_id}:{sq.query}"] = evs
        pool, _ = number_evidence(list(pools))
        if not pool:
            return {"question": question, "answer": NOT_FOUND, "found": False, "citations": [], "confidence": 0.0,
                    "plan": plan.model_dump()}
        prompt = f"QUESTION: {question}\n\nEVIDENCE:\n" + "\n\n".join(
            f"(bid {_bid_of(e.chunk_id)}) " + e.render() for e in pool)
        try:
            out = await self.require_llm().structured(ANSWER_SYSTEM, prompt, AnswerLLM, agent=self.name,
                                                      tracer=self.tracer)
        except (LLMError, RuntimeError) as exc:
            return {"question": question, "answer": f"Error: {exc}", "found": False, "citations": [],
                    "confidence": 0.0, "plan": plan.model_dump()}
        by_eid = {e.eid: e for e in pool}
        inline = set(re.findall(r"\[(E\d+)\]", out.answer))
        cited = [by_eid[e] for e in sorted(set(out.evidence_ids) | inline, key=lambda x: int(x[1:])) if e in by_eid]
        found = out.found and bool(cited)  # guardrail: an uncited answer is treated as not found
        return {
            "question": question,
            "answer": out.answer if found else NOT_FOUND,
            "found": found,
            "confidence": out.confidence if found else min(out.confidence, 0.5),
            "citations": [{"id": e.eid, "bid_id": _bid_of(e.chunk_id), "file": e.file, "page": e.page,
                           "doc_type": e.doc_type, "quote": e.text[:300]} for e in cited],
            "plan": plan.model_dump(),
        }

    # ---------------------------------------------------------------- extraction mode
    async def summarize(self, state: RunState) -> FieldResult:
        found = {n: r for n, r in state.drafts.items() if r.status == "found" and n != "Bid Summary"}
        if not found or self.ctx.llm is None:
            return FieldResult(value=None, notes="insufficient extracted data for a summary", status="not_found")
        facts = "\n".join(f"- {n}: {_flat(r.value)[:500]}" for n, r in found.items())
        changes = "\n".join(f"- Addendum {c.addendum_number}: {c.description}" for c in state.addendum_changes
                            if c.change_type != "clarification")[:2000]
        prompt = (f"Write a 3-6 sentence summary of bid {state.bid_id} for a sales team deciding whether to bid: "
                  f"who is buying what, quantities, key deadline, submission method, notable requirements and "
                  f"addendum changes. Use only these validated facts.\n\nFACTS:\n{facts}\n\nADDENDUM CHANGES:\n{changes or 'none'}")
        try:
            out = await self.ctx.llm.structured("You write concise, factual bid summaries.", prompt, SummaryLLM,
                                                agent=self.name + ".summary", tracer=self.tracer)
        except (LLMError, RuntimeError) as exc:
            return FieldResult(value=None, notes=f"summary failed: {exc}", status="failed")
        key = ["Title", "company_name", "Due Date", "Product", "Bid Submission Type"]
        sources: list[Source] = []
        for k in key:
            if k in found:
                sources.extend(s for s in found[k].sources[:1] if (s.file, s.page) not in {(x.file, x.page) for x in sources})
        conf = round(sum(found[k].confidence for k in key if k in found) / max(1, sum(k in found for k in key)), 2)
        return FieldResult(value=out.summary, sources=sources, confidence=conf,
                           notes="Generated from validated fields", status="found")

    def build_record(self, state: RunState) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        passed = failed = not_found = 0
        for spec in FIELDS:
            r = state.drafts.get(spec.name) or FieldResult(notes=NOT_FOUND)
            v = state.validation.get(spec.name)
            if r.status == "not_found":
                not_found += 1
            elif v is not None and not v.passed or r.status == "failed":
                failed += 1
            else:
                passed += 1
            notes = r.notes or ("" if r.value is not None else NOT_FOUND)
            if r.status == "failed" or (v is not None and not v.passed):
                failed_checks = [f"{i.check}: {i.detail}".strip(": ") for i in (v.issues if v else []) if not i.passed]
                notes = (notes + " | VALIDATION FAILED: " + "; ".join(failed_checks)).strip(" |")
            fields[spec.name] = {
                "value": r.value,
                "sources": [{"file": s.file, "page": s.page} for s in r.sources],
                "confidence": round(r.confidence, 2),
                "notes": notes,
            }
        return {"bid_id": state.bid_id, "fields": fields,
                "addendum_changes": [{"addendum": c.addendum_number, "file": c.file, "field": c.field,
                                      "change_type": c.change_type, "description": c.description,
                                      "old_value": c.old_value, "new_value": c.new_value, "applied": c.applied,
                                      "sources": [{"file": s.file, "page": s.page} for s in c.sources]}
                                     for c in state.addendum_changes],
                "validation": {"passed": passed, "failed": failed, "not_found": not_found}}


def _flat(v) -> str:
    return "; ".join(v) if isinstance(v, list) else (v or "")


def _bid_of(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def record_to_markdown(record: dict[str, Any]) -> str:
    lines = [f"# Bid report: {record['bid_id']}", ""]
    summary = record["fields"].get("Bid Summary", {}).get("value")
    if summary:
        lines += [summary, ""]
    v = record.get("validation", {})
    lines += [f"Validation: {v.get('passed', 0)} passed, {v.get('failed', 0)} failed, {v.get('not_found', 0)} not found.", "",
              "| Field | Value | Confidence | Sources |", "|---|---|---|---|"]
    for name, f in record["fields"].items():
        if name == "Bid Summary":
            continue
        val = _flat(f["value"]) if f["value"] is not None else "_Not found in documents_"
        val = val.replace("|", "/").replace("\n", "<br>")
        src = "; ".join(f"{s['file']} p.{s['page']}" for s in f["sources"]) or "-"
        lines.append(f"| {name} | {val} | {f['confidence']:.2f} | {src} |")
    if record.get("addendum_changes"):
        lines += ["", "## Addendum changes", ""]
        for c in record["addendum_changes"]:
            lines.append(f"- **Addendum {c['addendum']}** ({c['change_type']}{', ' + c['field'] if c['field'] else ''}): "
                         f"{c['description']}")
    return "\n".join(lines) + "\n"
