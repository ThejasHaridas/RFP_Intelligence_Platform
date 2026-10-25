"""Addendum Reconciliation agent.

1. ``analyze``: for every addendum of the bid (ordered by addendum number, then
   date) read its passages and produce a structured change log, mapping each
   change to one of the bid fields when applicable.
2. ``reconcile``: apply the changes to the draft fields in addendum order, so the
   latest valid change wins; keep the previous value in the notes and add the
   addendum page as a source.
"""
from __future__ import annotations

import asyncio
import json
import re

from ..llm.client import LLMError
from .base import GROUNDING_RULES, Agent
from .fields import EXTRACTABLE, FIELD_BY_NAME
from .retrieval import RetrievalAgent, number_evidence
from .schemas import AddendumAnalysisLLM, AddendumChange, FieldResult, RunState, Source

SYSTEM = (
    "You are the Addendum Reconciliation specialist. You read an addendum to a bid/RFP and list every change "
    "or clarification it makes relative to the original solicitation. Map each change to one of the given "
    "bid fields when it affects that field (e.g. a new due date -> 'Due Date'; a warranty clarification -> "
    "'Product Specification'); otherwise use null. For 'modified'/'added' changes give new_value formatted as "
    "the field requires (dates as 'YYYY-MM-DD HH:MM TZ'). Answers to vendor questions that only confirm the "
    "original terms are 'clarification'.\n" + GROUNDING_RULES
)


class AddendumAgent(Agent):
    name = "addendum_reconciliation"

    def addenda(self, bid_id: str) -> list[dict]:
        docs = [d for d in self.ctx.engine.index.docs.get(bid_id, []) if d.get("doc_type") == "addendum"]
        return sorted(docs, key=lambda d: (d.get("addendum_number") or 999, d.get("doc_date") or ""))

    async def analyze(self, state: RunState, bid_id: str) -> list[AddendumChange]:
        docs = self.addenda(bid_id)
        if not docs:
            self.tracer.emit(self.name, "no_addenda", bid_id=bid_id)
            return []
        retriever = RetrievalAgent(self.ctx)
        fields_desc = "\n".join(f"- {f.name}: {f.description}" for f in EXTRACTABLE)

        async def one(doc: dict) -> list[AddendumChange]:
            evs = retriever.document_chunks(bid_id, doc["file_name"])
            pool, _ = number_evidence([evs])
            state.evidence[f"addendum:{doc['file_name']}"] = pool
            if not pool:
                return []
            prompt = (f"Bid: {bid_id}\nAddendum file: {doc['file_name']} (addendum #{doc.get('addendum_number')}, "
                      f"dated {doc.get('doc_date')})\n\nBID FIELDS:\n{fields_desc}\n\nADDENDUM PASSAGES:\n"
                      + "\n\n".join(e.render(1800) for e in pool))
            try:
                out = await self.require_llm().structured(SYSTEM, prompt, AddendumAnalysisLLM, agent=self.name,
                                                          tracer=self.tracer)
            except (LLMError, RuntimeError) as exc:
                state.errors.append(f"addendum analysis failed for {doc['file_name']}: {exc}")
                return []
            by_eid = {e.eid: e for e in pool}
            changes = []
            for c in out.changes:
                field = c.field if c.field in FIELD_BY_NAME else None
                cited = [by_eid[e.strip("[] ")] for e in c.evidence_ids if e.strip("[] ") in by_eid] or pool[:1]
                changes.append(AddendumChange(
                    addendum_number=doc.get("addendum_number"), file=doc["file_name"], field=field,
                    change_type=c.change_type, description=c.description, old_value=c.old_value,
                    new_value=c.new_value,
                    sources=[Source(file=e.file, page=e.page, chunk_id=e.chunk_id) for e in cited]))
            return changes

        results = await asyncio.gather(*(one(d) for d in docs))
        changes = [c for r in results for c in r]
        self.tracer.emit(self.name, "change_log", bid_id=bid_id, addenda=[d["file_name"] for d in docs],
                         changes=[c.model_dump(exclude={"sources"}) for c in changes])
        return changes

    def reconcile(self, state: RunState) -> None:
        """Apply changes in addendum order (latest wins)."""
        ordered = sorted(state.addendum_changes, key=lambda c: (c.addendum_number or 0))
        for change in ordered:
            if not change.field or change.field not in state.drafts:
                continue
            spec = FIELD_BY_NAME[change.field]
            draft = state.drafts[change.field]
            if change.change_type in ("modified", "added") and change.new_value and spec.affected_by_addenda:
                current = _flat(draft.value)
                if _same(current, change.new_value):
                    _add_sources(draft, change.sources)
                    if f"Addendum {change.addendum_number}" not in draft.notes:
                        draft.notes = _join(draft.notes, f"Confirmed by Addendum {change.addendum_number}.")
                else:
                    previous = current or change.old_value
                    if spec.fmt == "list" and isinstance(draft.value, list):
                        draft.value = draft.value + [change.new_value]
                    else:
                        draft.value = change.new_value
                    draft.status = "found"
                    draft.confidence = max(draft.confidence, 0.85)
                    _add_sources(draft, change.sources, front=True)
                    draft.notes = _join(draft.notes, f"Updated by Addendum {change.addendum_number}"
                                        + (f" (original: {previous})" if previous else "") + ".")
                change.applied = True
            elif change.change_type == "clarification":
                draft.notes = _join(draft.notes, f"Addendum {change.addendum_number}: {change.description}")
                _add_sources(draft, change.sources)
                change.applied = True
        state.send(self.name, "orchestrator", "reconciled",
                   applied=sum(c.applied for c in state.addendum_changes), total=len(state.addendum_changes))


def _flat(v) -> str:
    if v is None:
        return ""
    return "; ".join(v) if isinstance(v, list) else str(v)


def _same(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    return bool(a) and (norm(a) == norm(b) or norm(b) in norm(a))


def _join(a: str, b: str) -> str:
    return f"{a} {b}".strip() if a else b


def _add_sources(draft: FieldResult, sources: list[Source], front: bool = False) -> None:
    have = {(s.file, s.page) for s in draft.sources}
    new = [s for s in sources if (s.file, s.page) not in have]
    draft.sources = new + draft.sources if front else draft.sources + new


def change_log_json(changes: list[AddendumChange]) -> list[dict]:
    return [json.loads(c.model_dump_json(exclude={"applied"})) | {"applied": c.applied} for c in changes]
