"""Orchestrator / Planner and the Ingestion agent.

Extraction flow (all agent hops are Pydantic messages recorded in ``RunState``)::

    ingest -> retrieve evidence (per field, parallel)
           -> extract (3 specialists, parallel)  ||  analyse addenda (parallel)
           -> reconcile addenda -> validate -> [re-retrieve + re-extract failed fields, <= N retries]
           -> summarise -> final JSON + Markdown

Q&A flow: plan sub-queries per bid -> retrieve -> answer with citations -> guardrail.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from ..config import Settings, get_settings
from ..llm.client import BaseLLM, get_llm
from ..observability.tracer import Tracer
from ..search.engine import SearchEngine
from ..search.index import IndexReport, SearchIndex
from .addendum import AddendumAgent
from .base import Agent, AgentContext
from .extraction import ExtractionAgent
from .fields import EXTRACTABLE, FIELD_BY_NAME, GROUPS
from .report import ReportAgent, record_to_markdown
from .retrieval import RetrievalAgent
from .schemas import Evidence, FieldResult, PlanStep, RunState
from .validator import ValidatorAgent

log = logging.getLogger("rfp_intel.agents")


class IngestionAgent(Agent):
    name = "ingestion"

    async def run(self, state: RunState, folder: Path, bid_id: Optional[str] = None, force: bool = False) -> IndexReport:
        report = await asyncio.to_thread(self.ctx.engine.index.index_folder, folder, bid_id, force)
        self.tracer.emit(self.name, "indexed", output=report.model_dump())
        state.send(self.name, "orchestrator", "index_report", **report.model_dump())
        return report


class Orchestrator:
    def __init__(self, settings: Settings | None = None, engine: SearchEngine | None = None,
                 llm: BaseLLM | None = None, use_llm: bool = True) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.engine = engine or SearchEngine(SearchIndex(self.settings), self.settings)
        if llm is None and use_llm:
            try:
                llm = get_llm(self.settings)
            except Exception as exc:  # missing SDK / key: search-only mode still works
                log.warning("LLM unavailable: %s", exc)
                llm = None
        self.llm = llm

    def _ctx(self, tracer: Tracer) -> AgentContext:
        return AgentContext(settings=self.settings, engine=self.engine, llm=self.llm, tracer=tracer)

    def _tracer(self, prefix: str) -> Tracer:
        t = Tracer()
        t.path = Path(self.settings.trace_dir) / f"{prefix}_{t.run_id}.jsonl"
        return t

    # ======================================================================== extraction
    async def extract_bid(self, folder: Optional[Path] = None, bid_id: Optional[str] = None,
                          force_reindex: bool = False) -> dict[str, Any]:
        if folder is None and bid_id is None:
            raise ValueError("give a bid folder or an indexed bid_id")
        bid_id = bid_id or Path(folder).name  # type: ignore[arg-type]
        tracer = self._tracer(f"extract_{bid_id}")
        ctx = self._ctx(tracer)
        state = RunState(run_id=tracer.run_id, mode="extract", goal=f"extract bid {bid_id}", bid_id=bid_id)
        state.plan = [PlanStep(step=s, agent=a) for s, a in [
            ("ingest", "ingestion"), ("retrieve", "retrieval"), ("extract", "extraction.*"),
            ("addenda", "addendum_reconciliation"), ("reconcile", "addendum_reconciliation"),
            ("validate", "validator"), ("retry_failed", "orchestrator"), ("report", "qa_report")]]
        tracer.emit("orchestrator", "plan", goal=state.goal, plan=[s.model_dump() for s in state.plan])

        # 1. ingestion
        if folder is not None:
            state.set_step("ingest", "running")
            try:
                rep = await IngestionAgent(ctx).run(state, Path(folder), bid_id, force_reindex)
                state.set_step("ingest", "done", f"{rep.status}: {rep.documents} docs, {rep.chunks} chunks")
            except Exception as exc:
                state.set_step("ingest", "failed", str(exc))
                state.errors.append(f"ingestion failed: {exc}")
                return self._finish(state, tracer, ctx)
        else:
            state.set_step("ingest", "skipped", "already indexed")
        if bid_id not in self.engine.index.bid_ids():
            state.errors.append(f"bid {bid_id} is not indexed")
            return self._finish(state, tracer, ctx)
        state.catalog = [c for c in self.engine.index.catalog() if c["bid_id"] == bid_id]

        retriever = RetrievalAgent(ctx)
        addendum_agent = AddendumAgent(ctx)

        # 2. evidence per field (parallel)
        state.set_step("retrieve", "running")
        await self._retrieve_fields(state, retriever, [f.name for f in EXTRACTABLE])
        state.set_step("retrieve", "done", f"{sum(len(v) for v in state.evidence.values())} passages")

        # 3. specialists in parallel with addendum analysis
        state.set_step("extract", "running")
        state.set_step("addenda", "running")
        extract_tasks = [self._extract_group(state, ctx, g, [s.name for s in specs]) for g, specs in GROUPS.items()]
        results = await asyncio.gather(addendum_agent.analyze(state, bid_id), *extract_tasks, return_exceptions=True)
        if isinstance(results[0], Exception):
            state.errors.append(f"addendum analysis failed: {results[0]}")
            state.set_step("addenda", "failed", str(results[0]))
        else:
            state.addendum_changes = results[0]
            state.set_step("addenda", "done", f"{len(state.addendum_changes)} changes")
        for r in results[1:]:
            if isinstance(r, Exception):
                state.errors.append(f"extraction group failed: {r}")
        for f in EXTRACTABLE:
            state.drafts.setdefault(f.name, FieldResult(notes="extraction did not run", status="failed"))
        state.set_step("extract", "done")

        # 4. reconcile
        addendum_agent.reconcile(state)
        state.set_step("reconcile", "done", f"{sum(c.applied for c in state.addendum_changes)} applied")

        # 5. validate + feedback loop
        validator = ValidatorAgent(ctx)
        state.set_step("validate", "running")
        state.validation = await validator.validate(state, [f.name for f in EXTRACTABLE])
        state.set_step("validate", "done", f"{sum(not v.passed for v in state.validation.values())} failed")
        state.set_step("retry_failed", "running")
        for attempt in range(1, self.settings.max_field_retries + 1):
            failed = [n for n, v in state.validation.items() if not v.passed]
            if not failed:
                break
            tracer.emit("orchestrator", "retry_round", attempt=attempt, fields=failed)
            state.send("orchestrator", "retrieval", "retry", fields=failed, attempt=attempt)
            feedback = {}
            for n in failed:
                state.retries[n] = state.retries.get(n, 0) + 1
                v = state.validation[n]
                feedback[n] = "; ".join(f"{i.check}: {i.detail}" for i in v.issues if not i.passed)[:400]
            await self._retrieve_fields(state, retriever, failed, extra={n: state.validation[n].retry_query for n in failed},
                                        broaden=True)
            by_group: dict[str, list[str]] = {}
            for n in failed:
                by_group.setdefault(FIELD_BY_NAME[n].group, []).append(n)
            await asyncio.gather(*(self._extract_group(state, ctx, g, names, feedback) for g, names in by_group.items()),
                                 return_exceptions=True)
            addendum_agent.reconcile(_only(state, failed))
            revalidated = await validator.validate(state, failed)
            state.validation.update(revalidated)
            for n in failed:
                state.drafts[n].attempts = state.retries[n]
        still = [n for n, v in state.validation.items() if not v.passed]
        state.set_step("retry_failed", "done", f"{len(still)} still failing" if still else "all fields pass")

        # 6. report
        state.set_step("report", "running")
        state.drafts["Bid Summary"] = await ReportAgent(ctx).summarize(state)
        state.set_step("report", "done")
        return self._finish(state, tracer, ctx)

    async def _retrieve_fields(self, state: RunState, retriever: RetrievalAgent, names: list[str],
                               extra: Optional[dict[str, str]] = None, broaden: bool = False) -> None:
        k = self.settings.evidence_per_field + (3 if broaden else 0)

        async def one(name: str) -> tuple[str, list[Evidence]]:
            spec = FIELD_BY_NAME[name]
            queries = list(spec.queries)
            if extra and extra.get(name):
                queries.insert(0, extra[name])
            if broaden:
                queries.append(f"{name} {spec.description}")
            evs = await retriever.gather(queries, state.bid_id, k, list(spec.doc_types) if spec.doc_types else None)
            return name, evs

        for name, evs in await asyncio.gather(*(one(n) for n in names)):
            state.evidence[name] = evs
            state.send("retrieval", "extraction", "evidence", field=name, chunk_ids=[e.chunk_id for e in evs])

    async def _extract_group(self, state: RunState, ctx: AgentContext, group: str, names: list[str],
                             feedback: Optional[dict[str, str]] = None) -> None:
        agent = ExtractionAgent(ctx, group)
        specs = [FIELD_BY_NAME[n] for n in names]
        res = await agent.extract(state.bid_id or "", specs, {n: state.evidence.get(n, []) for n in names}, feedback)
        for n, r in res.items():
            r.attempts = state.retries.get(n, 0)
            state.drafts[n] = r
        state.send(agent.name, "orchestrator", "draft_fields",
                   fields={n: {"status": r.status, "value": r.value} for n, r in res.items()})

    def _finish(self, state: RunState, tracer: Tracer, ctx: AgentContext) -> dict[str, Any]:
        record = ReportAgent(ctx).build_record(state) if state.drafts else {
            "bid_id": state.bid_id, "fields": {}, "addendum_changes": [], "validation": {}}
        record["run"] = {"run_id": state.run_id, "plan": [s.model_dump() for s in state.plan],
                         "errors": state.errors, "retries": state.retries, **tracer.summary()}
        state.final = record
        tracer.emit("orchestrator", "done", summary=tracer.summary(), errors=state.errors)
        out_dir = Path(self.settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{state.bid_id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        if record.get("fields"):
            (out_dir / f"{state.bid_id}_report.md").write_text(record_to_markdown(record), encoding="utf-8")
        (out_dir / "state").mkdir(exist_ok=True)
        (out_dir / "state" / f"{state.bid_id}_{state.run_id}.json").write_text(state.model_dump_json(indent=2),
                                                                                encoding="utf-8")
        record["trace_file"] = str(tracer.path)
        return record

    # ======================================================================== Q&A
    async def answer(self, question: str, bid_ids: Optional[list[str]] = None, top_k: int = 6) -> dict[str, Any]:
        tracer = self._tracer("qa")
        ctx = self._ctx(tracer)
        state = RunState(run_id=tracer.run_id, mode="qa", goal=question)
        state.catalog = self.engine.index.catalog()
        tracer.emit("orchestrator", "plan", goal=question, plan=["plan_subqueries", "retrieve", "answer", "guardrail"])
        if not state.catalog:
            return {"question": question, "answer": "No bids are indexed yet.", "found": False, "citations": []}
        result = await ReportAgent(ctx).answer(state, question, bid_ids, top_k)
        result["run"] = tracer.summary()
        result["trace_file"] = str(tracer.path)
        tracer.emit("orchestrator", "done", summary=tracer.summary(), answer=result.get("answer"))
        return result


def _only(state: RunState, fields: list[str]) -> RunState:
    """View of the state whose addendum changes touch only ``fields`` (re-apply after re-extraction)."""
    view = state.model_copy()
    view.addendum_changes = [c for c in state.addendum_changes if c.field in fields]
    for c in view.addendum_changes:
        c.applied = False
    return view
