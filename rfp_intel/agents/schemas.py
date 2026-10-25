"""Structured messages exchanged between agents + the shared run state.

LLM-facing models (suffix ``LLM``) are what agents ask the model to produce;
the rest are internal agent-to-agent messages. Everything is Pydantic so every
hop is validated.
"""
from __future__ import annotations

import time
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

NOT_FOUND = "Not found in documents"
FieldStatus = Literal["found", "not_found", "failed"]


# --------------------------------------------------------------------------- evidence
class Evidence(BaseModel):
    """A retrieved passage, numbered so the LLM can cite it as [E3]."""

    eid: str
    chunk_id: str
    file: str
    page: int
    doc_type: str
    addendum_number: Optional[int] = None
    doc_date: Optional[str] = None
    section: Optional[str] = None
    text: str
    score: float = 0.0
    query: str = ""

    def render(self, max_chars: int = 1400) -> str:
        tag = f"{self.doc_type}" + (f" #{self.addendum_number}" if self.addendum_number else "")
        head = f"[{self.eid}] ({self.file}, page {self.page}, {tag}" + (f", dated {self.doc_date}" if self.doc_date else "") + ")"
        sec = f"\nSection: {self.section}" if self.section else ""
        return f"{head}{sec}\n{self.text[:max_chars]}"


class Source(BaseModel):
    file: str
    page: int
    chunk_id: Optional[str] = None
    quote: Optional[str] = None


FieldValueType = Union[str, list[str], None]


class FieldResult(BaseModel):
    value: FieldValueType = None
    sources: list[Source] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    status: FieldStatus = "not_found"
    attempts: int = 0


# --------------------------------------------------------------------------- LLM outputs
class ExtractedFieldLLM(BaseModel):
    field: str = Field(description="exact field name")
    value: Optional[Union[str, list[str]]] = Field(description="extracted value, or null if not stated in the evidence")
    evidence_ids: list[str] = Field(default_factory=list, description="ids like E3 of passages that support the value")
    confidence: float = Field(ge=0, le=1, description="0-1 confidence that the value is correct and complete")
    notes: str = Field(default="", description="short justification, conflicts, or why not found")


class ExtractionLLM(BaseModel):
    fields: list[ExtractedFieldLLM]


class AddendumChangeLLM(BaseModel):
    field: Optional[str] = Field(description="affected field name from the list, or null if it affects none of them")
    change_type: Literal["modified", "added", "removed", "clarification"]
    description: str
    old_value: Optional[str] = None
    new_value: Optional[str] = Field(default=None, description="the new value for the field, formatted like the field requires")
    evidence_ids: list[str] = Field(default_factory=list)


class AddendumAnalysisLLM(BaseModel):
    addendum_number: Optional[int] = None
    summary: str
    changes: list[AddendumChangeLLM] = Field(default_factory=list)


class CriticVerdictLLM(BaseModel):
    field: str
    supported: bool = Field(description="true if the cited evidence fully supports the value")
    issue: str = Field(default="", description="what is wrong / missing, empty if supported")
    retry_query: str = Field(default="", description="a better search query to find the right evidence, if unsupported")


class CriticLLM(BaseModel):
    verdicts: list[CriticVerdictLLM]


class SubQuery(BaseModel):
    bid_id: Optional[str] = Field(default=None, description="bid to search, or null for all bids")
    query: str
    doc_types: list[str] = Field(default_factory=list, description="optional doc type filter, e.g. ['addendum']")


class QueryPlanLLM(BaseModel):
    intent: Literal["lookup", "compare", "changes", "summary", "other"]
    bids: list[str] = Field(default_factory=list, description="bid ids the question is about")
    sub_queries: list[SubQuery]


class AnswerLLM(BaseModel):
    found: bool = Field(description="false if the evidence does not contain the answer")
    answer: str = Field(description="answer with inline citations like [E2]; 'Not found in documents' if not found")
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class SummaryLLM(BaseModel):
    summary: str = Field(description="3-6 sentence summary of the bid")


# --------------------------------------------------------------------------- shared state
class AddendumChange(BaseModel):
    addendum_number: Optional[int]
    file: str
    field: Optional[str]
    change_type: str
    description: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    sources: list[Source] = Field(default_factory=list)
    applied: bool = False


class ValidationIssue(BaseModel):
    field: str
    check: str
    passed: bool
    detail: str = ""


class FieldValidation(BaseModel):
    field: str
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    retry_query: str = ""


class PlanStep(BaseModel):
    step: str
    agent: str
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    detail: str = ""


class AgentMessage(BaseModel):
    ts: float = Field(default_factory=time.time)
    sender: str
    recipient: str
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class RunState(BaseModel):
    """Shared memory of one orchestrated run (extraction or Q&A)."""

    run_id: str
    mode: Literal["extract", "qa"]
    goal: str
    bid_id: Optional[str] = None
    plan: list[PlanStep] = Field(default_factory=list)
    catalog: list[dict[str, Any]] = Field(default_factory=list)
    evidence: dict[str, list[Evidence]] = Field(default_factory=dict)  # field / sub-query -> passages
    drafts: dict[str, FieldResult] = Field(default_factory=dict)
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: dict[str, FieldValidation] = Field(default_factory=dict)
    retries: dict[str, int] = Field(default_factory=dict)
    messages: list[AgentMessage] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    final: dict[str, Any] = Field(default_factory=dict)

    def send(self, sender: str, recipient: str, kind: str, **payload: Any) -> None:
        self.messages.append(AgentMessage(sender=sender, recipient=recipient, kind=kind, payload=payload))

    def set_step(self, step: str, status: str, detail: str = "") -> None:
        for s in self.plan:
            if s.step == step:
                s.status = status  # type: ignore[assignment]
                s.detail = detail or s.detail
                return
