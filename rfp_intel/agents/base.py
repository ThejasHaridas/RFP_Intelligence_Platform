from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import Settings
from ..llm.client import BaseLLM
from ..observability.tracer import Tracer
from ..search.engine import SearchEngine


@dataclass
class AgentContext:
    """Dependencies shared by all agents (injected, so tests can swap the LLM)."""

    settings: Settings
    engine: SearchEngine
    llm: Optional[BaseLLM]
    tracer: Tracer


class Agent:
    name = "agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    @property
    def tracer(self) -> Tracer:
        return self.ctx.tracer

    def require_llm(self) -> BaseLLM:
        if self.ctx.llm is None:
            raise RuntimeError("No LLM configured: set LLM_PROVIDER and the provider's API key")
        return self.ctx.llm


GROUNDING_RULES = """Rules:
- Use ONLY the numbered evidence passages. Never use outside knowledge or guess.
- Every non-null value must cite at least one evidence id (e.g. "E3") that directly states it.
- If the evidence does not contain the information, return value null and say "Not found in documents" in notes.
- Addenda override the original RFP and portal page: when they conflict, use the latest addendum and mention the original in notes.
- Copy identifiers, model numbers, SKUs, emails and phone numbers exactly as written."""
