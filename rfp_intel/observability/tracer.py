"""Structured JSONL tracing for every agent step and LLM call.

Each event records: run_id, agent, step, input, tool calls, output, tokens and
latency. A trace file is written per run under ``settings.trace_dir``. Token
usage and latency are also aggregated into a per-run summary (cost tracking).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("rfp_intel")

# USD per 1M tokens (input, output); unknown models cost 0 in the summary.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
}


def _truncate(value: Any, limit: int = 1500) -> Any:
    """Keep traces readable: long strings are clipped, structures preserved."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"... [+{len(value) - limit} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(v, limit) for v in value[:50]]
    return value


@dataclass
class Tracer:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    cache_hits: int = 0

    def emit(self, agent: str, step: str, **data: Any) -> dict[str, Any]:
        event = {"ts": time.time(), "run_id": self.run_id, "agent": agent, "step": step, **_truncate(data)}
        with self._lock:
            self.events.append(event)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")
        log.debug("%s.%s %s", agent, step, {k: v for k, v in data.items() if k in ("latency_ms", "error")})
        return event

    def record_llm(self, model: str, tokens_in: int, tokens_out: int, cached: bool = False) -> None:
        with self._lock:
            self.llm_calls += 1
            if cached:
                self.cache_hits += 1
                return
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            pin, pout = PRICES.get(model, (0.0, 0.0))
            self.cost_usd += tokens_in / 1e6 * pin + tokens_out / 1e6 * pout

    @contextmanager
    def span(self, agent: str, step: str, **inputs: Any) -> Iterator[dict[str, Any]]:
        """Time a step; the caller fills ``out`` with output fields."""
        out: dict[str, Any] = {}
        start = time.perf_counter()
        try:
            yield out
        except Exception as exc:
            self.emit(agent, step, input=inputs, error=f"{type(exc).__name__}: {exc}",
                      latency_ms=round((time.perf_counter() - start) * 1000, 1))
            raise
        else:
            self.emit(agent, step, input=inputs, output=out,
                      latency_ms=round((time.perf_counter() - start) * 1000, 1))

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "events": len(self.events),
            "llm_calls": self.llm_calls,
            "llm_cache_hits": self.cache_hits,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "est_cost_usd": round(self.cost_usd, 4),
        }


_NULL = Tracer()


def null_tracer() -> Tracer:
    return _NULL
