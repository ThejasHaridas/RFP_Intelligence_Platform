"""Provider-agnostic LLM client with structured (Pydantic) outputs.

* ``AnthropicLLM`` uses the Anthropic SDK's structured outputs (``messages.parse``).
* ``OpenAICompatLLM`` covers OpenAI and any OpenAI-compatible server (OpenRouter, NVIDIA API catalog, Ollama, vLLM).
* ``ScriptedLLM`` returns canned responses; used by tests and offline demos.

Every call goes through ``BaseLLM.structured`` which adds: a disk cache keyed by
the full prompt (semantic-cache-lite: identical requests are free), retries with
exponential backoff on timeouts / transient errors, JSON repair when a provider
returns malformed JSON, Pydantic validation, and token/latency tracing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import Settings, get_settings
from ..observability.tracer import Tracer, null_tracer

log = logging.getLogger("rfp_intel.llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM fails after all retries; callers degrade gracefully."""


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model reply (handles fences / prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON object found")
    end = max(text.rfind("}"), text.rfind("]"))
    candidate = text[start : end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)  # trailing commas
    return json.loads(candidate)


class BaseLLM:
    provider = "base"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._sem = asyncio.Semaphore(self.settings.llm_concurrency)
        self._cache_dir = Path(self.settings.cache_dir) / "llm"

    # -- provider hook --------------------------------------------------------
    async def _complete(self, system: str, user: str, schema: type[BaseModel] | None,
                        model: str) -> tuple[str, Any, int, int]:
        """Return (raw_text, parsed_obj_or_None, tokens_in, tokens_out)."""
        raise NotImplementedError

    # -- public API -------------------------------------------------------------
    def model_name(self, fast: bool = False) -> str:
        return (self.settings.llm_fast_model or self.settings.llm_model) if fast else self.settings.llm_model

    async def structured(self, system: str, user: str, schema: type[T], *, agent: str,
                         tracer: Tracer | None = None, fast: bool = False) -> T:
        tracer = tracer or null_tracer()
        model = self.model_name(fast)
        key = hashlib.sha256(json.dumps([self.provider, model, system, user, schema.__name__]).encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            try:
                obj = schema.model_validate(cached)
                tracer.record_llm(model, 0, 0, cached=True)
                tracer.emit(agent, "llm_call", model=model, cached=True, schema=schema.__name__,
                            prompt=user, output=cached, tokens_in=0, tokens_out=0, latency_ms=0)
                return obj
            except ValidationError:
                pass

        last_err: Exception | None = None
        for attempt in range(1, self.settings.llm_max_retries + 1):
            start = time.perf_counter()
            try:
                async with self._sem:
                    raw, parsed, tin, tout = await asyncio.wait_for(
                        self._complete(system, user, schema, model), timeout=self.settings.llm_timeout_s)
                data = parsed if parsed is not None else extract_json(raw)
                if isinstance(data, BaseModel):
                    data = data.model_dump()
                obj = schema.model_validate(data)
                latency = round((time.perf_counter() - start) * 1000, 1)
                tracer.record_llm(model, tin, tout)
                tracer.emit(agent, "llm_call", model=model, attempt=attempt, schema=schema.__name__,
                            prompt=user, output=obj.model_dump(), tokens_in=tin, tokens_out=tout,
                            latency_ms=latency)
                self._cache_put(key, obj.model_dump())
                return obj
            except (asyncio.TimeoutError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                last_err = exc  # bad JSON / schema mismatch / timeout -> retry
            except Exception as exc:  # provider/network errors
                last_err = exc
                if not _is_retryable(exc):
                    break
            tracer.emit(agent, "llm_retry", model=model, attempt=attempt,
                        error=f"{type(last_err).__name__}: {last_err}")
            await asyncio.sleep(min(2 ** attempt, 20))
        tracer.emit(agent, "llm_failed", model=model, error=f"{type(last_err).__name__}: {last_err}")
        raise LLMError(f"{agent}: LLM call failed: {last_err}")

    # -- cache -------------------------------------------------------------------
    def _cache_get(self, key: str) -> Any:
        if not self.settings.llm_cache:
            return None
        p = self._cache_dir / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, value: Any) -> None:
        if not self.settings.llm_cache:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")


def _is_retryable(exc: Exception) -> bool:
    if "authentication" in str(exc).lower() or "api key" in str(exc).lower():
        return False  # missing/invalid credentials will not fix themselves
    status = getattr(exc, "status_code", None)
    if status is None:
        return True  # connection errors, unknown -> retry
    return status in (408, 409, 429) or status >= 500


class AnthropicLLM(BaseLLM):
    provider = "anthropic"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        import anthropic

        kwargs: dict[str, Any] = {"timeout": self.settings.llm_timeout_s, "max_retries": 2}
        if self.settings.llm_base_url:
            kwargs["base_url"] = self.settings.llm_base_url
        if self.settings.llm_api_key:
            kwargs["api_key"] = self.settings.llm_api_key
        self.client = anthropic.AsyncAnthropic(**kwargs)

    async def _complete(self, system, user, schema, model):
        extra: dict[str, Any] = {}
        if self.settings.llm_refusal_fallback and not model.startswith("claude-haiku"):
            # server-side fallback: a refused request is retried on another model
            extra = {"extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"},
                     "extra_body": {"fallbacks": "default"}}
        output_config: dict[str, Any] = {}
        if self.settings.llm_effort:
            output_config["effort"] = self.settings.llm_effort
        resp = await self.client.messages.parse(
            model=model,
            max_tokens=self.settings.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config=output_config or None,
            **extra,
        )
        if resp.stop_reason == "refusal":
            raise ValueError("model refused the request")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.parsed_output, resp.usage.input_tokens, resp.usage.output_tokens


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAICompatLLM(BaseLLM):
    """OpenAI or any OpenAI-compatible endpoint (OpenRouter, NVIDIA API catalog, Ollama, vLLM)."""

    provider = "openai"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        from openai import AsyncOpenAI

        self.provider = self.settings.llm_provider.lower()
        kwargs: dict[str, Any] = {"timeout": self.settings.llm_timeout_s}
        if self.settings.llm_base_url:
            kwargs["base_url"] = self.settings.llm_base_url
        if self.provider == "nvidia":
            kwargs.setdefault("base_url", NVIDIA_BASE_URL)
            kwargs["api_key"] = os.environ.get("NVIDIA_API_KEY", "")
        if self.provider == "openrouter":
            kwargs.setdefault("base_url", OPENROUTER_BASE_URL)
            kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
        if self.provider == "ollama":
            kwargs.setdefault("base_url", "http://localhost:11434/v1")
            kwargs["api_key"] = "ollama"
        if self.settings.llm_api_key:  # user-supplied key (e.g. from the UI) wins over env vars
            kwargs["api_key"] = self.settings.llm_api_key
        if self.provider == "openai":
            kwargs.setdefault("api_key", os.environ.get("OPENAI_API_KEY", ""))
        if not kwargs.get("api_key"):
            raise LLMError(f"no API key for LLM provider '{self.provider}'")
        self.client = AsyncOpenAI(**kwargs)

    async def _complete(self, system, user, schema, model):
        schema_hint = ""
        if schema is not None:
            schema_hint = ("\n\nRespond ONLY with a JSON object matching this JSON schema:\n"
                           + json.dumps(schema.model_json_schema()))
        extra: dict[str, Any] = {}
        if self.provider == "nvidia":
            # Nemotron reasoning toggle; the reasoning comes back in `reasoning_content`, not `content`
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.settings.llm_thinking}}
            extra["top_p"] = self.settings.llm_top_p
            extra["seed"] = self.settings.llm_seed
        else:
            extra["response_format"] = {"type": "json_object"}
        if self.provider == "openrouter":
            # unified reasoning toggle; reasoning comes back in `message.reasoning`, not `content`
            extra["extra_body"] = {"reasoning": {"enabled": self.settings.llm_thinking}}
            extra["top_p"] = self.settings.llm_top_p
            extra["seed"] = self.settings.llm_seed
        resp = await self.client.chat.completions.create(
            model=model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            messages=[{"role": "system", "content": system + schema_hint},
                      {"role": "user", "content": user}],
            **extra,
        )
        usage = resp.usage
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # in case reasoning is inlined
        return (text, None, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))


class ScriptedLLM(BaseLLM):
    """Deterministic LLM for tests: ``responder(agent_schema_name, system, user) -> dict``."""

    provider = "scripted"

    def __init__(self, responder: Callable[[str, str, str], Any], settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    async def _complete(self, system, user, schema, model):
        name = schema.__name__ if schema else ""
        self.calls.append((name, user))
        out = self.responder(name, system, user)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps(out)
        return text, None, len(user) // 4, len(text) // 4


def get_llm(settings: Settings | None = None) -> BaseLLM:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        return AnthropicLLM(settings)
    if provider in ("openrouter", "nvidia", "openai", "ollama", "vllm"):
        return OpenAICompatLLM(settings)
    raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider}")
