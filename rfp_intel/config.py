"""Central configuration.

Every tunable (model names, provider, paths, retrieval knobs) is read from
environment variables (optionally via a `.env` file) so nothing is hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Short `LLM` switch: LLM=gpt -> OpenAI, LLM=open -> OpenRouter (takes precedence over LLM_PROVIDER)
LLM_CHOICES = {"gpt": "openai", "open": "openrouter"}

# (main model, fast model) per provider; OPENAI_MODEL / OPENROUTER_MODEL override the main one
DEFAULT_MODELS = {
    "openai": ("gpt-4o-mini", "gpt-4o-mini"),
    "openrouter": ("nvidia/nemotron-3.5-super-vl:free", "nvidia/nemotron-3.5-super-vl:free"),
    "nvidia": ("nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3.5-super-vl-120b-a12b"),
    "anthropic": ("claude-opus-5-5", "claude-haiku-5-5"),
}


def provider_for(choice: str) -> str:
    choice = choice.strip().lower()
    return LLM_CHOICES.get(choice, choice)


def _llm_provider() -> str:
    if os.environ.get("LLM"):
        return provider_for(os.environ["LLM"])
    return _env("LLM_PROVIDER", "openrouter")


def default_models(provider: str) -> tuple[str, str]:
    main, fast = DEFAULT_MODELS.get(provider, ("", ""))
    override = os.environ.get(f"{provider.upper()}_MODEL")
    return (override, override) if override else (main, fast)


@dataclass
class Settings:
    # --- paths -------------------------------------------------------------
    index_dir: Path = field(default_factory=lambda: Path(_env("RFP_INDEX_DIR", "./.index")))
    output_dir: Path = field(default_factory=lambda: Path(_env("RFP_OUTPUT_DIR", "./outputs")))
    trace_dir: Path = field(default_factory=lambda: Path(_env("RFP_TRACE_DIR", "./outputs/traces")))
    cache_dir: Path = field(default_factory=lambda: Path(_env("RFP_CACHE_DIR", "./.cache")))

    # --- LLM -----------------------------------------------------------------
    # provider: openrouter | nvidia | anthropic | openai | ollama (any OpenAI-compatible server)
    llm_provider: str = field(default_factory=_llm_provider)
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", default_models(_llm_provider())[0]))
    llm_fast_model: str = field(default_factory=lambda: _env("LLM_FAST_MODEL", default_models(_llm_provider())[1]))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    # explicit key (e.g. typed into the UI); empty -> the provider's own env var (OPENAI_API_KEY, ...)
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    # Anthropic: reasoning effort (low|medium|high). OpenAI-compatible: temperature is used instead.
    llm_effort: str = field(default_factory=lambda: _env("LLM_EFFORT", "low"))
    llm_temperature: float = field(default_factory=lambda: _env_float(
        "LLM_TEMPERATURE", 0.0 if _llm_provider() == "openai" else 1.0))
    # OpenRouter / NVIDIA (Nemotron): sampling + reasoning toggle
    llm_top_p: float = field(default_factory=lambda: _env_float("LLM_TOP_P", 0.95))
    llm_seed: int = field(default_factory=lambda: _env_int("LLM_SEED", 42))
    llm_thinking: bool = field(default_factory=lambda: _env_bool("LLM_THINKING", True))
    # Anthropic server-side refusal fallback (routes a declined request to another model)
    llm_refusal_fallback: bool = field(default_factory=lambda: _env_bool("LLM_REFUSAL_FALLBACK", True))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 16384))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 300))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 3))
    llm_cache: bool = field(default_factory=lambda: _env_bool("LLM_CACHE", True))
    llm_concurrency: int = field(default_factory=lambda: _env_int("LLM_CONCURRENCY", 4))

    # --- embeddings / reranker ----------------------------------------------
    # embedding backend: sentence-transformers | openai | hash
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "sentence-transformers"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    # reranker backend: cross-encoder | llm | none
    reranker_backend: str = field(default_factory=lambda: _env("RERANKER_BACKEND", "cross-encoder"))
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # --- chunking / retrieval -----------------------------------------------
    chunk_target_chars: int = field(default_factory=lambda: _env_int("CHUNK_TARGET_CHARS", 1200))
    chunk_max_chars: int = field(default_factory=lambda: _env_int("CHUNK_MAX_CHARS", 1800))
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 200))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    candidate_pool: int = field(default_factory=lambda: _env_int("CANDIDATE_POOL", 40))
    rerank_top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 25))
    query_expansion: bool = field(default_factory=lambda: _env_bool("QUERY_EXPANSION", True))
    ocr_enabled: bool = field(default_factory=lambda: _env_bool("OCR_ENABLED", True))

    # --- agents -------------------------------------------------------------
    max_field_retries: int = field(default_factory=lambda: _env_int("MAX_FIELD_RETRIES", 2))
    evidence_per_field: int = field(default_factory=lambda: _env_int("EVIDENCE_PER_FIELD", 6))

    def with_llm(self, choice: str, api_key: str = "") -> "Settings":
        """Copy of these settings switched to another LLM (`gpt` / `open` or a provider name)."""
        provider = provider_for(choice)
        if provider == self.llm_provider:  # same provider: keep configured models, only swap the key
            return replace(self, llm_api_key=api_key or self.llm_api_key)
        main, fast = default_models(provider)
        return replace(self, llm_provider=provider, llm_model=main, llm_fast_model=fast, llm_api_key=api_key,
                       llm_base_url="", llm_temperature=0.0 if provider == "openai" else 1.0)

    def ensure_dirs(self) -> None:
        for p in (self.index_dir, self.output_dir, self.trace_dir, self.cache_dir):
            p.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Re-read the environment (used by tests)."""
    global _settings
    _settings = None
