"""Configuration loading.

All runtime configuration comes from environment variables, optionally supplied
via a `.env` file (searched in the current working directory first, then the
project root). Recognised variables:

    OPENAI_API_KEY          key for the LLM tiers (hybrid_rule_llm / llm detectors)
    POLKE_LLM_MODEL         chat model for the reading classifiers (default gpt-4o-mini)
    POLKE_SPACY_MODEL       spaCy pipeline to load (default en_core_web_sm)
    POLKE_LLM_CONCURRENCY   parallel LLM calls per text (default 8)
    POLKE_LLM_MAX_TOKENS    completion budget per classifier call (default 300)
    POLKE_LLM_NO_THINK      "1" disables reasoning mode on self-hosted models
                            (vLLM chat_template_kwargs); leave unset for the
                            OpenAI API
    POLKE_SEGMENT           sentence segmentation: "parser" (default; spaCy's
                            parser decides) or "line" (each input line is one
                            sentence — for transcribed speech, one utterance
                            per line)
"""
from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_env() -> None:
    """Load `.env` into os.environ (idempotent; existing vars win)."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        from dotenv import load_dotenv
    except ImportError:  # dotenv is optional; plain env vars still work
        return
    load_dotenv()  # cwd (and parents, via find_dotenv default)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def llm_model() -> str:
    return os.getenv("POLKE_LLM_MODEL", "gpt-4o-mini")


def spacy_model() -> str:
    return os.getenv("POLKE_SPACY_MODEL", "en_core_web_sm")


def llm_max_tokens() -> int:
    """Completion budget for the reading classifiers (default 300 — enough
    for the rationale-first JSON; raise it for models whose reasoning cannot
    be disabled)."""
    try:
        n = int(os.getenv("POLKE_LLM_MAX_TOKENS", "300"))
    except ValueError:
        return 300
    return max(50, n)


def llm_no_think() -> bool:
    """POLKE_LLM_NO_THINK=1 asks the serving layer to disable reasoning mode
    (chat_template_kwargs.enable_thinking=false) on every chat call. For
    self-hosted reasoning models (Qwen3.x etc.) served via vLLM, where
    thinking tokens would otherwise consume the whole completion budget.
    Leave unset for the OpenAI API — it rejects the extra parameter."""
    return os.getenv("POLKE_LLM_NO_THINK", "").strip().lower() in (
        "1", "true", "yes")


def segment_mode() -> str:
    v = os.getenv("POLKE_SEGMENT", "parser").strip().lower()
    return v if v in ("parser", "line") else "parser"


def llm_concurrency() -> int:
    try:
        n = int(os.getenv("POLKE_LLM_CONCURRENCY", "8"))
    except ValueError:
        return 8
    return max(1, n)
