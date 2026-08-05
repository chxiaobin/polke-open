"""Configuration loading.

All runtime configuration comes from environment variables, optionally supplied
via a `.env` file (searched in the current working directory first, then the
project root). Recognised variables:

    OPENAI_API_KEY      key for the LLM tiers (hybrid_rule_llm / llm detectors)
    POLKE_LLM_MODEL     chat model for the reading classifiers (default gpt-4o-mini)
    POLKE_SPACY_MODEL   spaCy pipeline to load (default en_core_web_sm)
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
