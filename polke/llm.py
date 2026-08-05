"""LLM client wiring + startup readiness check.

Two of the four detector tiers (`hybrid_rule_llm`, `llm`) need a chat model to
judge readings; the `rule` and `lexicon` tiers are fully offline. This module
decides which client the annotators get:

- `OpenAIClassifier` (polke.detectors.openai_client) when a key is configured
  and the API answers the startup probe;
- `NullClient` otherwise, so LLM-tier constructs simply produce no annotations
  instead of failing the run.

`check_llm()` is the startup probe: call it early and surface its `reason` to
the user when `ready` is False.
"""
from __future__ import annotations

import os

from .env import llm_model, load_env
from .registry import constructs

LLM_TYPES = {"llm", "hybrid_rule_llm", "hybrid_lexicon_llm"}


class NullClient:
    """Suppresses the LLM tiers: returns NONE so no reading/standalone detector
    fires. Rule/lexicon detection is unaffected."""

    model = None

    def classify(self, system, user, labels):
        return {"construct_id": "NONE", "confidence": 0.0, "rationale": "no-llm"}


def llm_construct_ids() -> set:
    """Construct ids whose detector needs a model."""
    return {c["id"] for c in constructs().values()
            if c.get("detector_type") in LLM_TYPES}


def check_llm(timeout: float = 10.0) -> dict:
    """Probe the model API. Returns {ready, model, reason}.

    ready=False never raises — the caller decides whether a missing LLM is a
    warning (annotate with rule/lexicon tiers only) or an error.
    """
    load_env()
    model = llm_model()
    if not os.getenv("OPENAI_API_KEY"):
        return {"ready": False, "model": model,
                "reason": "OPENAI_API_KEY is not set (create a .env file, "
                          "see .env.example)"}
    try:
        from openai import OpenAI
    except ImportError:
        return {"ready": False, "model": model,
                "reason": "the `openai` package is not installed "
                          "(pip install openai)"}
    try:
        client = OpenAI(timeout=timeout)
        client.models.retrieve(model)
    except Exception as exc:  # noqa: BLE001 — any failure means "not ready"
        return {"ready": False, "model": model,
                "reason": f"model API not reachable ({type(exc).__name__}: {exc})"}
    return {"ready": True, "model": model, "reason": "ok"}


def build_client(no_llm: bool = False, status: dict | None = None):
    """Return (client, status). `status` may be passed in to reuse an earlier
    check_llm() probe instead of probing twice."""
    if no_llm:
        return NullClient(), {"ready": False, "model": None,
                              "reason": "LLM tiers disabled (--no-llm)"}
    if status is None:
        status = check_llm()
    if not status["ready"]:
        return NullClient(), status
    from .detectors.openai_client import OpenAIClassifier
    return OpenAIClassifier(model=status["model"]), status


def warning_text(status: dict) -> str:
    """A human-readable warning for a not-ready LLM status."""
    n = len(llm_construct_ids())
    return (f"WARNING: LLM tiers unavailable — {status['reason']}. "
            f"{n} of {len(constructs())} constructs (detector types "
            f"hybrid_rule_llm/llm) will produce no annotations; "
            f"rule and lexicon tiers run normally.")
