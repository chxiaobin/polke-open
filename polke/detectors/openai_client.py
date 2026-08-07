"""Kit-local OpenAI implementation of the LLMClient protocol.

Mirrors nlp/app/constructions.py::OpenAIClassifier but keeps the kit
standalone, so `POLKE_LLM=1 pytest` can run against a real client without
importing the service layer. Requires OPENAI_API_KEY in the environment
(tests fall back to loading the repo root .env); per-(labels, text) cache so
identical spans cost one call.
"""
from __future__ import annotations
import hashlib
import json
import os
import threading


class OpenAIClassifier:
    """classify() is called from the Annotator's thread pool: the underlying
    OpenAI client is thread-safe, the cache is guarded by a lock."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # lazy: only when actually built
        self._client = OpenAI()
        self.model = model or os.getenv("POLKE_LLM_MODEL", "gpt-4o-mini")
        self._cache: dict = {}
        self._lock = threading.Lock()

    def classify(self, system: str, user: str, labels: list) -> dict:
        # The output contract is enforced here, uniformly, rather than in the
        # per-detector prompts (not all of which state it): without an
        # explicit schema, models other than the one the prompts were
        # developed against tend to invent their own JSON keys. The rationale
        # field comes FIRST so the model reasons before it commits to an id —
        # this matters when serving-layer reasoning mode is disabled
        # (POLKE_LLM_NO_THINK). Only construct *names* may enrich the menu:
        # catalog example sentences largely coincide with the contract test
        # sentences, so embedding them would contaminate the contract suite.
        from ..registry import constructs
        inv = constructs()
        menu = ", ".join(
            '"%s"' % l + (" (%s)" % inv[l]["name"] if l in inv else "")
            for l in labels if l != "NONE") or '"NONE"'
        system = (system + "\n\nReturn only JSON, reasoning first: "
                  '{"rationale": "<brief reasoning>", "construct_id": '
                  "<one of " + menu + ', or "NONE" if none applies>, '
                  '"confidence": <0.0-1.0>}')
        key = hashlib.sha1(
            (self.model + "\x00" + "|".join(labels) + "\x00"
             + system + "\x00" + user).encode("utf-8")).hexdigest()
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        try:
            from ..env import llm_max_tokens, llm_no_think
            extra = ({"extra_body": {"chat_template_kwargs":
                                     {"enable_thinking": False}}}
                     if llm_no_think() else {})
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=llm_max_tokens(),
                **extra,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            out = {
                "construct_id": data.get("construct_id"),
                "confidence": float(data.get("confidence", 0.0) or 0.0),
                "rationale": data.get("rationale", ""),
            }
        except Exception as exc:  # noqa: BLE001 — degrade, don't crash the run
            out = {"construct_id": "NONE", "confidence": 0.0,
                   "rationale": "llm-error: %s" % exc}
        with self._lock:
            self._cache[key] = out
        return out
