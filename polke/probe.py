"""LLM recall probe (polke probe): surface false-negative CANDIDATES.

The detectors only show what they caught; this module reads already-annotated
records and asks an LLM — per sentence, per category, with NO rule/lexicon
gate — which constructions of that category are present. Anything the LLM
claims that the system did not annotate in that sentence becomes a *probe
candidate*, written back into the record under ``probe.candidates`` so the
viewer can show it as a distinct layer for human adjudication (confirm = a
real miss / reject = probe noise).

Candidates are hypotheses, not annotations: they carry sentence-level spans
and the probe's stated evidence. Precision/recall arithmetic happens in
polke.score, after a human has judged them.

Because the annotator's own LLM tiers share failure modes with any probe LLM,
recall measured against this pool is an upper bound — and for LLM-tier
constructs it is strongest when the probe uses a *different* model
(``--model`` / POLKE_PROBE_MODEL). Two provider backends are supported and
selected by the model id: ``claude-*`` models run on the Anthropic API
(``pip install anthropic``, ANTHROPIC_API_KEY), everything else on the
OpenAI API — so the probe can be a genuinely independent second annotator.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .env import llm_concurrency, llm_model, load_env
from .registry import constructs

_SYSTEM = """You are an expert grammarian annotating English text.
You are given ONE category of grammatical constructions and one sentence
(with the preceding sentence as context where available). List every
construction of the category that is clearly present in the CURRENT sentence.
Be conservative: omit anything doubtful.
Return only JSON:
{"present": [{"construct_id": "...", "confidence": 0.0-1.0,
              "evidence": "the exact words realising it"}]}
Return {"present": []} if none apply.
"""


def probe_model() -> str:
    return os.getenv("POLKE_PROBE_MODEL") or llm_model()


def _files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(set(path.rglob("*.annotations.json"))
                  | set(path.rglob("*.jsonl")))


def _load(file: Path) -> List[dict]:
    if file.suffix == ".jsonl":
        return [json.loads(line) for line
                in file.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return [json.loads(file.read_text(encoding="utf-8"))]


def _dump(file: Path, records: List[dict]) -> None:
    if file.suffix == ".jsonl":
        file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in records), encoding="utf-8")
    else:
        file.write_text(json.dumps(records[0], ensure_ascii=False, indent=2),
                        encoding="utf-8")


def category_blocks(wanted_ids) -> Dict[str, str]:
    """category -> inventory block for the prompt (only selected ids)."""
    by_cat: Dict[str, List[str]] = {}
    inv = constructs()
    for cid in sorted(wanted_ids):
        c = inv.get(cid)
        if c is None:
            continue
        line = f"{cid}: {c.get('name', '')}"
        if c.get("example"):
            line += f" — e.g. {c['example']}"
        by_cat.setdefault(c["category"], []).append(line)
    return {cat: "\n".join(lines) for cat, lines in by_cat.items()}


def _backend_for(model: str) -> str:
    return "anthropic" if model.lower().startswith("claude") else "openai"


def make_prober(model: Optional[str] = None):
    """The right prober for the model id: claude-* -> Anthropic, else OpenAI.

    Raises RuntimeError with an actionable message when the needed SDK is
    not installed.
    """
    load_env()
    model = model or probe_model()
    if _backend_for(model) == "anthropic":
        return AnthropicProber(model)
    return Prober(model)


class Prober:
    """One chat call per (sentence, category), OpenAI backend. Thread-safe."""

    def __init__(self, model: Optional[str] = None):
        load_env()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "the `openai` package is not installed (pip install openai)"
            ) from exc
        self.model = model or probe_model()
        self._client = OpenAI()

    def check(self) -> Optional[str]:
        """None when ready, else the failure reason."""
        if not os.getenv("OPENAI_API_KEY"):
            return "OPENAI_API_KEY is not set"
        from .llm import model_available
        return model_available(self._client, self.model)

    def sentence(self, category: str, block: str, text: str,
                 prev: Optional[str] = None) -> List[dict]:
        user = f"Category {category} — constructions:\n{block}\n\n"
        if prev:
            user += f"[previous sentence] {prev}\n"
        user += f"[current sentence] {text}"
        try:
            from .env import llm_no_think
            extra = ({"extra_body": {"chat_template_kwargs":
                                     {"enable_thinking": False}}}
                     if llm_no_think() else {})
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=500,
                **extra,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            out = data.get("present", [])
            return out if isinstance(out, list) else []
        except Exception:  # noqa: BLE001 — a failed probe call is just no info
            return []


# Structured-output schema for the Anthropic backend: the response is
# schema-validated by the API, so no lenient parsing is needed.
_PRESENT_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "construct_id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["construct_id", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["present"],
    "additionalProperties": False,
}


class AnthropicProber:
    """Same contract as Prober, on the Anthropic API (claude-* models).

    Uses structured outputs (output_config.format) so the reply is guaranteed
    to match _PRESENT_SCHEMA, and treats a safety refusal (stop_reason
    "refusal") as "no findings" — same degradation as any failed probe call.
    """

    def __init__(self, model: str):
        load_env()
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "probing with a claude-* model needs the `anthropic` package "
                "(pip install anthropic) and ANTHROPIC_API_KEY in the "
                "environment / .env") from exc
        self.model = model
        self._client = anthropic.Anthropic()

    def check(self) -> Optional[str]:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return "ANTHROPIC_API_KEY is not set"
        try:
            self._client.models.retrieve(self.model)
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        return None

    def sentence(self, category: str, block: str, text: str,
                 prev: Optional[str] = None) -> List[dict]:
        user = f"Category {category} — constructions:\n{block}\n\n"
        if prev:
            user += f"[previous sentence] {prev}\n"
        user += f"[current sentence] {text}"
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema",
                                          "schema": _PRESENT_SCHEMA}},
            )
            if resp.stop_reason == "refusal":
                return []
            text_out = next((b.text for b in resp.content
                             if b.type == "text"), "{}")
            out = json.loads(text_out).get("present", [])
            return out if isinstance(out, list) else []
        except Exception:  # noqa: BLE001 — a failed probe call is just no info
            return []


def _sentences(nlp, text: str) -> List[Tuple[int, int, int, int]]:
    """(start_char, end_char, token_start, token_end) per sentence."""
    return [(s.start_char, s.end_char, s.start, s.end - 1)
            for s in nlp(text).sents]


def _already_annotated(rec: dict, cid: str, lo: int, hi: int) -> bool:
    """Does a system annotation for cid overlap the [lo, hi) char range?"""
    for a in rec.get("annotations") or []:
        if a.get("construct_id") != cid:
            continue
        sp = a.get("span") or {}
        if sp.get("start_char", 0) < hi and sp.get("end_char", 0) > lo:
            return True
    return False


def run(path: Path, nlp, prober: Prober, wanted_ids,
        progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Probe every record under path; write candidates back into the files.

    Returns {"files": n, "sentences": n, "calls": n, "candidates": n}.
    """
    blocks = category_blocks(wanted_ids)
    if not blocks:
        raise ValueError("no known constructs selected")
    per_file: List[Tuple[Path, List[dict]]] = []
    for f in _files(path):
        recs = _load(f)
        if any(isinstance(r, dict) and "annotations" in r for r in recs):
            per_file.append((f, recs))
    if not per_file:
        raise ValueError(
            f"no annotation records under {path} (expected the output of "
            "`polke annotate`)")

    # (record, sent span, prev text, category) -> one probe call each.
    tasks = []
    n_sents = 0
    for _f, recs in per_file:
        for rec in recs:
            if not isinstance(rec, dict) or "annotations" not in rec:
                continue
            text = rec.get("text")
            if not isinstance(text, str):
                continue   # older record without embedded text: skip
            sents = _sentences(nlp, text)
            n_sents += len(sents)
            prev_end = None
            for (lo, hi, t0, t1) in sents:
                prev = text[prev_end[0]:prev_end[1]] if prev_end else None
                for cat, block in blocks.items():
                    tasks.append((rec, (lo, hi, t0, t1), prev, cat, block))
                prev_end = (lo, hi)

    n_candidates = 0
    with ThreadPoolExecutor(
            max_workers=max(1, min(llm_concurrency(), len(tasks) or 1))) as ex:
        futs = {ex.submit(prober.sentence, cat, block,
                          rec["text"][lo:hi], prev): (rec, (lo, hi, t0, t1))
                for rec, (lo, hi, t0, t1), prev, cat, block in tasks}
        for done, fut in enumerate(as_completed(futs), 1):
            rec, (lo, hi, t0, t1) = futs[fut]
            if progress is not None:
                progress(done, len(futs))
            for hit in fut.result():
                cid = str(hit.get("construct_id", "")).strip()
                if cid not in wanted_ids:
                    continue                      # hallucinated / off-menu id
                if _already_annotated(rec, cid, lo, hi):
                    continue                      # not a miss: system got it
                cands = rec.setdefault("probe", {}).setdefault(
                    "candidates", [])
                dup = next((c for c in cands if c["construct_id"] == cid
                            and c["span"]["start_char"] == lo), None)
                if dup is not None:
                    # A second probe model proposing the same candidate is
                    # agreement, not noise — record it so adjudication can
                    # prioritise multi-model candidates.
                    models = dup.setdefault("models", [dup.get("model")])
                    if prober.model not in models:
                        models.append(prober.model)
                    continue
                try:
                    conf = max(0.0, min(1.0, float(hit.get("confidence", 0))))
                except (TypeError, ValueError):
                    conf = 0.0
                cands.append({
                    "text_id": rec.get("text_id", ""),
                    "construct_id": cid,
                    "models": [prober.model],
                    "span": {"start_char": lo, "end_char": hi,
                             "token_start": t0, "token_end": t1},
                    "detector_type": "llm_probe",
                    "detector_version": f"probe@{prober.model}",
                    "confidence": conf,
                    "model": prober.model,
                    "evidence": {"quoted": str(hit.get("evidence", ""))},
                    "source": "probe",
                })
                n_candidates += 1

    for f, recs in per_file:
        for rec in recs:
            if isinstance(rec, dict) and "probe" in rec:
                rec["probe"]["model"] = prober.model
        _dump(f, recs)
    return {"files": len(per_file), "sentences": n_sents,
            "calls": len(tasks), "candidates": n_candidates}
