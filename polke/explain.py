"""Introspection of the built detector registry: HOW a construct is detected.

Works over live detector objects (after build_all), pulling out whatever the
detector shape exposes: the spaCy dependency/phrase patterns, the lexicon it
is gated by, the LLM system prompt, the sibling group a router serves, gates,
and context use. Everything is extracted defensively — a detector shape this
module doesn't know still yields its class name, docstring, and version.

Used by `polke explain` (full detail) and the annotation viewer (one-line
mechanism summary per construct).
"""
from __future__ import annotations

from typing import List, Optional

# Friendly one-liners per detector class; fallback is the class docstring.
MECHANISMS = {
    "RuleRoutingDetector": "spaCy dependency pattern + deterministic routing "
                           "among sibling constructs (no LLM)",
    "LexiconRuleDetector": "spaCy dependency pattern gated by a curated "
                           "lexicon (may route among siblings)",
    "LexiconDetector": "spaCy dependency pattern gated by a curated lexicon",
    "PhraseLexiconDetector": "fixed-phrase lexicon match on surface/lemma "
                             "forms (longest match wins)",
    "Scan": "hand-written token-scan rule over the parsed sentence",
    "CompositeDetector": "OR-combination of sub-detectors",
    "LLMReadingDetector": "rule proposes a candidate span; the LLM picks one "
                          "reading from the sibling group",
    "LLMStandaloneDetector": "the LLM judges presence per sentence (optionally "
                             "pre-filtered by a cheap surface gate)",
    "DependencyRuleDetector": "spaCy dependency-pattern match",
    "_Scan": "hand-written token-scan rule over the parsed sentence",
}

# Docstring-less one-off detector classes fall back to a tier description.
TIER_FALLBACK = {
    "rule": "hand-written structural rule over the parse",
    "lexicon": "hand-written rule gated by curated word lists",
    "hybrid_rule_llm": "a rule proposes candidates; the LLM adjudicates",
    "hybrid_lexicon_llm": "a lexicon proposes candidates; the LLM adjudicates",
    "llm": "the LLM judges presence over sentence spans",
}


def _first_para(obj) -> str:
    doc = (getattr(obj, "__doc__", None) or "").strip()
    return " ".join(doc.split("\n\n")[0].split()) if doc else ""


def mechanism(det) -> str:
    cls = type(det).__name__
    return (MECHANISMS.get(cls) or _first_para(type(det))
            or TIER_FALLBACK.get(det.detector_type, cls))


def _dep_patterns(det) -> Optional[list]:
    """The DependencyMatcher patterns, tried under the keys our shapes use."""
    for attr in ("_m", "_matcher"):
        m = getattr(det, attr, None)
        if m is None or type(m).__name__ != "DependencyMatcher":
            continue
        for key in ["form"] + list(det.construct_ids):
            try:
                return list(m.get(key)[1])
            except (KeyError, ValueError, TypeError):
                continue
    return None


def _phrases(det) -> Optional[List[str]]:
    """PhraseLexiconDetector: reconstruct the phrase table from the Matcher."""
    m, route = getattr(det, "_m", None), getattr(det, "_route", None)
    if m is None or route is None or type(m).__name__ != "Matcher":
        return None
    out = []
    for str_id in route:
        try:
            label = m.vocab.strings[str_id]
            for pat in m.get(label)[1]:
                out.append(" ".join(str(next(iter(t.values()))) for t in pat))
        except (KeyError, ValueError, TypeError, StopIteration):
            continue
    return sorted(set(out)) or None


def _lexicon(det) -> Optional[List[str]]:
    lex = getattr(det, "_lex", None)
    if lex is None:
        return None
    return sorted(" ".join(k) if isinstance(k, (tuple, list)) else str(k)
                  for k in lex)


def detector_info(det) -> dict:
    """Everything introspectable about one detector object."""
    info = {
        "class": type(det).__name__,
        "version": det.version,
        "tier": det.detector_type,
        "serves": list(det.construct_ids),
        "mechanism": mechanism(det),
    }
    pats = _dep_patterns(det)
    if pats is not None:
        info["patterns"] = pats
    phrases = _phrases(det)
    if phrases is not None:
        info["phrases"] = phrases
    lex = _lexicon(det)
    if lex is not None:
        info["lexicon"] = lex
    prompt = getattr(det, "_system", None)
    if prompt:
        info["llm_prompt"] = prompt
    fn = getattr(det, "_fn", None)
    if fn is not None:
        info["scan_rule"] = {"name": getattr(fn, "__name__", "?"),
                             "doc": _first_para(fn)}
    if getattr(det, "_gate", None) is not None:
        info["gate"] = _first_para(det._gate) or "surface pre-filter"
    if getattr(det, "wants_context", False):
        info["wants_context"] = True
    subs = getattr(det, "_subs", None)
    if subs:
        info["sub_detectors"] = [detector_info(s) for s in subs]
    return info


def explain(cid: str) -> Optional[dict]:
    """detector_info for a construct id, from the process-wide registry
    (None if no detector is registered / build_all has not run)."""
    from .detectors.base import registry
    det = registry().get(cid)
    return None if det is None else detector_info(det)
