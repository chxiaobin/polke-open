"""Deterministic sibling routers and lexicon-gated rule detectors.

These extend the three reference shapes for the common case where a *group* of
sibling constructs share one formal trigger (a candidate span) and the choice
among them is itself decidable by FORM (aux-chain tense/aspect, particle type,
clause shape) rather than by meaning. Where the choice needs meaning, use the
LLM hybrid tier instead (detectors/llm.py).

- RuleRoutingDetector: one form matcher + a deterministic ``classify`` that
  returns the winning construct_id (or None). Serves a group of ids, exactly
  like LLMReadingDetector but with no model call. Precise by construction: a
  span is emitted for at most one sibling.
- LexiconRuleDetector: a form matcher gated by a ``key_fn`` whose key must be in
  a lexicon set, but which may assign one of several sibling ids (so a single
  phrasal/verb-pattern table can drive a group). Generalises LexiconDetector.
"""
from __future__ import annotations
from typing import Callable, List, Optional, Sequence
from ..schema import Annotation, Span
from .base import Detector


def _add_patterns(matcher, key, patterns):
    # patterns may be a single pattern (list[dict]) or a list of patterns.
    if patterns and isinstance(patterns[0], dict):
        patterns = [patterns]
    matcher.add(key, list(patterns))


def _group_span(doc, anchor, token_ids):
    """Span the verb group: the matched tokens plus the anchor's aux chain."""
    toks = set(token_ids)
    toks.add(anchor.i)
    for k in anchor.children:
        if k.dep_ in ("aux", "auxpass", "prt", "neg"):
            toks.add(k.i)
    toks = sorted(toks)
    # keep it contiguous from first to last matched token
    return toks[0], toks[-1]


class RuleRoutingDetector(Detector):
    detector_type = "rule"

    def __init__(self, nlp, construct_ids: Sequence[str], patterns: list,
                 classify: Callable, version: str = "0.1",
                 span: str = "group"):
        from spacy.matcher import DependencyMatcher
        self.construct_ids = list(construct_ids)
        self.version = version
        self._m = DependencyMatcher(nlp.vocab)
        _add_patterns(self._m, "form", patterns)
        self._classify = classify   # (doc, token_ids) -> construct_id | None
        self._span = span           # "group" (verb group) | "match" (matched toks)

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out, seen = [], set()
        for _mid, token_ids in self._m(doc):
            cid = self._classify(doc, token_ids)
            if not cid or cid not in self.construct_ids:
                continue
            anchor = doc[token_ids[0]]
            if self._span == "match":
                toks = sorted(token_ids)
                lo, hi = toks[0], toks[-1]
            else:
                lo, hi = _group_span(doc, anchor, token_ids)
            key = (cid, lo, hi)
            if key in seen:
                continue
            seen.add(key)
            span = doc[lo:hi + 1]
            out.append(Annotation(
                text_id=text_id, construct_id=cid,
                span=Span(span.start_char, span.end_char, lo, hi),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text}))
        return out


class CompositeDetector(Detector):
    """OR several sub-detectors that all serve the SAME construct id(s).

    Useful when one construct has two structural realisations (e.g. a dependency
    form plus a fixed multiword phrase). Deduplicates identical spans.
    """
    def __init__(self, construct_ids, subs, detector_type="rule", version="0.1"):
        self.construct_ids = [construct_ids] if isinstance(construct_ids, str) else list(construct_ids)
        self._subs = subs
        self.detector_type = detector_type
        self.version = version

    def match(self, doc, text_id="doc"):
        out, seen = [], set()
        for sub in self._subs:
            for a in sub.match(doc, text_id=text_id):
                if a.construct_id not in self.construct_ids:
                    continue
                k = (a.construct_id, a.span.token_start, a.span.token_end)
                if k in seen:
                    continue
                seen.add(k)
                out.append(a)
        return out


class LexiconRuleDetector(Detector):
    """Form matcher gated by lexicon membership; may route to sibling ids.

    key_fn(doc, token_ids) -> key ; the key must be in ``lexicon``. If ``route``
    is given it maps key -> construct_id (for tables that drive several sibling
    constructs); otherwise the single construct_id is used.
    """
    detector_type = "lexicon"

    def __init__(self, nlp, construct_ids, patterns: list, lexicon,
                 key_fn: Callable, route: Optional[Callable] = None,
                 version: str = "0.1", span: str = "group"):
        from spacy.matcher import DependencyMatcher
        self.construct_ids = list(construct_ids) if not isinstance(construct_ids, str) else [construct_ids]
        self.version = version
        self._m = DependencyMatcher(nlp.vocab)
        _add_patterns(self._m, "form", patterns)
        self._lex = lexicon
        self._key_fn = key_fn
        self._route = route
        self._span = span

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out, seen = [], set()
        for _mid, token_ids in self._m(doc):
            key = self._key_fn(doc, token_ids)
            if key is None or key not in self._lex:
                continue
            cid = self._route(key) if self._route else self.construct_ids[0]
            if not cid or cid not in self.construct_ids:
                continue
            anchor = doc[token_ids[0]]
            if self._span == "match":
                toks = sorted(token_ids)
                lo, hi = toks[0], toks[-1]
            else:
                lo, hi = _group_span(doc, anchor, token_ids)
            k = (cid, lo, hi)
            if k in seen:
                continue
            seen.add(k)
            span = doc[lo:hi + 1]
            out.append(Annotation(
                text_id=text_id, construct_id=cid,
                span=Span(span.start_char, span.end_char, lo, hi),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": list(range(lo, hi + 1)),
                          "key": list(key) if isinstance(key, tuple) else key,
                          "matched": span.text}))
        return out
