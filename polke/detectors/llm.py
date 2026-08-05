"""Tiers hybrid_rule_llm and llm.

A cheap rule proposes candidate spans; an LLM adjudicates the reading among
sibling constructs. LLM calls go through a small client protocol so the API is
swappable. DummyClient lets offline tests run (it never needs a network call).
"""
from __future__ import annotations
from typing import List, Optional, Protocol
from ..schema import Annotation, Span
from .base import Detector


class LLMClient(Protocol):
    def classify(self, system: str, user: str, labels: List[str]) -> dict:
        """Return {"construct_id": str, "confidence": float, "rationale": str}."""
        ...


class DummyClient:
    """Deterministic stand-in for offline runs/tests: returns the first label."""
    model = "dummy"

    def classify(self, system, user, labels):
        return {"construct_id": labels[0], "confidence": 0.0, "rationale": "dummy"}


def _mark(doc, token_ids):
    toks = sorted(token_ids)
    return doc[toks[0]: toks[-1] + 1], toks


class LLMReadingDetector(Detector):
    """Serves a GROUP of sibling constructs (e.g. VTA-29..34).

    Stage 1: rule matches the FORM (a candidate span).
    Stage 2: the LLM picks exactly one construct_id from the group for that span.
    """
    detector_type = "hybrid_rule_llm"

    def __init__(self, nlp, construct_ids: List[str], form_pattern: list,
                 system_prompt: str, client: Optional[LLMClient] = None,
                 version: str = "0.1"):
        from spacy.matcher import DependencyMatcher
        self.construct_ids = list(construct_ids)
        self.version = version
        self._matcher = DependencyMatcher(nlp.vocab)
        self._matcher.add("form", [form_pattern])
        self._system = system_prompt
        self._client = client or DummyClient()

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        for _mid, token_ids in self._matcher(doc):
            span, toks = _mark(doc, token_ids)
            user = doc.text.replace(span.text, "[[" + span.text + "]]", 1)
            res = self._client.classify(self._system, user, self.construct_ids)
            cid = res.get("construct_id")
            if cid not in self.construct_ids:
                continue
            out.append(Annotation(
                text_id=text_id, construct_id=cid,
                span=Span(span.start_char, span.end_char, toks[0], toks[-1]),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=float(res.get("confidence", 0.0)),
                model=getattr(self._client, "model", None),
                evidence={"tokens": toks, "matched": span.text,
                          "rationale": res.get("rationale", "")}))
        return out


class LLMStandaloneDetector(Detector):
    """No reliable formal trigger: the LLM decides presence over sentence spans.

    ``gate(sent) -> bool`` optionally pre-filters sentences with a cheap
    surface check (e.g. "ends in ?") so obvious non-candidates never cost an
    LLM call. ``wants_context=True`` opts in to the pipeline's dialogue
    context (see pipeline.annotate): the preceding utterance(s) are prepended
    to the user prompt — needed for discourse-level constructs like follow-up
    questions (QIN-04). The single-text API is unchanged: ``context`` is a
    keyword-only addition with a None default.
    """
    detector_type = "llm"

    def __init__(self, construct_id: str, system_prompt: str,
                 client: Optional[LLMClient] = None, version: str = "0.1",
                 gate=None, wants_context: bool = False):
        self.construct_ids = [construct_id]
        self.version = version
        self._system = system_prompt
        self._client = client or DummyClient()
        self._gate = gate
        self.wants_context = wants_context

    def match(self, doc, text_id: str = "doc", context=None) -> List[Annotation]:
        out = []
        for sent in doc.sents:
            if self._gate is not None and not self._gate(sent):
                continue
            user = sent.text
            if context:
                prev = "\n".join("[previous utterance] %s" % u for u in context)
                user = prev + "\n[current utterance] " + sent.text
            res = self._client.classify(self._system, user,
                                        [self.construct_ids[0], "NONE"])
            if res.get("construct_id") != self.construct_ids[0]:
                continue
            out.append(Annotation(
                text_id=text_id, construct_id=self.construct_ids[0],
                span=Span(sent.start_char, sent.end_char, sent.start, sent.end - 1),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=float(res.get("confidence", 0.0)),
                model=getattr(self._client, "model", None),
                evidence={"rationale": res.get("rationale", "")}))
        return out
