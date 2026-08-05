"""Tier P: dependency-pattern rule detectors (spaCy DependencyMatcher).

Dependency labels below use the spaCy en_core_web_* scheme
(auxpass / nsubjpass / prep). For a UD pipeline (Stanza/Trankit) translate to
aux:pass / nsubj:pass / case.
"""
from __future__ import annotations
from typing import Callable, List, Optional
from ..schema import Annotation, Span
from .base import Detector


def _span_from_tokens(doc, token_ids):
    toks = sorted(token_ids)
    return doc[toks[0]: toks[-1] + 1], toks


class DependencyRuleDetector(Detector):
    detector_type = "rule"

    def __init__(self, nlp, construct_id: str, pattern: list,
                 exclude: Optional[Callable] = None, version: str = "0.1"):
        from spacy.matcher import DependencyMatcher
        self.construct_ids = [construct_id]
        self.version = version
        self._matcher = DependencyMatcher(nlp.vocab)
        self._matcher.add(construct_id, [pattern])
        self._exclude = exclude  # exclude(doc, token_ids) -> bool

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        for _match_id, token_ids in self._matcher(doc):
            if self._exclude and self._exclude(doc, token_ids):
                continue
            span, toks = _span_from_tokens(doc, token_ids)
            out.append(Annotation(
                text_id=text_id, construct_id=self.construct_ids[0],
                span=Span(span.start_char, span.end_char, toks[0], toks[-1]),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": toks, "matched": span.text}))
        return out
