"""Tier L: a dependency pattern filtered by membership in a lexicon table."""
from __future__ import annotations
from typing import Callable, List, Set, Tuple
from ..schema import Annotation, Span
from .base import Detector


class LexiconDetector(Detector):
    detector_type = "lexicon"

    def __init__(self, nlp, construct_id: str, pattern: list,
                 lexicon: Set[Tuple], key_fn: Callable, version: str = "0.1"):
        from spacy.matcher import DependencyMatcher
        self.construct_ids = [construct_id]
        self.version = version
        self._matcher = DependencyMatcher(nlp.vocab)
        self._matcher.add(construct_id, [pattern])
        self._lex = lexicon
        self._key_fn = key_fn  # key_fn(doc, token_ids) -> hashable key

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        for _mid, token_ids in self._matcher(doc):
            key = self._key_fn(doc, token_ids)
            if key not in self._lex:
                continue
            toks = sorted(token_ids)
            span = doc[toks[0]: toks[-1] + 1]
            out.append(Annotation(
                text_id=text_id, construct_id=self.construct_ids[0],
                span=Span(span.start_char, span.end_char, toks[0], toks[-1]),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": toks, "key": list(key), "matched": span.text}))
        return out
