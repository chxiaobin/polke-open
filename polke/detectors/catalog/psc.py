"""Category PSC — pseudo-coordination & phrasal intensification, Part VI.

Phase 2 implements PSC-03 (nice and / good and + adjective). "nice and" is
productive before any conjoined adjective/adverb; the other first elements
are intensifying only in fixed collocations ("good and ready"), so they are
whitelisted per second adjective — plain coordination "good and kind" stays
out. A following determiner ("nice and the service was quick") signals real
clause coordination and is excluded. PSC-01/02 are rule tier -> Phase 3.
"""
from __future__ import annotations
from ...registry import lexicons
from ...schema import Annotation, Span
from ..base import Detector


class _BinomialIntensifier(Detector):
    detector_type = "lexicon"
    version = "psc03-binomial@0.1"
    construct_ids = ["PSC-03"]

    def __init__(self):
        lx = lexicons()["binomial_intensifiers"]
        self._open = set(lx["open_first"])
        self._restricted = {k: set(v) for k, v in lx["restricted_first"].items()}

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            first = t.lower_
            if first not in self._open and first not in self._restricted:
                continue
            if t.i + 2 >= len(doc) or doc[t.i + 1].lower_ != "and":
                continue
            second = doc[t.i + 2]
            if first in self._open:
                if second.pos_ not in ("ADJ", "ADV"):
                    continue
            elif second.lower_ not in self._restricted[first]:
                continue
            span = doc[t.i:t.i + 3]
            out.append(Annotation(
                text_id=text_id, construct_id="PSC-03",
                span=Span(span.start_char, span.end_char, t.i, t.i + 2),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": [t.i, t.i + 1, t.i + 2],
                          "matched": span.text}))
        return out


def _pseudo_coord(doc, first_lemmas):
    """try/go/come + and + BASE verb, both verbs uninflected — the single-
    predicate reading ("try and come", "Go and get it"). Past-tense "I tried
    and failed" / "He went home and I stayed" are real coordination: the
    conjunct is VBD and/or has its own subject."""
    for t in doc:
        if t.lemma_.lower() not in first_lemmas or t.tag_ != "VB":
            continue
        conj = next((c for c in t.children if c.dep_ == "conj"), None)
        if conj is None or conj.tag_ != "VB":
            continue
        if any(c.dep_ == "cc" and c.lower_ == "and" for c in t.children) is False:
            continue
        if any(c.dep_ in ("nsubj", "nsubjpass") for c in conj.children):
            continue                        # own subject -> clausal coordination
        if conj.i - t.i > 3:
            continue                        # keep the pseudo-coordination tight
        yield (t.i, conj.i)


def _psc01(doc):
    yield from _pseudo_coord(doc, {"try"})


def _psc02(doc):
    yield from _pseudo_coord(doc, {"go", "come"})


def build(nlp, client=None):
    from .common import Scan
    return [
        _BinomialIntensifier(),
        Scan("PSC-01", _psc01, version="psc01-try-and@0.1"),
        Scan("PSC-02", _psc02, version="psc02-go-and@0.1"),
    ]
