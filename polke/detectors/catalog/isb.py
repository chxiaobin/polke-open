"""Category ISB — insubordination (freestanding subordinate clauses), Part VI.

The formal signal is that the SENTENCE ROOT itself carries the subordinator:
in "If you could just sign here." the root is *sign* with mark *If*, whereas
in the integrated "If you could sign here, we'd be done" the if-clause is an
advcl and the root is the apodosis. Same logic for because/cos (ISB-02:
"Because it's late." root has mark Because; "Because of the rain, ..." has
prep *Because* and a full main clause). ISB-03 fires on a declarative
sentence opening with WDT *which* ("... Which was nice."); interrogative
"Which book do you want?" is excluded by the "?".
"""
from __future__ import annotations
from .common import Scan, ends_with


def _root_mark(sent, markers):
    root = sent.root
    for c in root.children:
        if c.dep_ == "mark" and c.lower_ in markers:
            return c
    return None


def _isb01(doc):
    for sent in doc.sents:
        m = _root_mark(sent, {"if"})
        if m is not None:
            yield (m.i, sent.root.i)


def _isb02(doc):
    for sent in doc.sents:
        m = _root_mark(sent, {"because", "cos", "cause", "cuz", "coz"})
        if m is not None:
            yield (m.i, sent.root.i)


def _isb03(doc):
    for sent in doc.sents:
        if ends_with(sent, "?"):
            continue                        # interrogative which
        first = next((t for t in sent if not t.is_punct), None)
        if first is None or first.tag_ != "WDT" or first.lower_ != "which":
            continue
        yield (first.i, sent.root.i)


def build(nlp, client=None):
    return [
        Scan("ISB-01", _isb01, version="isb01-if@0.1"),
        Scan("ISB-02", _isb02, version="isb02-because@0.1"),
        Scan("ISB-03", _isb03, version="isb03-which@0.1"),
    ]
