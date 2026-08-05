"""Category VAG — vague language & approximation, Part VI.

VAG-01 general extenders: fixed phrases, list-final (next token punctuation
or utterance end), so ordinary coordination "knives and forks" and initial
"And so we left" never fire. VAG-02: placeholders fire on lexicon membership;
bare thing/things/stuff only without det/amod/poss and not as parse root
(excludes "the real thing", "The first thing on the agenda" and the
NOUN-mis-tagged imperative "Stuff the turkey"). VAG-03 hedging sort/kind of:
en_core_web_sm tags the hedge ADV/advmod, the classifying ART-19 use NOUN —
that tag difference is the router. VAG-04 (numeral approximators) is rule
tier -> Phase 3.
"""
from __future__ import annotations
from ...registry import lexicons
from ...schema import Annotation, Span
from ..base import Detector
from ..lexical import PhraseLexiconDetector
from ..spoken import tokenize_phrase


def _extender_gate(doc, start, end):
    return end >= len(doc) or doc[end].is_punct


class _VagueNouns(Detector):
    detector_type = "lexicon"
    version = "vag02-vague-nouns@0.1"
    construct_ids = ["VAG-02"]

    def __init__(self, nlp):
        lx = lexicons()["vague_nouns"]
        self._placeholders = set(lx["placeholders"])
        self._bare = {b for b in lx["bare_vague"] if " " not in b}
        self._multi = [tokenize_phrase(nlp, b) for b in lx["bare_vague"]
                       if " " in b]
        self._multi += [tokenize_phrase(nlp, p)
                        for p in lx["hyphenated_placeholders"]]
        self._multi.sort(key=len, reverse=True)

    def _emit(self, doc, lo, hi, text_id):
        span = doc[lo:hi + 1]
        return Annotation(
            text_id=text_id, construct_id="VAG-02",
            span=Span(span.start_char, span.end_char, lo, hi),
            detector_type=self.detector_type, detector_version=self.version,
            confidence=1.0,
            evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text})

    def match(self, doc, text_id="doc"):
        out, covered = [], set()
        for toks in self._multi:
            n = len(toks)
            for i in range(len(doc) - n + 1):
                if tuple(t.lower_ for t in doc[i:i + n]) == toks:
                    out.append(self._emit(doc, i, i + n - 1, text_id))
                    covered.update(range(i, i + n))
        for t in doc:
            if t.i in covered:
                continue
            if t.lower_ in self._placeholders:
                out.append(self._emit(doc, t.i, t.i, text_id))
            elif t.lower_ in self._bare:
                if t.dep_ == "ROOT":
                    continue  # "Stuff the turkey" parses as NOUN root
                if any(k.dep_ in ("det", "poss", "amod", "nummod")
                       for k in t.children):
                    continue  # referential: "the real thing", "his things"
                out.append(self._emit(doc, t.i, t.i, text_id))
        return out


class _HedgingSortKind(Detector):
    """VAG-03: sort of / kind of hedging a VP/AdjP/AdvP (ADV-tagged), plus
    transcribed sorta/kinda."""
    detector_type = "lexicon"
    version = "vag03-hedge@0.1"
    construct_ids = ["VAG-03"]

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            two = None
            if t.lower_ in ("sort", "kind") and t.i + 1 < len(doc) \
                    and doc[t.i + 1].lower_ == "of":
                if t.pos_ != "ADV" and t.dep_ != "advmod":
                    continue  # classifying "kind of + noun" -> ART-19
                two = (t.i, t.i + 1)
            elif t.lower_ in ("sorta", "kinda") and t.pos_ in ("ADV", "ADJ", "INTJ"):
                two = (t.i, t.i)
            if two is None:
                continue
            lo, hi = two
            span = doc[lo:hi + 1]
            out.append(Annotation(
                text_id=text_id, construct_id="VAG-03",
                span=Span(span.start_char, span.end_char, lo, hi),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text}))
        return out


_NUMBER_STEMS = {"ten", "eleven", "twelve", "twenty", "thirty", "forty",
                 "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
                 "thousand", "million"}


def _vag04(doc):
    """VAG-04 (rule tier): numeral approximators — approximator adverb + CD
    ("about fifty"), CD + or so/thereabouts ("twenty or so"), CD or CD
    ("five or six people"), -ish numerals ("fiftyish"), CD(-)odd
    ("twenty-odd"). Exact numerals ("three books"), fractions ("two-thirds")
    and multipliers ("twice the size") match none of these."""
    pre = set(lexicons()["numeral_approximators"]["pre"])
    for t in doc:
        # about/around/roughly/some/nearly/almost + numeral
        if t.lower_ in pre and t.i + 1 < len(doc) and doc[t.i + 1].tag_ == "CD":
            yield (t.i, t.i + 1)
        if t.tag_ == "CD":
            nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
            nx2 = doc[t.i + 2] if t.i + 2 < len(doc) else None
            if nxt is not None and nxt.lower_ == "or" and nx2 is not None:
                if nx2.lower_ in ("so", "thereabouts"):
                    yield (t.i, t.i + 2)          # twenty or so
                elif nx2.tag_ == "CD":
                    yield (t.i, t.i + 2)          # five or six
            if nxt is not None and nxt.lower_ == "odd":
                yield (t.i, t.i + 1)              # twenty odd
            if nxt is not None and nxt.text == "-" and nx2 is not None \
                    and nx2.lower_ == "odd":
                yield (t.i, t.i + 2)              # twenty-odd
        # -ish numerals: fiftyish, twentyish
        if t.lower_.endswith("ish") and t.lower_[:-3] in _NUMBER_STEMS:
            yield (t.i, t.i)


def build(nlp, client=None):
    from .common import Scan
    return [
        PhraseLexiconDetector(nlp, "VAG-01", lexicons()["general_extenders"]["entries"],
                              gate=_extender_gate, version="vag01-extender@0.1"),
        _VagueNouns(nlp),
        _HedgingSortKind(),
        Scan("VAG-04", _vag04, version="vag04-approx@0.1"),
    ]
