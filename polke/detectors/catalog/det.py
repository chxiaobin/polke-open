"""Category DET - determiners & quantifiers.

Demonstratives, possessive determiners, its, both/half/no, the intensified
quantifiers (too/so/as much/many), the a-great-deal-of set and distributive
per/apiece/each are CLOSED-class FORM triggers -> small rules and surface
matchers with word lists kept here. The rows whose reading is semantic
(assertive some vs offer some vs certain some; non-assertive any vs free-choice
any; much/many vs a-lot-of style choice; (a) few / (a) little nuance; each vs
every; predeterminers; another/other; own) are the LLM tier and are registered
but skipped offline.
"""
from __future__ import annotations
from ..lexical import PhraseLexiconDetector
from ..llm import LLMStandaloneDetector
from ...schema import Annotation, Span
from ..base import Detector

DEMONSTRATIVES = {"this", "that", "these", "those"}
TIME_NOUNS = {
    "week", "day", "month", "year", "morning", "afternoon", "evening",
    "night", "time", "moment", "minute", "second", "hour", "weekend",
    "decade", "century", "summer", "winter", "spring", "autumn", "semester",
    "term", "season", "instant", "period", "while", "fortnight",
}
POSS_DET = {"my", "your", "his", "her", "our", "their"}  # 'its' -> DET-04

GREAT_DEAL = [
    "a great deal of", "a large number of", "a good many", "a good deal of",
    "a great many", "a large amount of", "a vast amount of",
    "a huge number of", "a great quantity of", "a fair amount of",
    "a small number of", "a great number of",
]
INTENSIFIED = ["too much", "too many", "so much", "so many",
               "as much", "as many"]


def _ann(text_id, cid, doc, lo, hi, ver, dtype="rule", matched=None):
    span = doc[lo:hi + 1]
    return Annotation(
        text_id=text_id, construct_id=cid,
        span=Span(span.start_char, span.end_char, lo, hi),
        detector_type=dtype, detector_version=ver, confidence=1.0,
        evidence={"tokens": list(range(lo, hi + 1)),
                  "matched": matched or span.text})


class _Demonstrative(Detector):
    """DET-01: demonstrative determiner (this/that/these/those + noun), not the
    temporal use (this week -> DET-02)."""
    construct_ids = ["DET-01"]
    detector_type = "rule"
    version = "det01-demonstrative@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ in DEMONSTRATIVES and t.dep_ == "det":
                if t.head.lemma_.lower() in TIME_NOUNS:
                    continue
                out.append(_ann(text_id, "DET-01", doc, t.i, t.i, self.version))
        return out


class _PossDeterminer(Detector):
    """DET-03: possessive determiners (my/your/his/her/our/their). 'its' is
    owned by DET-04 (its vs it's)."""
    construct_ids = ["DET-03"]
    detector_type = "rule"
    version = "det03-possdet@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ in POSS_DET and t.pos_ == "PRON":
                out.append(_ann(text_id, "DET-03", doc, t.i, t.i, self.version))
        return out


class _Its(Detector):
    """DET-04: possessive determiner 'its' modifying a following noun (the
    orthographic its/it's contrast). Skips the bare citation-list use."""
    construct_ids = ["DET-04"]
    detector_type = "lexicon"
    version = "det04-its@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ != "its" or t.pos_ != "PRON":
                continue
            if t.i + 1 < len(doc) and doc[t.i + 1].pos_ in ("NOUN", "PROPN",
                                                            "ADJ", "NUM"):
                out.append(_ann(text_id, "DET-04", doc, t.i, t.i, self.version,
                                dtype="lexicon"))
        return out


class _Both(Detector):
    """DET-18: both (of)."""
    construct_ids = ["DET-18"]
    detector_type = "rule"
    version = "det18-both@0.1"

    def match(self, doc, text_id="doc"):
        return [_ann(text_id, "DET-18", doc, t.i, t.i, self.version)
                for t in doc if t.lower_ == "both"]


class _Half(Detector):
    """DET-21: half (of)."""
    construct_ids = ["DET-21"]
    detector_type = "rule"
    version = "det21-half@0.1"

    def match(self, doc, text_id="doc"):
        return [_ann(text_id, "DET-21", doc, t.i, t.i, self.version)
                for t in doc if t.lower_ == "half"]


class _No(Detector):
    """DET-22: negative determiner 'no' (+ sg/pl/uncount noun)."""
    construct_ids = ["DET-22"]
    detector_type = "rule"
    version = "det22-no@0.1"

    def match(self, doc, text_id="doc"):
        return [_ann(text_id, "DET-22", doc, t.i, t.i, self.version)
                for t in doc if t.lower_ == "no" and t.tag_ == "DT"]


class _Distributive(Detector):
    """DET-26: distributive per / apiece / each (post-nominal each)."""
    construct_ids = ["DET-26"]
    detector_type = "lexicon"
    version = "det26-per@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ in ("per", "apiece"):
                out.append(_ann(text_id, "DET-26", doc, t.i, t.i, self.version,
                                dtype="lexicon"))
            elif t.lower_ == "each":
                prev = doc[t.i - 1] if t.i > 0 else None
                if prev is not None and (prev.like_num or prev.pos_ == "NOUN"):
                    out.append(_ann(text_id, "DET-26", doc, t.i, t.i,
                                    self.version, dtype="lexicon"))
        return out


def build(nlp, client=None):
    dets = []

    # --- form / rule tiers ------------------------------------------------- #
    dets.append(_Demonstrative())    # DET-01
    dets.append(_PossDeterminer())   # DET-03
    dets.append(_Its())              # DET-04
    dets.append(_Both())             # DET-18
    dets.append(_Half())             # DET-21
    dets.append(_No())               # DET-22
    dets.append(_Distributive())     # DET-26

    # --- surface lexicon tiers -------------------------------------------- #
    dets.append(PhraseLexiconDetector(nlp, "DET-13", GREAT_DEAL,
                                      version="det13-greatdeal@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "DET-16", INTENSIFIED,
                                      version="det16-intensified@0.1"))

    # --- LLM tier (registered structurally; skipped offline) --------------- #
    _llm = {
        "DET-02": "a demonstrative in a temporal / textual / emotive (non-"
                  "spatial) use (this week; that idea; this guy...)",
        "DET-05": "assertive 'some' quantifying an indefinite amount (I bought "
                  "some apples); NOT stressed 'some' meaning certain/"
                  "particular (some people just don't listen - a different "
                  "construct)",
        "DET-06": "'some' softening an offer or request (would you like some "
                  "tea?)",
        "DET-07": "'some' meaning a certain / particular (some people just "
                  "don't listen)",
        "DET-08": "non-assertive 'any' in a negative, question or conditional "
                  "(have you any questions?)",
        "DET-09": "free-choice 'any' meaning no-matter-which / every (take any "
                  "seat)",
        "DET-10": "'any' intensifying a comparative (is it any better?)",
        "DET-11": "much / many as the quantifier of a mass / count noun (much "
                  "time; many people)",
        "DET-12": "a lot of / lots of / plenty of quantifying a noun",
        "DET-14": "(a) few / (a) little with the positive vs negative nuance "
                  "(a few friends; few options; little hope)",
        "DET-15": "fewer / less / least / fewest comparative quantifiers "
                  "(fewer cars; less traffic)",
        "DET-17": "all / whole quantifying a whole set or entity (all the "
                  "students; the whole class)",
        "DET-19": "each / every distributive determiner with singular agreement",
        "DET-20": "either / neither (of) selecting between two",
        "DET-23": "a predeterminer standing BEFORE another determiner (all the/"
                  "both these/half a; such a; what a; rather/quite a; many a); "
                  "a bare quantifier directly before a noun (enough chairs, "
                  "many people) is a different construct",
        "DET-24": "enough / several / various / certain / numerous quantifying "
                  "a noun",
        "DET-25": "a number, fraction or multiplier quantifying a noun (three "
                  "books; two-thirds of; twice the size); NOT half/all/both "
                  "before a determiner (predeterminer - a different construct) "
                  "and NOT distributive per",
        "DET-27": "another / other(s) / the other(s)",
        "DET-28": "possessive + own for emphatic ownership (my own room; a place "
                  "of my own)",
    }
    for cid, desc in _llm.items():
        dets.append(LLMStandaloneDetector(
            cid, f"Decide whether the sentence contains: {desc}. Return JSON "
            f'{{"construct_id":"{cid}"|"NONE","confidence":0..1,'
            f'"rationale":"..."}}.', client=client,
            version=f"{cid.lower()}-llm@0.1"))
    return dets
