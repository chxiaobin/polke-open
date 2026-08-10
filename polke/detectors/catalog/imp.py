"""Category IMP - imperatives & directives (IMP-01..09).

Rule tier (tested offline):
  IMP-01 base-form imperative (initial VB, no subject, no operator).
  IMP-02 negative imperative (don't / do not + base verb, no subject).
  IMP-03 emphatic/persuasive do (do + base verb, no subject, no negation).
  IMP-05 always / never + imperative.
  IMP-06 let's / let us (first-person-plural suggestion) + negatives.
  IMP-09 softened imperative (please / just + imperative).
LLM tier (registered, skipped offline):
  IMP-04 subject-for-emphasis (you/somebody + imperative);
  IMP-07 let me / let + 3rd-person object; IMP-08 imperative + and/or condition.

The rule ids share one candidate - a subjectless base-form main verb - so one
custom router classifies among them by the leading adverb / operator / let form.
spaCy en_core_web_sm sometimes mis-tags the head of a bare two-word imperative
(``Turn left`` -> Turn=nsubj, left=ROOT), so the candidate verb is taken as the
first sentence-initial VB, falling back to the ROOT.
"""
from __future__ import annotations
from ...schema import Annotation, Span
from ..base import Detector
from .common import SentenceScoped
from ..llm import LLMReadingDetector

_SUBJECT_DEPS = ("nsubj", "nsubjpass", "expl")


def _imperative_verb(doc):
    """The main verb of an imperative clause, or None."""
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    # prefer a sentence-initial base-form VB (robust to Turn/left mis-parse)
    lead = None
    for t in doc:
        if t.is_punct:
            continue
        if t.tag_ == "VB":
            lead = t
        break
    v = lead or root
    if v is None or v.tag_ not in ("VB", "VBP"):
        # allow a base verb mis-tagged VBN as ROOT when the first token is a VB
        if root is not None and lead is not None and root.pos_ in ("VERB", "AUX"):
            v = root
        else:
            return None
    return v


def _has_real_subject(v):
    for c in v.children:
        if c.dep_ in _SUBJECT_DEPS and c.lemma_ != "let" and c.lower_ not in ("'s",):
            # 'you'/'somebody' etc. -> IMP-04 (LLM); a genuine subject
            return True
    return False


class _Imperatives(SentenceScoped):
    construct_ids = ["IMP-01", "IMP-02", "IMP-03", "IMP-05", "IMP-06"]
    detector_type = "rule"
    version = "imp-router@0.1"

    def _classify(self, doc):
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        # IMP-06: let's / let us (the embedded subject is 's/us)
        if root is not None and root.lemma_ == "let":
            for t in doc:
                if t.dep_ == "nsubj" and t.lower_ in ("'s", "us"):
                    return "IMP-06", root
            return None, None            # let me/them -> IMP-07 (LLM)
        v = _imperative_verb(doc)
        if v is None:
            return None, None
        if _has_real_subject(v):
            return None, None            # You sit here / Somebody call -> IMP-04
        auxes = [c for c in v.children if c.dep_ in ("aux", "auxpass")]
        negs = [c for c in v.children if c.dep_ == "neg"]
        do_aux = [a for a in auxes if a.lemma_ == "do"]
        if negs and do_aux:
            return "IMP-02", v           # Don't move
        if do_aux:
            return "IMP-03", v           # Do come in
        if auxes:
            return None, None            # other operator -> not a bare imperative
        # leading always / never -> IMP-05
        lead = next((t for t in doc if not t.is_punct), None)
        if lead is not None and lead.lower_ in ("always", "never") and lead.i < v.i:
            return "IMP-05", v
        return "IMP-01", v

    def match_sent(self, doc, text_id="doc"):
        cid, v = self._classify(doc)
        if cid is None:
            return []
        lo, hi = 0, len(doc) - 1
        # trim trailing punctuation
        while hi > lo and doc[hi].is_punct:
            hi -= 1
        span = doc[lo:hi + 1]
        return [Annotation(
            text_id=text_id, construct_id=cid,
            span=Span(span.start_char, span.end_char, lo, hi),
            detector_type=self.detector_type, detector_version=self.version,
            confidence=1.0, evidence={"tokens": list(range(lo, hi + 1)),
                                      "matched": span.text, "verb": v.text})]


# --------------------------------------------------------------------------- #
# IMP-09: softened imperative (please / just + imperative).
# --------------------------------------------------------------------------- #
class _SoftImperative(SentenceScoped):
    construct_ids = ["IMP-09"]
    detector_type = "rule"
    version = "imp09-soft@0.1"

    def match_sent(self, doc, text_id="doc"):
        v = _imperative_verb(doc)
        if v is None or _has_real_subject(v):
            return []
        if any(c.dep_ in ("aux", "auxpass") for c in v.children):
            return []
        softener = None
        for t in doc:
            if t.is_punct:
                continue
            if t.lower_ in ("please", "just", "kindly") and t.i <= v.i:
                softener = t
            break
        # "please" may also attach later in the clause
        if softener is None:
            softener = next((t for t in doc if t.lower_ == "please"), None)
        if softener is None:
            return []
        span = doc[0:len(doc)]
        hi = len(doc) - 1
        while hi > 0 and doc[hi].is_punct:
            hi -= 1
        span = doc[0:hi + 1]
        return [Annotation(
            text_id=text_id, construct_id="IMP-09",
            span=Span(span.start_char, span.end_char, 0, hi),
            detector_type=self.detector_type, detector_version=self.version,
            confidence=1.0, evidence={"tokens": list(range(0, hi + 1)),
                                      "matched": span.text})]


# --------------------------------------------------------------------------- #
# LLM tier.
# --------------------------------------------------------------------------- #
_IMP04_SYS = ("Return IMP-04 for an imperative with an overt subject for "
              "emphasis/contrast (You sit here; Somebody call a doctor).")
_IMP07_SYS = ("Return IMP-07 for a third-person/indirect directive with let + "
              "object (Let me help; Let them wait) - NOT the let's suggestion.")
_IMP08_SYS = ("Return IMP-08 for an imperative coordinated with and/or expressing "
              "a condition/consequence (Hurry, or we'll be late; Move and I'll "
              "shout).")

_VB_FORM = [{"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VB"}}]
_LET_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"LEMMA": "let"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "o",
     "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "dobj"]}}},
]


def build(nlp, client=None):
    return [
        _Imperatives(),
        _SoftImperative(),
        LLMReadingDetector(nlp, ["IMP-04"], _VB_FORM, _IMP04_SYS,
                           client=client, version="imp04-subj@0.1"),
        LLMReadingDetector(nlp, ["IMP-07"], _LET_FORM, _IMP07_SYS,
                           client=client, version="imp07-let@0.1"),
        LLMReadingDetector(nlp, ["IMP-08"], _VB_FORM, _IMP08_SYS,
                           client=client, version="imp08-cond@0.1"),
    ]
