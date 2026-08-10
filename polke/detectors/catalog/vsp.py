"""Category VSP - special verb-meaning areas (VSP-01..13).

used to (01 discontinued habit) vs its neg/question do-support (02) vs be/get
used to + -ing/NP familiarity (03); would rather/sooner/prefer preference (04);
the mandative subjunctive (05) and formulaic (06) and were- (07) subjunctives;
the wish/if-only regret family (08..11); it's (high) time + past (12); and
possessive have / have got (13).

Verified on en_core_web_sm: discontinued 'used to' = 'used' (VBD) + to-inf
xcomp; neg/question keeps base 'use' (VB) under do-support; be/get used to =
'used' (VBN) with a be/get auxpass and a prepositional 'to'; 'have got' = 'got'
(VBN, lemma get) with a have/be aux and a direct object; possessive 'have' is a
main VERB with a dobj (not an auxiliary).

LLM tiers (skip offline): the wish/if-only readings VSP-08..11 and VSP-12
it's-time.
"""
from __future__ import annotations
from ..base import Detector
from ..rules import DependencyRuleDetector
from ..routing import RuleRoutingDetector
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector

_MANDATIVE_VERBS = {"demand", "insist", "suggest", "recommend", "request",
                    "require", "propose", "order", "ask", "urge", "move",
                    "stipulate", "advise", "command", "decree", "beg"}
_MANDATIVE_ADJS = {"essential", "important", "vital", "necessary", "crucial",
                   "imperative", "advisable", "desirable", "fitting", "urgent",
                   "mandatory", "obligatory", "preferable", "critical", "key"}
_PREF_WORDS = {"rather", "sooner"}


# --------------------------------------------------------------------------- #
# VSP-01 used to + base (discontinued habit/state).
# --------------------------------------------------------------------------- #
_USED_TO = [
    {"RIGHT_ID": "u", "RIGHT_ATTRS": {"LEMMA": "use", "TAG": "VBD"}},
    {"LEFT_ID": "u", "REL_OP": ">", "RIGHT_ID": "x", "RIGHT_ATTRS": {"DEP": "xcomp"}},
    {"LEFT_ID": "x", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": "TO"}},
]


def _used_to_exclude(doc, token_ids):
    u = doc[token_ids[0]]
    return any(c.dep_ in ("aux", "neg") and c.lemma_ == "do" for c in u.children)


# --------------------------------------------------------------------------- #
# VSP-02 used to negatives / questions (do-support + base 'use').
# --------------------------------------------------------------------------- #
_USE_DO = [
    {"RIGHT_ID": "u", "RIGHT_ATTRS": {"LEMMA": "use", "TAG": "VB"}},
    {"LEFT_ID": "u", "REL_OP": ">", "RIGHT_ID": "d",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "do"}},
    {"LEFT_ID": "u", "REL_OP": ">", "RIGHT_ID": "x", "RIGHT_ATTRS": {"DEP": "xcomp"}},
    {"LEFT_ID": "x", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": "TO"}},
]


# --------------------------------------------------------------------------- #
# VSP-03 be/get used to + -ing/NP (familiarity).
# --------------------------------------------------------------------------- #
_BE_USED_TO = [
    {"RIGHT_ID": "u", "RIGHT_ATTRS": {"LEMMA": "use", "TAG": "VBN"}},
    {"LEFT_ID": "u", "REL_OP": ">", "RIGHT_ID": "p",
     "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "to"}},
]


def _classify_be_used(doc, token_ids):
    u = doc[token_ids[0]]
    # 'be'/'being' lemmatise to 'be' but 'getting' keeps its own lemma, so
    # accept the surface forms of get too.
    for a in u.children:
        if a.dep_ not in ("auxpass", "aux"):
            continue
        if a.lemma_ in ("be", "get") or a.lower_ in ("getting", "being", "got",
                                                     "gets", "get"):
            return "VSP-03"
    return None


# --------------------------------------------------------------------------- #
# VSP-04 would rather/sooner/prefer (preference).
# --------------------------------------------------------------------------- #
class _PreferenceDetector(Detector):
    detector_type = "lexicon"
    construct_ids = ["VSP-04"]

    def __init__(self, version="vsp04-preference@0.1"):
        self.version = version

    def match(self, doc, text_id="doc"):
        from ...schema import Annotation, Span
        out = []
        has_md = any(t.tag_ == "MD" for t in doc)
        if not has_md:
            return out
        for t in doc:
            if t.lower_ in _PREF_WORDS or t.lemma_.lower() == "prefer":
                out.append(Annotation(
                    text_id=text_id, construct_id="VSP-04",
                    span=Span(t.idx, t.idx + len(t.text), t.i, t.i),
                    detector_type=self.detector_type, detector_version=self.version,
                    confidence=1.0,
                    evidence={"tokens": [t.i], "matched": t.text}))
                break
        return out


# --------------------------------------------------------------------------- #
# VSP-05 mandative subjunctive.
# --------------------------------------------------------------------------- #
_THAT_CL = [
    {"RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "acl"]}}},
    {"LEFT_ID": "c", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"DEP": "mark", "LOWER": "that"}},
]


def _is_subjunctive(c):
    if c.tag_ in ("VB", "VBP"):
        return True
    return any(a.lemma_ == "be" and a.tag_ == "VB" for a in c.children)


def _classify_mandative(doc, token_ids):
    c = doc[token_ids[0]]
    h = c.head
    trig = h.lemma_.lower() in _MANDATIVE_VERBS or \
        any(ch.lemma_.lower() in _MANDATIVE_ADJS
            for ch in h.children if ch.dep_ in ("acomp", "attr", "oprd", "amod"))
    if trig and _is_subjunctive(c):
        return "VSP-05"
    return None


# --------------------------------------------------------------------------- #
# VSP-07 were-subjunctive.
# --------------------------------------------------------------------------- #
class _WereSubjunctiveDetector(Detector):
    detector_type = "lexicon"
    construct_ids = ["VSP-07"]
    _SING = {"i", "he", "she", "it", "one"}

    def __init__(self, version="vsp07-were@0.1"):
        self.version = version

    def match(self, doc, text_id="doc"):
        from ...schema import Annotation, Span
        out = []
        for t in doc:
            if t.lower_ != "were" or t.lemma_ != "be":
                continue
            subjs = [c for c in t.children if c.dep_ in ("nsubj", "nsubjpass", "expl")]
            marks = [c.lower_ for c in t.children if c.dep_ == "mark"]
            prev = doc[t.i - 1] if t.i > 0 else None
            # subjunctive needs BOTH an irrealis frame (if/as/though/wish) and
            # a singular subject; bare singular "she were" is dialect past-BE
            # (VER-03), bare "if they were" is the ordinary indicative.
            irrealis = (any(m in ("if", "as", "though", "whether") for m in marks)
                        or (prev is not None and prev.lower_ in ("if", "as"))
                        or any(a.lemma_ == "wish" for a in t.ancestors))
            singular = any(s.lower_ in self._SING for s in subjs)
            if not (irrealis and (singular or not subjs)):
                continue
            out.append(Annotation(
                text_id=text_id, construct_id="VSP-07",
                span=Span(t.idx, t.idx + len(t.text), t.i, t.i),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": [t.i], "matched": t.text}))
        return out


# --------------------------------------------------------------------------- #
# VSP-13 possessive have / have got.
# --------------------------------------------------------------------------- #
class _HaveGotDetector(Detector):
    detector_type = "lexicon"
    construct_ids = ["VSP-13"]

    def __init__(self, version="vsp13-have@0.1"):
        self.version = version

    def match(self, doc, text_id="doc"):
        from ...schema import Annotation, Span
        out, seen = [], set()
        for t in doc:
            hit = False
            if t.lemma_ == "get" and t.tag_ == "VBN":
                has_aux = any(c.dep_ in ("aux", "auxpass") for c in t.children)
                has_obj = any(c.dep_ == "dobj" for c in t.children)
                hit = has_aux and has_obj                # have/has got + NP
            elif t.lemma_ == "have" and t.pos_ == "VERB" and \
                    t.dep_ in ("ROOT", "ccomp", "conj", "relcl", "advcl", "acl"):
                hit = any(c.dep_ == "dobj" for c in t.children)   # possessive have + NP
            if not hit or t.i in seen:
                continue
            seen.add(t.i)
            out.append(Annotation(
                text_id=text_id, construct_id="VSP-13",
                span=Span(t.idx, t.idx + len(t.text), t.i, t.i),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": [t.i], "matched": t.text}))
        return out


# --------------------------------------------------------------------------- #
# LLM tier (registered, skipped offline).
# --------------------------------------------------------------------------- #
_WISH_SYS = (
    "The bracketed clause is a wish / 'if only' construction. Choose one:\n"
    "VSP-08 wish + past for a present regret ('I wish I knew') | "
    "VSP-09 wish/if-only + past perfect for a past regret ('If only I had "
    "asked') | VSP-10 wish + would for annoyance / a desired change ('I wish "
    "you'd stop') | VSP-11 wish + could ('I wish I could help')."
)
_TIME_SYS = (
    "Return VSP-12 for the 'It's (high) time + past subjunctive' construction "
    "expressing that something is overdue ('It's time we left')."
)
_WISH_FORM = [
    {"RIGHT_ID": "w", "RIGHT_ATTRS": {"LEMMA": "wish"}},
    {"LEFT_ID": "w", "REL_OP": ">", "RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": "ccomp"}},
]


def build(nlp, client=None):
    dets = []

    dets.append(DependencyRuleDetector(
        nlp, "VSP-01", _USED_TO, exclude=_used_to_exclude,
        version="vsp01-usedto@0.1"))
    dets.append(DependencyRuleDetector(
        nlp, "VSP-02", _USE_DO, version="vsp02-usedto-neg@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["VSP-03"], _BE_USED_TO, _classify_be_used,
        version="vsp03-beusedto@0.1"))
    dets.append(_PreferenceDetector())
    dets.append(RuleRoutingDetector(
        nlp, ["VSP-05"], _THAT_CL, _classify_mandative,
        version="vsp05-mandative@0.1"))
    dets.append(PhraseLexiconDetector(
        nlp, "VSP-06",
        ["god save", "god bless", "god help", "long live", "be that as it may",
         "come what may", "if need be", "so be it", "far be it",
         "suffice it to say", "heaven forbid", "perish the thought",
         "would that", "God willing"],
        version="vsp06-formulaic@0.1"))
    dets.append(_WereSubjunctiveDetector())
    dets.append(_HaveGotDetector())

    # LLM tier.
    dets.append(LLMReadingDetector(
        nlp, ["VSP-08", "VSP-09", "VSP-10", "VSP-11"], _WISH_FORM, _WISH_SYS,
        client=client, version="vsp-wish@0.1"))
    dets.append(LLMStandaloneDetector(
        "VSP-12", _TIME_SYS, client=client, version="vsp12-time@0.1"))

    return dets
