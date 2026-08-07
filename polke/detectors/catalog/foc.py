"""Category FOC - focus, fronting, inversion, emphasis (FOC-01..17).

Rule/lexicon tier (tested offline):
  FOC-09 directional fronting + subject-verb inversion (Down came the rain;
         Here comes the bus; There goes the train).
  FOC-10 fronted place-PP inversion, or so/as/than + operator + full-NP subject
         (On the hill stood a castle; so did her sister).
  FOC-11 negative/restrictive fronting + subject-operator inversion (Never have
         I...; Not only did he...; Only then did I realise).
  FOC-12 emphatic do/did (declarative do-support with the subject before do).
  FOC-13 additive inversion so/neither/nor + operator + pronoun subject
         (So do I; Neither did she).
  FOC-14 so/such for emphasis (so + adj/adv; such + NP).
  FOC-15 wh-intensifiers (what on earth / the hell / in the world) - lexicon.
LLM tier (registered, skipped offline):
  FOC-01/02/03 it-clefts; FOC-04/05/06 wh-/pseudo-/reversed-cleft & all-cleft;
  FOC-07 object/complement fronting; FOC-08 adverbial fronting (marked theme);
  FOC-16 end-weight/end-focus; FOC-17 right/left dislocation.

FOC-09/10/11/12/13/14 share the theme "marked word order/emphasis" and are
decidable by FORM, so one custom router classifies them; FOC-13 is split from
FOC-10 by a pronoun (additive echo) vs full-NP (locative-style) subject.
"""
from __future__ import annotations
from ...schema import Annotation, Span
from ..base import Detector
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector

_DIRECTIONAL = {"down", "up", "here", "there", "in", "out", "off", "away",
                "back", "along", "over", "round", "on", "onward", "onwards"}
_FOC11_FRONT = {"never", "rarely", "seldom", "hardly", "scarcely", "barely",
                "little", "nowhere", "no", "not", "only", "neither", "nor"}
_ADD_INV = {"so", "neither", "nor"}
_PRON_SUBJ = {"i", "he", "she", "we", "they", "you", "it"}


def _is_operator(t):
    return t.tag_ == "MD" or (t.lemma_ in ("do", "have", "be")
                              and t.tag_ in ("VBD", "VBP", "VBZ", "VB"))


class _FocusRules(Detector):
    construct_ids = ["FOC-09", "FOC-10", "FOC-11", "FOC-12", "FOC-13", "FOC-14"]
    detector_type = "rule"
    version = "foc-router@0.1"

    # -- individual tests --------------------------------------------------- #
    def _foc11(self, doc):
        if len(doc) < 2 or doc[0].lower_ not in _FOC11_FRONT:
            return None
        j = 1
        while j < len(doc) and doc[j].tag_ in ("RB", "RBR", "RBS"):
            j += 1
        if j >= len(doc) or not _is_operator(doc[j]):
            return None
        if not any(t.pos_ in ("PRON", "NOUN", "PROPN") for t in doc[j + 1:j + 4]):
            return None
        return (0, j)

    def _add_inv(self, doc):
        """so/neither/nor + operator + subject. Return (span, is_pronoun)."""
        if len(doc) < 2 or doc[0].lower_ not in _ADD_INV:
            return None
        j = 1
        while j < len(doc) and doc[j].tag_ in ("RB", "RBR", "RBS"):
            j += 1
        if j >= len(doc) or not _is_operator(doc[j]):
            return None
        subj = next((t for t in doc[j + 1:] if t.pos_ in ("PRON", "NOUN", "PROPN")),
                    None)
        if subj is None:
            return None
        return ((0, subj.i), subj.lower_ in _PRON_SUBJ or subj.tag_ == "PRP")

    def _foc10_pp(self, doc):
        """Fronted place-PP + inversion (On the hill stood a castle)."""
        if not doc or doc[0].tag_ != "IN" or doc[0].dep_ != "prep":
            return None
        head = doc[0].head
        if head.pos_ not in ("VERB", "AUX") or head.i <= doc[0].i:
            return None
        if not any(c.dep_ in ("nsubj", "dobj", "attr", "nsubjpass")
                   and c.i > head.i for c in head.children):
            return None
        return (0, head.i)

    def _foc09(self, doc):
        if not doc or doc[0].lower_ not in _DIRECTIONAL:
            return None
        if doc[0].dep_ not in ("advmod", "expl"):
            return None
        v = doc[0].head
        if v.pos_ not in ("VERB", "AUX"):
            return None
        after = any(c.dep_ in ("nsubj", "attr", "nsubjpass") and c.i > v.i
                    for c in v.children)
        if not (after or doc[0].dep_ == "expl"):
            return None
        return (0, v.i)

    def _foc12(self, doc):
        for t in doc:
            if t.lemma_ == "do" and t.dep_ == "aux" and t.tag_ in ("VBP", "VBZ", "VBD"):
                h = t.head
                if h.tag_ != "VB":
                    continue
                if any(c.dep_ == "neg" for c in h.children):
                    continue
                subj = [c for c in h.children if c.dep_ in ("nsubj", "nsubjpass")]
                if subj and subj[0].i < t.i and subj[0].pos_ in ("PRON", "NOUN", "PROPN"):
                    lo = min(subj[0].i, t.i)
                    return (lo, h.i)
        return None

    def _foc14(self, doc):
        for t in doc:
            if t.lower_ == "so" and t.tag_ == "RB" and t.head.pos_ in ("ADJ", "ADV"):
                return (min(t.i, t.head.i), max(t.i, t.head.i))
            if t.lower_ == "such" and (t.tag_ == "PDT" or t.dep_ == "predet"):
                return (t.i, t.head.i)
        return None

    def match(self, doc, text_id="doc"):
        out = []

        def emit(cid, span):
            lo, hi = span
            if lo > hi:
                lo, hi = hi, lo
            s = doc[lo:hi + 1]
            out.append(Annotation(
                text_id=text_id, construct_id=cid,
                span=Span(s.start_char, s.end_char, lo, hi),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0, evidence={"tokens": list(range(lo, hi + 1)),
                                          "matched": s.text}))

        f11 = self._foc11(doc)
        if f11:
            emit("FOC-11", f11)
        add = self._add_inv(doc)
        if add:
            span, is_pron = add
            emit("FOC-13" if is_pron else "FOC-10", span)
        pp = self._foc10_pp(doc)
        if pp:
            emit("FOC-10", pp)
        f09 = self._foc09(doc)
        if f09:
            emit("FOC-09", f09)
        f12 = self._foc12(doc)
        if f12:
            emit("FOC-12", f12)
        f14 = self._foc14(doc)
        if f14:
            emit("FOC-14", f14)
        return out


# --------------------------------------------------------------------------- #
# FOC-15: wh-intensifiers - a fixed phrase gated by a preceding wh word.
# --------------------------------------------------------------------------- #
_WH = {"what", "who", "whom", "where", "when", "why", "how", "which", "whose"}
_INTENSIFIERS = ["on earth", "the hell", "the heck", "the devil",
                 "in the world", "on god's earth", "ever"]


def _wh_before(doc, start, end):
    return any(doc[i].lower_ in _WH for i in range(0, start))


# --------------------------------------------------------------------------- #
# LLM tier.
# --------------------------------------------------------------------------- #
_CLEFT_IT_SYS = (
    "The sentence is an it-cleft (It + be + focus + that/who-clause). Choose:\n"
    "FOC-01 subject focus (It was Maria who solved it) | FOC-02 object/adjunct "
    "focus (It was the report that I read; It was here that we met) | "
    "FOC-03 'it was not until X that ...'.")
_CLEFT_WH_SYS = (
    "The sentence is a wh-/pseudo-cleft or an all/reason-cleft. Choose:\n"
    "FOC-04 wh-cleft (What I need is a break) | FOC-05 reversed pseudo-cleft "
    "(A break is what I need; That's what I meant) | FOC-06 all/the reason "
    "cleft (All I want is the truth; The reason I called is...).")
_FRONT_SYS = (
    "A constituent is fronted (marked theme). Choose:\n"
    "FOC-07 object/complement fronting (That I can't accept; Brilliant it was "
    "not) | FOC-08 adverbial fronting without inversion (In the corner stood a "
    "lamp / To this day I remember).\n"
    "It-clefts (It was X that ...) and wh-/pseudo-clefts (What I need is ...) "
    "are different constructs - return NONE for those.")
_FOC16_SYS = ("Return FOC-16 for end-weight / end-focus packaging ONLY via "
              "extraposition (It's clear that ...), existential there, or a "
              "passive used to place heavy/new information last. It-clefts "
              "and wh-clefts are different constructs - return NONE for "
              "those.")
_FOC17_SYS = ("Return FOC-17 for right/left dislocation, where a detached NP is "
              "co-referential with a pronoun (My brother, he's a doctor; She's "
              "clever, your daughter). NOT FOC-17: a tail that merely copies "
              "the host clause's pronoun subject and operator ('He's mad, he "
              "is.') or a bare pronoun tail ('I'm not stupid, me.') — those "
              "are QIN-02 statement (copy) tags, the mirror exclude of qin.py.")

_IT_FORM = [
    {"RIGHT_ID": "be", "RIGHT_ATTRS": {"LEMMA": "be"}},
    {"LEFT_ID": "be", "REL_OP": ">", "RIGHT_ID": "it",
     "RIGHT_ATTRS": {"LOWER": "it", "DEP": {"IN": ["nsubj", "expl"]}}},
]
_WHCLEFT_FORM = [{"RIGHT_ID": "w", "RIGHT_ATTRS": {"TAG": {"IN": ["WP", "WDT"]}}}]
_FRONT_FORM = [{"RIGHT_ID": "r", "RIGHT_ATTRS": {"DEP": "ROOT"}}]


class _ThingIsFocus:
    """FOC-18 (Part VI addition): the thing/problem/fact is (, is) + clause.

    The subject noun must come from the thing_is_nouns lexicon and the copula
    must take a CLAUSAL complement (ccomp — covers "that we're broke", bare
    "he never listens" and the double-is variant, where the second "is" is
    itself the ccomp). Predicative "The thing is heavy." (acomp) and wh-clefts
    ("What I need is a break.", no lexicon noun as nsubj) do not fire.
    """
    construct_ids = ["FOC-18"]
    detector_type = "lexicon"
    version = "foc18-thing-is@0.1"

    def __init__(self):
        from ...registry import lexicons
        self._nouns = {l for l in lexicons()["thing_is_nouns"]["lemmas"]
                       if " " not in l}

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lemma_.lower() not in self._nouns or t.dep_ != "nsubj":
                continue
            cop = t.head
            if cop.lemma_ != "be":
                continue
            if not any(k.dep_ == "ccomp" for k in cop.children):
                continue
            lo = min([t.i] + [k.i for k in t.children if k.dep_ in ("det", "poss")])
            hi = cop.i
            span = doc[lo:hi + 1]
            out.append(Annotation(
                text_id=text_id, construct_id="FOC-18",
                span=Span(span.start_char, span.end_char, lo, hi),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text}))
        return out


def build(nlp, client=None):
    return [
        _FocusRules(),
        _ThingIsFocus(),
        PhraseLexiconDetector(nlp, "FOC-15", _INTENSIFIERS, gate=_wh_before,
                              version="foc15-intensifier@0.1"),
        LLMReadingDetector(nlp, ["FOC-01", "FOC-02", "FOC-03"], _IT_FORM,
                           _CLEFT_IT_SYS, client=client, version="foc-itcleft@0.1"),
        LLMReadingDetector(nlp, ["FOC-04", "FOC-05", "FOC-06"], _WHCLEFT_FORM,
                           _CLEFT_WH_SYS, client=client, version="foc-whcleft@0.1"),
        LLMReadingDetector(nlp, ["FOC-07", "FOC-08"], _FRONT_FORM, _FRONT_SYS,
                           client=client, version="foc-fronting@0.1"),
        LLMStandaloneDetector("FOC-16", _FOC16_SYS, client=client,
                              version="foc16-endfocus@0.1"),
        LLMReadingDetector(nlp, ["FOC-17"], _FRONT_FORM, _FOC17_SYS,
                           client=client, version="foc17-disloc@0.1"),
    ]
