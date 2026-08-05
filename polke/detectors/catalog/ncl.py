"""Category NCL - noun / complement clauses (NCL-01..12).

That-clauses split by the host: object of a verb (01), subject/extraposed (02),
appositive after a noun (03), after an adjective (04), that-omission (05),
subjunctive (06). Wh-nominal clauses: finite (07) and wh + to-infinitive (08).
Extraposition/anticipatory it (09). Polar embedded: if/whether (10) and the
whether frames (11). Nominal -ing clause (12).

Verified on en_core_web_sm: the complementiser 'that' is a ``mark`` (tag IN, or
DT when parenthesised); a that-clause object is ``ccomp`` under a VERB, an
appositive that-clause is ``acl`` under a NOUN, and after a predicative
adjective it is ``ccomp`` under the ADJ. Subject that-clauses are ``csubj``.
if/whether are ``mark`` (lemma if/whether). Non-finite whether keeps a TO aux.

LLM tiers (skip offline): NCL-02 subject/extraposed, NCL-05 that-omission,
NCL-09 extraposition, NCL-12 nominal -ing clause.
"""
from __future__ import annotations
from ...registry import lexicons
from ..base import Detector
from ..rules import DependencyRuleDetector
from ..routing import RuleRoutingDetector, CompositeDetector
from ..lexicon import LexiconDetector
from ..llm import LLMStandaloneDetector

# NCL-14 (Part VI addition): noun + of + -ing complement. The head noun must
# come from the noun_of_ing lexicon so ordinary of-PPs ("a photo of a man
# running") do not fire; "look forward TO seeing" and "good AT cooking" fail
# on the preposition, "reason for concern" on the missing -ing clause.
_NOUN_OF_ING = [
    {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}},
    {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "of",
     "RIGHT_ATTRS": {"LOWER": "of", "DEP": "prep"}},
    {"LEFT_ID": "of", "REL_OP": ">", "RIGHT_ID": "ing",
     "RIGHT_ATTRS": {"TAG": "VBG", "DEP": {"IN": ["pcomp", "pobj"]}}},
]


def _ncl14_key(doc, tids):
    return (doc[tids[0]].lemma_.lower(),)


# NCL-13 (Part VI addition): noun + wh-complement ("the question (of)
# whether to stay", "no idea how it works"). The parses of these are
# unreliable (sm attaches "divided" inside the acl), so this is a positional
# scan: NOUN, optional "of"/"as to", then a clause-introducing wh-word with
# verbal material after it. Verb heads ("I wonder whether", "Ask if") never
# have the noun immediately before the wh-word, and NCL-03's "that" is not
# in the wh set.
_NCL13_WH = {"whether", "how", "why", "where", "when", "who", "what"}


def _ncl13(doc):
    for t in doc:
        if t.pos_ != "NOUN":
            continue
        j = t.i + 1
        if j < len(doc) and doc[j].lower_ == "of":
            j += 1
        elif j + 1 < len(doc) and doc[j].lower_ == "as" \
                and doc[j + 1].lower_ == "to":
            j += 2
        if j >= len(doc) or doc[j].lower_ not in _NCL13_WH:
            continue
        if doc[j].dep_ == "det":
            continue                        # "which book" style determiner
        sent = t.sent
        if not any(k.pos_ in ("VERB", "AUX") for k in doc[j + 1:sent.end]):
            continue                        # needs clausal material after
        yield (t.i, j)

_WH_TAGS = {"WP", "WDT", "WP$", "WRB"}
_WH_NOM_DEPS = {"nsubj", "nsubjpass", "dobj", "dative", "attr", "oprd",
                "pobj", "acomp"}

# Mandative triggers (subjunctive that-clause, NCL-06).
_MANDATIVE_VERBS = {"demand", "insist", "suggest", "recommend", "request",
                    "require", "propose", "order", "ask", "urge", "move",
                    "stipulate", "advise", "command", "decree", "beg", "pray",
                    "prefer", "desire", "intend"}
_MANDATIVE_ADJS = {"essential", "important", "vital", "necessary", "crucial",
                   "imperative", "advisable", "desirable", "fitting", "urgent",
                   "mandatory", "obligatory", "preferable", "critical", "key"}


def _is_question(doc):
    for t in reversed(doc):
        if t.is_punct:
            if t.text == "?":
                return True
            continue
        break
    return False


# --------------------------------------------------------------------------- #
# NCL-01 that-clause as object of a verb.
# --------------------------------------------------------------------------- #
_THAT_OBJ = [
    {"RIGHT_ID": "h", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "h", "REL_OP": ">", "RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": "ccomp"}},
    {"LEFT_ID": "c", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"DEP": "mark", "LOWER": "that"}},
]


def _that_obj_exclude(doc, token_ids):
    h = doc[token_ids[0]]
    return h.lemma_.lower() in _MANDATIVE_VERBS       # subjunctive -> NCL-06


# --------------------------------------------------------------------------- #
# NCL-03 appositive that-clause after a noun.
# --------------------------------------------------------------------------- #
_THAT_APPOS = [
    {"RIGHT_ID": "n", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "n", "REL_OP": ">", "RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": "acl"}},
    {"LEFT_ID": "c", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"DEP": "mark", "LOWER": "that"}},
]
# Fragment form ('the idea that...'): the complementiser 'that' hangs directly
# off the noun (dep nmod), no clause. Distinct from a relative 'that' (a
# relativiser under a relcl verb) and from the determiner 'that book' (dep det).
_THAT_APPOS_FRAG = [
    {"RIGHT_ID": "n", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "n", "REL_OP": ">", "RIGHT_ID": "t",
     "RIGHT_ATTRS": {"LOWER": "that", "DEP": {"IN": ["nmod", "mark", "acl"]}}},
]


# --------------------------------------------------------------------------- #
# NCL-04 that-clause after an adjective.
# --------------------------------------------------------------------------- #
_THAT_ADJ = [
    {"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": "ADJ"}},
    {"LEFT_ID": "a", "REL_OP": ">", "RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": "ccomp"}},
    {"LEFT_ID": "c", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"DEP": "mark", "LOWER": "that"}},
]
# Fragment form ('sure/glad/afraid that...'): 'that' hangs directly off the
# predicative adjective (parser tags it prep/dobj) with no following clause.
_THAT_ADJ_FRAG = [
    {"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": "ADJ"}},
    {"LEFT_ID": "a", "REL_OP": ">", "RIGHT_ID": "t",
     "RIGHT_ATTRS": {"LOWER": "that",
                     "DEP": {"IN": ["prep", "dobj", "mark", "advmod", "nmod"]}}},
]


# --------------------------------------------------------------------------- #
# NCL-06 subjunctive that-clause (mandative).
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


def _mandative_trigger(c):
    h = c.head
    if h.lemma_.lower() in _MANDATIVE_VERBS:
        return True
    return any(ch.lemma_.lower() in _MANDATIVE_ADJS
               for ch in h.children if ch.dep_ in ("acomp", "attr", "oprd", "amod"))


def _classify_subjunctive(doc, token_ids):
    c = doc[token_ids[0]]
    if _mandative_trigger(c) and _is_subjunctive(c):
        return "NCL-06"
    return None


# --------------------------------------------------------------------------- #
# NCL-07 wh-nominal clause (finite subject/object/complement).
# --------------------------------------------------------------------------- #
_WH_FORM = [{"RIGHT_ID": "w", "RIGHT_ATTRS": {"TAG": {"IN": list(_WH_TAGS)}}}]


def _classify_wh_nom(doc, token_ids):
    w = doc[token_ids[0]]
    if _is_question(doc):
        return None
    h = w.head
    if h.dep_ == "relcl":
        return None                                  # relative clause -> REL
    if any(a.dep_ == "aux" and a.tag_ == "TO" for a in h.children):
        return None                                  # wh + to-inf -> NCL-08
    if w.tag_ in ("WP", "WDT") and w.dep_ in _WH_NOM_DEPS:
        return "NCL-07"
    if w.tag_ == "WRB" and h.pos_ in ("VERB", "AUX") and \
            h.dep_ in ("ccomp", "xcomp", "acl", "pcomp"):
        return "NCL-07"
    return None


# --------------------------------------------------------------------------- #
# NCL-08 wh- + to-infinitive.
# --------------------------------------------------------------------------- #
_TO_INF = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": "TO"}},
]


def _classify_wh_inf(doc, token_ids):
    v = doc[token_ids[0]]
    if any(c.tag_ in _WH_TAGS for c in v.children):
        return "NCL-08"
    if any(c.dep_ == "mark" and c.lemma_ in ("whether", "if") for c in v.children):
        return "NCL-08"
    return None


# --------------------------------------------------------------------------- #
# NCL-10 if / whether clause (embedded polar, finite).
# --------------------------------------------------------------------------- #
_IF_WH = [
    {"RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "advcl", "acl",
                                                     "pcomp", "xcomp", "dobj"]}}},
    {"LEFT_ID": "c", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"DEP": "mark", "LEMMA": {"IN": ["if", "whether"]}}},
]


def _classify_if_wh(doc, token_ids):
    c = doc[token_ids[0]]
    if any(a.dep_ == "aux" and a.tag_ == "TO" for a in c.children):
        return None                                  # whether + to -> NCL-11
    return "NCL-10"


# --------------------------------------------------------------------------- #
# NCL-11 whether (non-finite / prepositional object).
# --------------------------------------------------------------------------- #
class _WhetherDetector(Detector):
    detector_type = "lexicon"
    construct_ids = ["NCL-11"]

    def __init__(self, version="ncl11-whether@0.1"):
        self.version = version

    def match(self, doc, text_id="doc"):
        from ...schema import Annotation, Span
        out, seen = [], set()
        for t in doc:
            if t.lower_ != "whether":
                continue
            fire = False
            if t.dep_ in ("pobj", "pcomp"):
                fire = True                          # the question of whether
            elif t.dep_ == "mark":
                h = t.head
                if any(a.dep_ == "aux" and a.tag_ == "TO" for a in h.children):
                    fire = True                      # whether (or not) to VERB
            if not fire or t.i in seen:
                continue
            seen.add(t.i)
            out.append(Annotation(
                text_id=text_id, construct_id="NCL-11",
                span=Span(t.idx, t.idx + len(t.text), t.i, t.i),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": [t.i], "matched": t.text}))
        return out


# --------------------------------------------------------------------------- #
# LLM tier (registered, skipped offline).
# --------------------------------------------------------------------------- #
_SUBJ_SYS = (
    "Return NCL-02 if a that-clause is the grammatical subject or its "
    "extraposed 'it ... that' variant ('That he lied is clear' / 'It is clear "
    "that he lied')."
)
_OMIT_SYS = (
    "Return NCL-05 if a that-clause object appears with the complementiser "
    "'that' omitted ('I know Ø you're busy')."
)
_EXTRA_SYS = (
    "Return NCL-09 if the sentence uses anticipatory 'it' as an extraposed "
    "subject or object ('It's no use complaining' / 'I find it odd that...')."
)
_ING_SYS = (
    "Return NCL-12 if an -ing clause functions nominally (as subject or object) "
    "('His leaving early surprised us' / 'I appreciate your helping')."
)


def build(nlp, client=None):
    dets = []

    dets.append(DependencyRuleDetector(
        nlp, "NCL-01", _THAT_OBJ, exclude=_that_obj_exclude,
        version="ncl01-thatobj@0.1"))
    dets.append(CompositeDetector("NCL-03", [
        DependencyRuleDetector(nlp, "NCL-03", _THAT_APPOS),
        DependencyRuleDetector(nlp, "NCL-03", _THAT_APPOS_FRAG),
    ], detector_type="lexicon", version="ncl03-appos@0.1"))
    dets.append(CompositeDetector("NCL-04", [
        DependencyRuleDetector(nlp, "NCL-04", _THAT_ADJ),
        DependencyRuleDetector(nlp, "NCL-04", _THAT_ADJ_FRAG),
    ], detector_type="lexicon", version="ncl04-adj@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NCL-06"], _THAT_CL, _classify_subjunctive,
        version="ncl06-subjunctive@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NCL-07"], _WH_FORM, _classify_wh_nom, version="ncl07-whnom@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NCL-08"], _TO_INF, _classify_wh_inf, version="ncl08-whinf@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NCL-10"], _IF_WH, _classify_if_wh, version="ncl10-ifwh@0.1"))
    dets.append(_WhetherDetector())
    dets.append(LexiconDetector(
        nlp, "NCL-14", _NOUN_OF_ING,
        {(l,) for l in lexicons()["noun_of_ing"]["lemmas"]},
        _ncl14_key, version="ncl14-noun-of-ing@0.1"))
    from .common import Scan
    dets.append(Scan("NCL-13", _ncl13, version="ncl13-noun-wh@0.1"))

    # LLM tier.
    dets.append(LLMStandaloneDetector(
        "NCL-02", _SUBJ_SYS, client=client, version="ncl02-subj@0.1"))
    dets.append(LLMStandaloneDetector(
        "NCL-05", _OMIT_SYS, client=client, version="ncl05-omit@0.1"))
    dets.append(LLMStandaloneDetector(
        "NCL-09", _EXTRA_SYS, client=client, version="ncl09-extra@0.1"))
    dets.append(LLMStandaloneDetector(
        "NCL-12", _ING_SYS, client=client, version="ncl12-ing@0.1"))

    return dets
