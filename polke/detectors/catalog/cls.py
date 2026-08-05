"""Category CLS - basic clause patterns (CLS-01..17).

Clause types route on the dependency frame of the main (ROOT) verb:
  CLS-01 SV (intransitive) | CLS-02 SVC (acomp/attr copular complement) |
  CLS-03 SVO (dobj) | CLS-04 SVOO (dative + dobj) | CLS-05 SVOC (dobj + oprd)
  are mutually exclusive by object/complement children -> ONE RuleRoutingDetector.
  CLS-06 SVA/SVOA (obligatory adverbial with a locative/position verb) and
  CLS-07 canonical SVO with a full lexical-NP subject are separate detectors
  (they overlap CLS-01/03 and cannot share the exclusive router).
Agreement: CLS-08 basic 3sg -s (simple singular subject + VBZ). The tricky
  agreement siblings (CLS-09 coordinate, CLS-10 quantifier-headed, CLS-11
  measure, CLS-12 notional) are the LLM tier.
Existential/dummy: CLS-13 there is/are (finite be), CLS-14 there + modal/
  perfect/seem, CLS-15 change/perception copular verbs, CLS-17 seem/appear +
  to-infinitive. CLS-16 (adj vs adverb after a copula) is the LLM tier.

spaCy en_core_web_sm labels verified empirically: dative = indirect object,
oprd = object predicative complement, attr/acomp = subject complement of be,
expl = existential 'there', csubj = clausal subject (what-clauses).
"""
from __future__ import annotations
from ..base import Detector
from ..routing import RuleRoutingDetector
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector
from ...schema import Annotation, Span

_COPULAR = {"become", "get", "turn", "grow", "go", "come", "seem", "appear",
            "look", "sound", "feel", "taste", "smell", "remain", "stay",
            "prove", "keep"}
# Verbs that require an obligatory adverbial (SVA / SVOA).
_ADVERBIAL_VERBS = {"live", "put", "lie", "stand", "sit", "stay", "remain",
                    "place", "set", "lean", "hang", "belong", "reside",
                    "dwell", "lay", "head", "reach", "point", "rest",
                    "locate", "situate"}
# Quantifier/measure nouns whose agreement is notional (defer to LLM tier).
_QUANT_HEADS = {"number", "none", "majority", "minority", "lot", "couple",
                "group", "pair", "half", "rest", "plenty", "handful", "bunch",
                "variety", "range", "series"}
_SG_PRON = {"he", "she", "it"}


class _Scan(Detector):
    def __init__(self, cid, fn, detector_type="rule", version="0.1"):
        self.construct_ids = [cid]
        self._fn = fn
        self.detector_type = detector_type
        self.version = version

    def match(self, doc, text_id="doc"):
        out, seen = [], set()
        for lo, hi in self._fn(doc):
            if (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            span = doc[lo:hi + 1]
            out.append(Annotation(
                text_id=text_id, construct_id=self.construct_ids[0],
                span=Span(span.start_char, span.end_char, lo, hi),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text}))
        return out


# --- CLS-01..05: the object/complement frame of the ROOT verb --------------- #
_ROOT_SUBJ_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "ROOT"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "s",
     "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
]


def _classify_frame(doc, tids):
    v = doc[tids[0]]
    deps = {c.dep_ for c in v.children}
    if "dative" in deps and "dobj" in deps:
        return "CLS-04"                                # SVOO
    if "dobj" in deps and "oprd" in deps:
        return "CLS-05"                                # SVOC
    if "acomp" in deps:
        return "CLS-02"                                # SVC (adjective)
    if "attr" in deps and v.lemma_ == "be":
        return "CLS-02"                                # SVC (noun, be)
    if "dobj" in deps:
        if "dative" in deps:
            return "CLS-04"
        return "CLS-03"                                # SVO
    # intransitive: only a subject (+ aux/advmod/punct), no object, complement
    # or obligatory PP adverbial.
    if deps & {"prep", "advcl", "npadvmod", "xcomp", "ccomp", "dative",
               "oprd", "attr", "acomp", "agent"}:
        return None                                    # SVA -> CLS-06, etc.
    if v.pos_ in ("VERB", "AUX"):
        return "CLS-01"                                # SV
    return None


# --- CLS-06 SVA/SVOA: obligatory adverbial with a locative verb ------------- #
def _cls06(doc):
    for v in doc:
        if v.pos_ != "VERB" or v.lemma_.lower() not in _ADVERBIAL_VERBS:
            continue
        prep = [c for c in v.children if c.dep_ in ("prep", "advmod")
                and c.pos_ in ("ADP", "ADV")]
        has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in v.children)
        if prep and has_subj:
            hi = max(x.i for x in prep[0].subtree)
            yield (v.i, max(hi, v.i))


# --- CLS-07 canonical SVO with a full lexical-NP subject -------------------- #
def _cls07(doc):
    for v in doc:
        if v.dep_ != "ROOT" or v.pos_ != "VERB" or v.lemma_ == "be":
            continue
        subj = [c for c in v.children if c.dep_ == "nsubj"]
        objs = [c for c in v.children if c.dep_ == "dobj"]
        deps = {c.dep_ for c in v.children}
        if not subj or not objs:
            continue
        if deps & {"dative", "oprd"}:
            continue
        if subj[0].pos_ in ("NOUN", "PROPN"):          # not a pronoun subject
            yield (v.i, objs[0].i if objs[0].i > v.i else v.i)


# --- CLS-08 basic 3sg -s agreement ------------------------------------------ #
# Sibling routing (Part VI): a clause showing systematic vernacular concord
# (VER-03: "we was", "he don't", "them things is") must not be scored as a
# CLS-08 hit or miss — the speaker's grammar is consistent, just non-standard.
# The shared predicate lives in ver.py so the two detectors cannot drift.
def _cls08(doc):
    from .ver import nonstandard_concord
    for v in doc:
        if v.tag_ != "VBZ":
            continue
        if nonstandard_concord(v):
            continue                        # VER-03 territory, not CLS-08
        subj = [c for c in v.children if c.dep_ == "nsubj"]
        if not subj:
            continue
        s = subj[0]
        # simple singular subject: he/she/it or a singular common/proper noun.
        if s.tag_ == "PRP" and s.lower_ in _SG_PRON:
            ok = True
        elif s.tag_ in ("NN", "NNP") and s.lemma_.lower() not in _QUANT_HEADS:
            ok = True
        else:
            ok = False
        # exclude "the number/lot ... of <plural> ..." notional subjects.
        if any(c.dep_ == "prep" and c.lower_ == "of" for c in s.children):
            ok = False
        if ok:
            yield (s.i, v.i)


# --- CLS-13 / CLS-14 existential --------------------------------------------- #
def _cls13(doc):
    for v in doc:
        if v.lemma_ != "be" or v.dep_ != "ROOT":
            continue
        if not any(c.dep_ == "expl" for c in v.children):
            continue
        if any(c.tag_ == "MD" for c in v.children):
            continue                                   # modal -> CLS-14
        if any(c.dep_ == "aux" and c.lemma_ == "have" for c in v.children):
            continue                                   # perfect -> CLS-14
        if v.tag_ not in ("VBZ", "VBP", "VBD"):
            continue                                   # non-finite -> CLS-14
        yield (v.i, v.i)


def _cls14(doc):
    for v in doc:
        expl = any(c.dep_ == "expl" for c in v.children)
        if not expl:
            continue
        if v.lemma_ == "be" and (any(c.tag_ == "MD" for c in v.children)
                                 or any(c.dep_ == "aux" and c.lemma_ == "have"
                                        for c in v.children)):
            yield (v.i, v.i)                            # modal/perfect existential
        elif v.lemma_ in ("seem", "appear"):
            yield (v.i, v.i)                            # There seems to be ...


# --- CLS-15 change/perception copular verbs --------------------------------- #
def _cls15(doc):
    for t in doc:
        if t.pos_ not in ("VERB", "AUX") or t.lemma_.lower() not in _COPULAR:
            continue
        if any(c.dep_ == "acomp" for c in t.children):
            continue                                   # copula + adj -> CLS-16
        # seem/appear + to-infinitive -> CLS-17
        if any(c.dep_ == "xcomp" and any(g.tag_ == "TO" for g in c.children)
               for c in t.children):
            continue
        yield (t.i, t.i)


# --- CLS-17 seem/appear + to-infinitive ------------------------------------- #
def _cls17(doc):
    for v in doc:
        if v.lemma_ not in ("seem", "appear"):
            continue
        if any(c.dep_ == "expl" for c in v.children):
            continue                                   # existential -> CLS-14
        xc = [c for c in v.children if c.dep_ == "xcomp"]
        if xc and any(g.tag_ == "TO" for g in xc[0].children):
            yield (v.i, xc[0].i)


# --- LLM tier forms --------------------------------------------------------- #
_COORD_SUBJ_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "s",
     "RIGHT_ATTRS": {"DEP": "nsubj"}},
    {"LEFT_ID": "s", "REL_OP": ">", "RIGHT_ID": "c",
     "RIGHT_ATTRS": {"DEP": "conj"}},
]
_QUANT_SUBJ_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "s",
     "RIGHT_ATTRS": {"DEP": "nsubj"}},
]
_COP_ADJ_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "a",
     "RIGHT_ATTRS": {"DEP": "acomp"}},
]


def build(nlp, client=None):
    dets = []

    # CLS-01..05 clause-type router.
    dets.append(RuleRoutingDetector(
        nlp, ["CLS-01", "CLS-02", "CLS-03", "CLS-04", "CLS-05"],
        _ROOT_SUBJ_FORM, _classify_frame, version="cls-frame@0.1"))

    dets.append(_Scan("CLS-06", _cls06, version="cls06-sva@0.1"))
    dets.append(_Scan("CLS-07", _cls07, version="cls07-canonical@0.1"))
    dets.append(_Scan("CLS-08", _cls08, version="cls08-agreement@0.1"))
    dets.append(_Scan("CLS-13", _cls13, version="cls13-existential@0.1"))
    dets.append(_Scan("CLS-14", _cls14, version="cls14-existmodal@0.1"))
    dets.append(_Scan("CLS-15", _cls15, detector_type="lexicon",
                      version="cls15-copular@0.1"))
    dets.append(_Scan("CLS-17", _cls17, version="cls17-seem-toinf@0.1"))

    # --- LLM tier (registered, skipped offline) ---
    dets.append(LLMReadingDetector(
        nlp, ["CLS-09"], _COORD_SUBJ_FORM,
        "The subject is coordinated. Return CLS-09 if agreement follows "
        "coordinate rules: 'and' -> plural verb; 'or/nor/either...or' -> "
        "proximity (verb agrees with the nearest conjunct).",
        client=client, version="cls09-coord@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["CLS-10"], _QUANT_SUBJ_FORM,
        "Return CLS-10 if the subject is quantifier-headed and agreement is "
        "notional: each/either/neither/none (sg or pl), 'a number of X are' "
        "(plural) vs 'the number of X is' (singular).",
        client=client, version="cls10-quantifier@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["CLS-11"], _QUANT_SUBJ_FORM,
        "Return CLS-11 if a measure/amount subject takes a singular verb "
        "(Ten years is a long time; Five euros is enough).",
        client=client, version="cls11-measure@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["CLS-12"], _QUANT_SUBJ_FORM,
        "Return CLS-12 for notional agreement: a what-clause subject (What we "
        "need is/are...) or a British collective taking a plural verb (The team "
        "are).",
        client=client, version="cls12-notional@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["CLS-16"], _COP_ADJ_FORM,
        "A copular/perception verb takes a predicative complement. Return CLS-16 "
        "if the point is the adjective-vs-adverb choice after the copula (It "
        "looks good, not well; She seems nice).",
        client=client, version="cls16-adjadv@0.1"))
    return dets
