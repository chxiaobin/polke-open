"""Category MOD - modality (MOD-01..37).

Modal verbs + semi-modals across meanings (ability / permission / possibility /
obligation / deduction / advice / volition). The modal+MEANING is the LLM tier;
the FORM (semi-modal periphrases, negated necessity, had better, would rather) is
routable/lexicon.

Rule/lexicon tier (tested offline):
  MOD-03/04   be able to, split by finiteness: was/were able to & managed to (03)
              vs been/other be able to (04)
  MOD-08/18/19  be + past participle + to-inf, one lexicon-routed table:
              allowed/permitted (08), supposed/meant (18), required/obliged (19)
  MOD-16      have to / have got to
  MOD-17      need to (affirmative)
  MOD-21      don't have to / needn't (+bare) / don't need to  (absence of necessity)
  MOD-26      had better
  MOD-37      would rather / would sooner / would prefer
LLM tier (registered, skipped offline): the modal+meaning readings, grouped by
communicative family, plus perfect modals and 'didn't need to'.
"""
from __future__ import annotations
from ...registry import lexicons
from ...schema import Annotation, Span
from ..base import Detector
from ..routing import RuleRoutingDetector, LexiconRuleDetector, CompositeDetector
from ..rules import DependencyRuleDetector
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector

# be + PARTICIPLE + to  ->  which obligation/permission semi-modal.
_BE_PP_ROUTE = {
    "allow": "MOD-08", "permit": "MOD-08",
    "suppose": "MOD-18", "mean": "MOD-18",
    "require": "MOD-19", "oblige": "MOD-19",
}


# --------------------------------------------------------------------------- #
# FORM patterns.
# --------------------------------------------------------------------------- #
# "able" as an adjectival complement of a be verb ( ... be able to VERB ).
_ABLE_FORM = [
    {"RIGHT_ID": "able", "RIGHT_ATTRS": {"LOWER": "able", "DEP": "acomp"}},
]
# "managed to VERB"
_MANAGE_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"LEMMA": "manage"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp"}},
]
# be + VBN(participle) + to-infinitive
_BE_PP_TO = [
    {"RIGHT_ID": "p", "RIGHT_ATTRS": {"TAG": "VBN"}},
    {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "be",
     "RIGHT_ATTRS": {"DEP": "auxpass", "LEMMA": "be"}},
    {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp"}},
]
# have (finite) heading a to-infinitive: "have to VERB"
_HAVE_TO = [
    {"RIGHT_ID": "h", "RIGHT_ATTRS": {"LEMMA": "have",
                                      "TAG": {"IN": ["VBP", "VBZ", "VBD"]}}},
    {"LEFT_ID": "h", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp", "TAG": "VB"}},
]
# have + got + to VERB  ("have got to")
_HAVE_GOT_TO = [
    {"RIGHT_ID": "g", "RIGHT_ATTRS": {"LEMMA": "get", "TAG": "VBN"}},
    {"LEFT_ID": "g", "REL_OP": ">", "RIGHT_ID": "h",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "have"}},
    {"LEFT_ID": "g", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp", "TAG": "VB"}},
]
# have (any tag) heading a to-infinitive - for the NEGATED 'don't have to'
# where do-support leaves 'have' bare (TAG VB).
_HAVE_TO_ANY = [
    {"RIGHT_ID": "h", "RIGHT_ATTRS": {"LEMMA": "have"}},
    {"LEFT_ID": "h", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp", "TAG": "VB"}},
]
# need (finite lexical verb) + to-infinitive
_NEED_TO = [
    {"RIGHT_ID": "n", "RIGHT_ATTRS": {"LEMMA": "need",
                                      "TAG": {"IN": ["VBP", "VBZ", "VBD"]}}},
    {"LEFT_ID": "n", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp", "TAG": "VB"}},
]
# any modal + main verb (for the reading groups)
_MODAL_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "md",
     "RIGHT_ATTRS": {"TAG": "MD"}},
]
# had/'d better + bare verb
_HAD_BETTER = [
    {"RIGHT_ID": "b", "RIGHT_ATTRS": {"LOWER": "better", "DEP": "advmod"}},
]


# --------------------------------------------------------------------------- #
# Classifiers / keys.
# --------------------------------------------------------------------------- #
def _neg_children(tok):
    return any(c.dep_ == "neg" for c in tok.children)


def classify_able(doc, tids):
    """... be able to ...  ->  MOD-03 (was/were) or MOD-04 (been / other be)."""
    able = doc[tids[0]]
    be = able.head
    if be.lemma_ != "be":
        return None
    # require a to-infinitive complement of 'able'
    if not any(c.dep_ == "xcomp" for c in able.children):
        return None
    if be.tag_ == "VBD":                     # was/were able to
        return "MOD-03"
    return "MOD-04"                          # been / am / is / are / will be able to


def _manage_ok(doc, tids):
    return "MOD-03"


def _be_pp_key(doc, tids):
    p = doc[tids[0]]
    # participle must take a to-infinitive (be supposed TO wear).
    if not any(x.dep_ == "xcomp" and any(g.tag_ == "TO" for g in x.children)
               for x in p.children):
        return None
    return p.lemma_


def _be_pp_route(key):
    return _BE_PP_ROUTE.get(key)


def classify_have_to(doc, tids):
    """have to / have got to  ->  MOD-16 (affirmative only)."""
    h = doc[tids[0]]                          # the 'have' or the 'got'
    # exclude negation (negated -> MOD-21 absence of necessity)
    trigger = h
    if _neg_children(trigger):
        return None
    return "MOD-16"


def classify_need_to(doc, tids):
    n = doc[tids[0]]
    if _neg_children(n):
        return None                          # negated -> MOD-21/23
    return "MOD-17"


def classify_absence(doc, tids):
    """Negated necessity: don't have to / needn't (+bare) / don't need to -> MOD-21.

    NOT: needn't have + pp (MOD-22), didn't need to (MOD-23), mustn't (MOD-24).
    """
    h = doc[tids[0]]
    # 'have to' negated by do-support: have + neg + to-inf
    if h.lemma_ == "have" and _neg_children(h) \
            and any(c.dep_ == "xcomp" for c in h.children):
        return "MOD-21"
    # needn't + bare infinitive, with NO perfect 'have'
    if _neg_children(h):
        aux = [c for c in h.children if c.dep_ in ("aux", "auxpass")]
        needs = [a for a in aux if a.lemma_ == "need"]
        haves = [a for a in aux if a.lemma_ == "have"]
        dos = [a for a in aux if a.lemma_ == "do"]
        if needs and not haves and not dos and h.tag_ == "VB":
            return "MOD-21"                  # needn't do
    return None


def _had_better_ok(doc, tids):
    b = doc[tids[0]]                          # 'better'
    head = b.head
    if any(c.dep_ == "aux" and c.lower_ in ("'d", "had") for c in head.children):
        return "MOD-26"
    return None


# --------------------------------------------------------------------------- #
# LLM reading prompts, grouped by communicative family.
# --------------------------------------------------------------------------- #
_ABILITY_SYS = (
    "The bracketed modal expresses ABILITY. Choose one:\nMOD-01 can = present "
    "ability (I can swim) | MOD-02 could = past/general ability (I could read at "
    "four) | MOD-05 could have + pp = unrealised past ability (I could have gone, "
    "but I didn't)."
)
_PERMISSION_SYS = (
    "The bracketed modal expresses PERMISSION. Choose one:\nMOD-06 can (You can "
    "use my phone) | MOD-07 could/may/might, more formal/tentative (May I leave "
    "early?)."
)
_REQOFF_SYS = (
    "The bracketed modal is interpersonal. Choose one:\nMOD-09 request with can/"
    "could/will/would you (Could you help me?) | MOD-10 offer/suggestion: shall "
    "I/we, why don't we, how about, let's (Shall I open the window?)."
)
_POSS_SYS = (
    "The bracketed modal expresses epistemic POSSIBILITY. Choose one:\nMOD-11 "
    "may/might/could + base (It may snow; they might be late) | MOD-12 may/might/"
    "could have + pp = past possibility (She may have missed the train) | MOD-13 "
    "can = general/theoretical possibility (It can get very cold here) | MOD-14 "
    "could/might = tentative suggestion (You could try restarting it)."
)
_OBLIG_SYS = (
    "The bracketed modal expresses OBLIGATION/NECESSITY. Choose one:\nMOD-15 must "
    "= speaker-internal obligation (I must finish this today) | MOD-20 shall = "
    "legalistic obligation (Tenants shall maintain the property)."
)
_ABSENCE_SYS = (
    "The bracketed form is about ABSENCE OF NECESSITY or PROHIBITION. Choose one:\n"
    "MOD-22 needn't have + pp = unnecessary past action done (You needn't have "
    "paid) | MOD-23 didn't need to = it was unnecessary (I didn't need to wait) | "
    "MOD-24 mustn't / can't / not allowed to / may not = prohibition (You mustn't "
    "smoke here)."
)
_ADVICE_SYS = (
    "The bracketed modal gives ADVICE. Choose one:\nMOD-25 should/ought to (You "
    "should rest) | MOD-27 should/ought to/could/might have + pp = past criticism "
    "(You should have told me) | MOD-28 why don't you / why not / it might be a "
    "good idea to (Why not ask her?)."
)
_DEDUCT_SYS = (
    "The bracketed modal expresses epistemic DEDUCTION/CERTAINTY. Choose one:\n"
    "MOD-29 must = confident deduction (That must be the postman) | MOD-30 can't/"
    "couldn't = negative deduction (He can't be serious) | MOD-31 must have/can't "
    "have + pp = deduction about the past (They must have left already) | MOD-32 "
    "should/ought to = expectation (The parcel should arrive today) | MOD-33 will/"
    "would = confident assumption (That'll be the courier; he'd be about fifty)."
)
_VOLITION_SYS = (
    "The bracketed modal expresses VOLITION/HABIT/PREFERENCE. Choose one:\nMOD-34 "
    "will/won't = willingness/insistence of things (The door won't open) | MOD-35 "
    "would = past habit (On Sundays we would visit grandma) | MOD-36 will = "
    "characteristic/general truth (Oil will float on water)."
)
_MOD23_SYS = ("Return MOD-23 if 'didn't need to' says an action was UNNECESSARY "
              "(and typically not done): I didn't need to wait.")
_MOD33_SYS = ("Return MOD-33 if will/would expresses a confident epistemic "
              "assumption (That'll be the courier; he'd be about fifty now).")
_MOD36_SYS = ("Return MOD-36 if 'will' states a characteristic or general truth "
              "(Oil will float on water).")


# --------------------------------------------------------------------------- #
def build(nlp, client=None):
    dets = []

    # === Ability ============================================================
    # MOD-03 / MOD-04: be able to (+ managed to for MOD-03).
    dets.append(CompositeDetector(
        ["MOD-03", "MOD-04"],
        [RuleRoutingDetector(nlp, ["MOD-03", "MOD-04"], _ABLE_FORM,
                             classify_able, version="mod-able@0.1"),
         RuleRoutingDetector(nlp, ["MOD-03"], _MANAGE_FORM, _manage_ok,
                             version="mod03-managed@0.1")],
        detector_type="lexicon", version="mod-ability-semimodal@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["MOD-01", "MOD-02", "MOD-05"], _MODAL_FORM, _ABILITY_SYS,
        client=client, version="mod-ability@0.1"))

    # === Permission =========================================================
    # MOD-08 be allowed/permitted to (shares the be+pp+to table below).
    dets.append(LLMReadingDetector(
        nlp, ["MOD-06", "MOD-07"], _MODAL_FORM, _PERMISSION_SYS,
        client=client, version="mod-permission@0.1"))

    # === be + past participle + to  (MOD-08 / MOD-18 / MOD-19) ==============
    dets.append(LexiconRuleDetector(
        nlp, ["MOD-08", "MOD-18", "MOD-19"], _BE_PP_TO, set(_BE_PP_ROUTE),
        _be_pp_key, route=_be_pp_route, version="mod-be-pp-to@0.1"))

    # === Requests / offers / suggestions (LLM) =============================
    dets.append(LLMReadingDetector(
        nlp, ["MOD-09", "MOD-10"], _MODAL_FORM, _REQOFF_SYS, client=client,
        version="mod-req-offer@0.1"))

    # === Possibility (LLM) =================================================
    dets.append(LLMReadingDetector(
        nlp, ["MOD-11", "MOD-12", "MOD-13", "MOD-14"], _MODAL_FORM, _POSS_SYS,
        client=client, version="mod-possibility@0.1"))

    # === Obligation & necessity ============================================
    dets.append(LLMReadingDetector(
        nlp, ["MOD-15", "MOD-20"], _MODAL_FORM, _OBLIG_SYS, client=client,
        version="mod-obligation@0.1"))
    # MOD-16 have to / have got to
    dets.append(CompositeDetector(
        ["MOD-16"],
        [RuleRoutingDetector(nlp, ["MOD-16"], _HAVE_TO, classify_have_to,
                             version="mod16-have-to@0.1"),
         RuleRoutingDetector(nlp, ["MOD-16"], _HAVE_GOT_TO, classify_have_to,
                             version="mod16-have-got-to@0.1")],
        detector_type="lexicon", version="mod16-composite@0.1"))
    # MOD-17 need to
    dets.append(RuleRoutingDetector(
        nlp, ["MOD-17"], _NEED_TO, classify_need_to, version="mod17-need-to@0.1"))

    # === Absence of necessity & prohibition ================================
    # MOD-21 don't have to / needn't (+bare) / don't need to
    dets.append(CompositeDetector(
        ["MOD-21"],
        [RuleRoutingDetector(nlp, ["MOD-21"], _HAVE_TO_ANY, classify_absence,
                             version="mod21-have-to@0.1"),
         RuleRoutingDetector(nlp, ["MOD-21"], _MODAL_FORM, classify_absence,
                             version="mod21-neednt@0.1")],
        detector_type="lexicon", version="mod21-composite@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["MOD-22", "MOD-24"], _MODAL_FORM, _ABSENCE_SYS, client=client,
        version="mod-absence@0.1"))
    dets.append(LLMStandaloneDetector("MOD-23", _MOD23_SYS, client=client,
                                      version="mod23-didnt-need@0.1"))

    # === Advice / recommendation ===========================================
    dets.append(LLMReadingDetector(
        nlp, ["MOD-25", "MOD-27", "MOD-28"], _MODAL_FORM, _ADVICE_SYS,
        client=client, version="mod-advice@0.1"))
    # MOD-26 had better
    dets.append(RuleRoutingDetector(
        nlp, ["MOD-26"], _HAD_BETTER, _had_better_ok, version="mod26-had-better@0.1"))

    # === Deduction / certainty (LLM) =======================================
    dets.append(LLMReadingDetector(
        nlp, ["MOD-29", "MOD-30", "MOD-31", "MOD-32"], _MODAL_FORM, _DEDUCT_SYS,
        client=client, version="mod-deduction@0.1"))
    dets.append(LLMStandaloneDetector("MOD-33", _MOD33_SYS, client=client,
                                      version="mod33-will-would@0.1"))

    # === Volition / habit / preference =====================================
    dets.append(LLMReadingDetector(
        nlp, ["MOD-34", "MOD-35"], _MODAL_FORM, _VOLITION_SYS, client=client,
        version="mod-volition@0.1"))
    dets.append(LLMStandaloneDetector("MOD-36", _MOD36_SYS, client=client,
                                      version="mod36-will-truth@0.1"))
    # MOD-37 would rather / would sooner / would prefer
    dets.append(PhraseLexiconDetector(
        nlp, "MOD-37", ["would rather", "would sooner", "would prefer",
                        "'d rather", "'d sooner", "'d prefer"],
        version="mod37-would-rather@0.1"))

    # === Part VI additions: epistemic periphrastics + dare =================
    dets.append(_EpistemicPeriphrastic(
        "MOD-38", "certainty_adjs", allow_that=False))
    dets.append(_EpistemicPeriphrastic(
        "MOD-39", "likelihood_adjs", allow_that=True))
    dets.append(_DareModal())

    return dets


# === Part VI additions (Phase 2, lexicon tier) =============================

def _mk_ann(det, doc, cid, lo, hi, text_id):
    span = doc[lo:hi + 1]
    return Annotation(
        text_id=text_id, construct_id=cid,
        span=Span(span.start_char, span.end_char, lo, hi),
        detector_type=det.detector_type, detector_version=det.version,
        confidence=1.0,
        evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text})


class _EpistemicPeriphrastic(Detector):
    """MOD-38 be bound/sure/certain to + V; MOD-39 be (un)likely to + V and
    the extraposed "it is (un)likely that ...". The adjective needs a be-form
    on its left (copula or passive aux — sm parses "bound" as VBN+auxpass)
    and to+V (or, for MOD-39, "that") on its right; "sure that he'll notice"
    and "bound for London" therefore stay out of MOD-38."""
    detector_type = "lexicon"

    def __init__(self, cid, lex_key, allow_that):
        self.construct_ids = [cid]
        self.version = f"{cid.lower()}-periphrastic@0.1"
        self._items = set(lexicons()["epistemic_periphrastics"][lex_key])
        self._allow_that = allow_that

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ not in self._items:
                continue
            be_left = (
                any(k.dep_ in ("aux", "auxpass", "cop") and k.lemma_ == "be"
                    for k in t.children)
                or t.head.lemma_ == "be"
                or (t.i > 0 and doc[t.i - 1].lemma_ == "be"))
            if not be_left:
                continue
            nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
            to_inf = (nxt is not None and nxt.lower_ == "to"
                      and t.i + 2 < len(doc) and doc[t.i + 2].pos_ in ("VERB", "AUX"))
            that_cl = (self._allow_that and nxt is not None
                       and nxt.lower_ == "that")
            if not (to_inf or that_cl):
                continue
            hi = t.i + 2 if to_inf else t.i + 1
            out.append(_mk_ann(self, doc, self.construct_ids[0], t.i, hi, text_id))
        return out


class _DareModal(Detector):
    """MOD-40 dare in its modal/blend uses: negated bare "daren't/dare not",
    "how dare ...", and inverted "Dare I say it". Main-verb object-control
    "He dared me to jump" and the noun "a silly dare" match none of the
    three frames."""
    detector_type = "lexicon"
    version = "mod40-dare@0.1"
    construct_ids = ["MOD-40"]

    def match(self, doc, text_id="doc"):
        out = []
        for sent in doc.sents:
            first = next((t for t in sent if not t.is_punct), None)
            for t in sent:
                if t.lower_ not in ("dare", "dares", "dared"):
                    continue
                nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
                negated = nxt is not None and nxt.lower_ in ("n't", "not")
                how = t.i > 0 and doc[t.i - 1].lower_ == "how"
                inverted = (first is not None and t.i == first.i
                            and nxt is not None and nxt.pos_ == "PRON")
                if negated or how or inverted:
                    lo = t.i - 1 if how else t.i
                    hi = t.i + 1 if (negated or inverted) else t.i
                    out.append(_mk_ann(self, doc, "MOD-40", lo, hi, text_id))
        return out
