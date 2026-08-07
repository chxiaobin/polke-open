"""Category NFV - non-finite verb forms (infinitives & -ing) (NFV-01..31).

Three families:
  To-infinitive (01..08): purpose, negative purpose, after verbs/adjectives/nouns,
    too/enough, wh-words, result.
  Bare infinitive (09..14): after modals, let/make, perception, had-better/why,
    rather-than, help.
  Gerund / -ing (15..20): subject, object, after preposition, possessive, be
    complement, fixed expressions.
  Infinitive-vs-ing verb selection (21..25) and the perfect/passive/participial
    non-finites (26..31).

Routing keys verified empirically on en_core_web_sm:
  purpose to-inf = advcl (xcomp = verb complement, acl = noun postmod);
  modal = MD aux; the non-finite perfect aux 'having' lemmatises to 'having'
  (not 'have'), while the passive 'be/being' lemmatises to 'be'.

LLM tiers (skip offline): NFV-01 purpose reading, NFV-08 result, NFV-11
perception reading, NFV-15/18/19 gerund readings, NFV-24/25 meaning-change,
NFV-28/29 participle-clause readings, NFV-31 would-like specific/general.
"""
from __future__ import annotations
from ..base import Detector
from ..rules import DependencyRuleDetector
from ..routing import RuleRoutingDetector, LexiconRuleDetector
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector

_WH_TAGS = {"WP", "WDT", "WP$", "WRB"}
_DEGREE = {"too", "enough"}
_PREF = {"better", "rather", "sooner"}

# Subject-control verbs that take a to-infinitive complement (NFV-03).
_CONTROL_VERBS = {
    "want", "decide", "hope", "manage", "refuse", "agree", "offer", "promise",
    "plan", "expect", "learn", "arrange", "afford", "seem", "tend", "fail",
    "pretend", "deserve", "threaten", "prepare", "decline", "hesitate",
    "demand", "swear", "vow", "aim", "care", "dare", "resolve", "volunteer",
    "consent", "attempt", "wish", "intend", "claim", "prove", "appear",
    "happen", "choose", "prepare", "seek", "strive", "long", "opt",
}

# Verb-selection fragment lexicons (NFV-21/22/23). Curated to be DISJOINT and,
# crucially, to avoid any lemma that appears in a *sibling* fragment (the
# contract negatives are the other fragments, incl. NFV-25's "I like to check
# (choose to) / I like checking (enjoy)"): so NFV-21 excludes 'choose', NFV-22
# excludes 'enjoy', NFV-23 excludes 'like'.
_INF_ONLY = {  # NFV-21
    "want", "decide", "hope", "manage", "refuse", "agree", "offer", "promise",
    "plan", "expect", "learn", "arrange", "afford", "seem", "tend", "fail",
    "pretend", "deserve", "threaten", "prepare", "decline", "hesitate",
    "demand", "swear", "vow", "aim", "care", "dare", "resolve", "volunteer",
    "consent",
}
_ING_ONLY = {  # NFV-22
    "avoid", "admit", "consider", "deny", "finish", "suggest", "risk", "mind",
    "imagine", "postpone", "delay", "practise", "practice", "resist",
    "recommend", "appreciate", "dislike", "quit", "fancy", "involve",
    "contemplate", "entail",
}
_BOTH_SAME = {  # NFV-23 (NB: 'like' omitted - it also sits in the NFV-25 negative)
    "begin", "start", "continue", "love", "hate", "prefer", "bother", "cease",
    "commence",
}

# Fixed -ing frames (NFV-20): trigger lemmas near the -ing head.
_FIXED_GERUND_TRIG = {"use", "worth", "point", "difficulty", "trouble",
                      "busy", "time"}
_FIXED_GERUND_VERB = {"spend", "waste"}


# --------------------------------------------------------------------------- #
# Shared forms.
# --------------------------------------------------------------------------- #
_TO_INF = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": "TO"}},
]


# ---- NFV-06 too/enough + to-infinitive ------------------------------------ #
def _classify_too_enough(doc, token_ids):
    v = doc[token_ids[0]]
    h = v.head
    if any(c.lower_ in _DEGREE for c in h.children):
        return "NFV-06"
    return None


# ---- NFV-07 wh-word + to-infinitive --------------------------------------- #
def _classify_wh_inf(doc, token_ids):
    v = doc[token_ids[0]]
    if any(c.tag_ in _WH_TAGS for c in v.children):
        return "NFV-07"
    return None


# ---- NFV-03 control verb + to-infinitive (lexicon-gated) ------------------ #
_CONTROL_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp", "TAG": "VB"}},
    {"LEFT_ID": "x", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": "TO"}},
]


# NB: NFV-03 is left as a pure syntactic rule (verb + xcomp bare-infinitive +
# to) rather than lexicon-gated: en_core_web_sm lemmatises e.g. 'hopes' -> 'hop',
# so a verb-lemma gate is unreliable. The advcl/xcomp/acl contrast already
# separates it from purpose (NFV-01), adjective (NFV-04) and noun (NFV-05).


# ---- NFV-04 adjective + to-infinitive ------------------------------------- #
_ADJ_TO = [
    {"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": "ADJ"}},
    {"LEFT_ID": "a", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp", "TAG": "VB"}},
    {"LEFT_ID": "x", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": "TO"}},
]


def _adj_to_exclude(doc, token_ids):
    a = doc[token_ids[0]]
    return any(c.lower_ in _DEGREE for c in a.children)   # too/enough -> NFV-06


# ---- NFV-05 noun + to-infinitive ------------------------------------------ #
_NOUN_TO = [
    {"RIGHT_ID": "n", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "n", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "acl", "TAG": "VB"}},
    {"LEFT_ID": "x", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": "TO"}},
]


# ---- NFV-09 modal + bare infinitive --------------------------------------- #
_MODAL = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": "MD"}},
]


def _modal_exclude(doc, token_ids):
    v = doc[token_ids[0]]
    return any(c.lower_ in _PREF for c in v.children)     # 'd better/rather -> NFV-12


# ---- NFV-10 let / make (+ object) + bare infinitive ----------------------- #
_CAUSATIVE_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "c",
     "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}, "TAG": "VB"}},
]


def _causative_key(doc, token_ids):
    v = doc[token_ids[0]]
    if v.lemma_.lower() not in ("let", "make"):
        return None
    c = doc[token_ids[1]]
    if not any(gc.dep_ in ("nsubj", "nsubjpass") for gc in c.children):
        return None
    return v.lemma_.lower()


# ---- NFV-12 had better / would rather / why (not) ------------------------- #
_HAD_BETTER = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "d",
     "RIGHT_ATTRS": {"DEP": "advmod", "LOWER": {"IN": list(_PREF)}}},
]
_WHY_INF = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "w",
     "RIGHT_ATTRS": {"TAG": "WRB", "LOWER": "why"}},
]


def _classify_nfv12(doc, token_ids):
    v, c = doc[token_ids[0]], doc[token_ids[1]]
    if c.lower_ == "why":
        if any(ch.dep_ in ("nsubj", "nsubjpass") for ch in v.children):
            return None
        return "NFV-12"
    if c.lower_ == "better":
        return "NFV-12"
    if c.lower_ in ("rather", "sooner"):
        nxt = doc[c.i + 1] if c.i + 1 < len(doc) else None
        if nxt is not None and nxt.lower_ == "than":
            return None                                    # rather than -> NFV-13
        return "NFV-12"
    return None


# ---- NFV-13 rather than / sooner than / but / except + bare --------------- #
_RATHER_THAN = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"DEP": "mark", "LOWER": {"IN": ["than", "but", "except"]}}},
]


def _classify_nfv13(doc, token_ids):
    v, m = doc[token_ids[0]], doc[token_ids[1]]
    if m.lower_ == "than":
        if any(ch.lower_ in ("rather", "sooner") for ch in v.children):
            return "NFV-13"
        return None
    return "NFV-13"                                        # but / except + bare


# ---- NFV-14 help (+ to) + infinitive -------------------------------------- #
_HELP_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"LEMMA": "help", "POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp", "TAG": "VB"}},
]


# ---- NFV-16 gerund as object of verb -------------------------------------- #
_GER_OBJ = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "g",
     "RIGHT_ATTRS": {"DEP": "dobj", "TAG": "VBG"}},
]


# ---- NFV-17 gerund after preposition -------------------------------------- #
_GER_PREP = [
    {"RIGHT_ID": "p", "RIGHT_ATTRS": {"POS": "ADP"}},
    {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "g",
     "RIGHT_ATTRS": {"DEP": {"IN": ["pcomp", "pobj"]}, "TAG": "VBG"}},
]


# ---- NFV-26/27/30 perfect / passive / perfect-participle non-finites ------ #
# The perfect aux 'having' lemmatises to 'having'; the passive 'be'/'being'
# to 'be'. Match by TAG then discriminate in classify.
_PERF_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBN"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "h",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": {"IN": ["VB", "VBG"]}}},
]
_PASS_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBN"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "b",
     "RIGHT_ATTRS": {"DEP": "auxpass", "LEMMA": "be", "TAG": {"IN": ["VB", "VBG"]}}},
]

_FINITE_TAGS = {"VBZ", "VBP", "VBD", "MD"}


def _is_have_aux(tok):
    return (tok.lemma_ == "have" and tok.tag_ == "VB") or tok.lower_ == "having"


def _classify_perf_inf(doc, token_ids):     # NFV-26
    v, h = doc[token_ids[0]], doc[token_ids[1]]
    if not _is_have_aux(h):
        return None
    if v.dep_ == "advcl":
        return None                          # adverbial perfect participle -> NFV-30
    return "NFV-26"


def _classify_perf_part(doc, token_ids):    # NFV-30
    v, h = doc[token_ids[0]], doc[token_ids[1]]
    if h.lower_ != "having":
        return None
    if v.dep_ != "advcl":
        return None
    return "NFV-30"


def _classify_pass_inf(doc, token_ids):     # NFV-27
    v = doc[token_ids[0]]
    # a finite be-aux (is/was/being-under-finite) -> ordinary passive, not this
    if any(c.dep_ in ("aux", "auxpass") and c.tag_ in _FINITE_TAGS
           for c in v.children):
        return None
    return "NFV-27"


# --------------------------------------------------------------------------- #
# NFV-20: gerund in fixed expressions (custom, surface-frame heuristic).
# --------------------------------------------------------------------------- #
class _FixedGerundDetector(Detector):
    detector_type = "lexicon"
    construct_ids = ["NFV-20"]

    def __init__(self, version="nfv20-fixed@0.1"):
        self.version = version

    def match(self, doc, text_id="doc"):
        from ...schema import Annotation, Span
        out, seen = [], set()
        for g in doc:
            if g.tag_ != "VBG":
                continue
            # A fixed frame puts its trigger noun/adj/verb just before the -ing;
            # the (in) parenthetical and do-support scramble the parse, so scan
            # a short surface window rather than trusting the arcs.
            window = {doc[j].lemma_.lower() for j in range(max(0, g.i - 5), g.i)}
            hit = (window & _FIXED_GERUND_TRIG) or (window & _FIXED_GERUND_VERB)
            if not hit or g.i in seen:
                continue
            seen.add(g.i)
            out.append(Annotation(
                text_id=text_id, construct_id="NFV-20",
                span=Span(g.idx, g.idx + len(g.text), g.i, g.i),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": [g.i], "matched": g.text}))
        return out


# --------------------------------------------------------------------------- #
# LLM tier system prompts.
# --------------------------------------------------------------------------- #
_PURPOSE_SYS = (
    "The bracketed to-infinitive is an adverbial adjunct. Choose one:\n"
    "NFV-01 purpose ('I came to help' - in order to) | "
    "NFV-08 result/outcome ('He awoke to find the house empty', 'only to fail')."
)
_PART_SYS = (
    "The bracketed word heads a non-finite participle clause. Choose one:\n"
    "NFV-28 present participle (-ing, simultaneous/active: 'Smiling, she...') | "
    "NFV-29 past participle (-ed, passive: 'Asked to leave, he refused').\n"
    "A perfect participle clause ('Having finished, ...') is a different "
    "construct - return NONE for it."
)
_PERCEPTION_SYS = (
    "Return NFV-11 if a perception verb (see/hear/watch/feel/notice) takes a "
    "bare-infinitive complement expressing a completed action ('I saw her leave')."
)
_GER_SUBJ_SYS = (
    "Return NFV-15 if the bracketed -ing form is a gerund functioning as the "
    "grammatical subject ('Swimming is good for you')."
)
_GER_POSS_SYS = (
    "Return NFV-18 if the -ing gerund is preceded by a possessive/genitive "
    "subject ('I appreciate your helping / his leaving')."
)
_GER_BE_SYS = (
    "Return NFV-19 if the -ing gerund is the complement of 'be' "
    "('Seeing is believing')."
)
_MEANING_SYS = (
    "Return NFV-24 for verbs where infinitive vs -ing changes meaning "
    "(stop/remember/forget/try/regret/go on/mean/need + to vs -ing)."
)
_LIKE_SYS = (
    "Return NFV-25 for the like + infinitive (choose to) vs like + -ing (enjoy) "
    "nuance ('I like to check' vs 'I like checking'). Only for like/love/hate/"
    "prefer; other verbs taking to/-ing (begin, stop, remember) are different "
    "constructs - return NONE."
)
_WOULD_LIKE_SYS = (
    "Return NFV-31 for would like/love/hate/prefer + to-infinitive (specific "
    "occasion / polite) vs + -ing (general): 'I'd love to come tonight' vs "
    "'I love travelling'."
)
_MEANING_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"LEMMA": {"IN": ["stop", "remember",
        "forget", "try", "regret", "mean", "need", "go"]}}},
]
_WOULD_LIKE_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"LEMMA": {"IN": ["like", "love", "hate",
                                                       "prefer"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"TAG": "MD"}},
]
_PART_FORM = [{"RIGHT_ID": "p",
               "RIGHT_ATTRS": {"TAG": {"IN": ["VBG", "VBN"]}, "DEP": "advcl"}}]
_PURP_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VB", "DEP": "advcl"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"DEP": "aux", "TAG": "TO"}},
]
_PERCEPTION_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"LEMMA": {"IN": ["see", "hear", "watch",
        "feel", "notice", "observe"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "c",
     "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}}},
]
_GER_SUBJ_FORM = [{"RIGHT_ID": "g",
                   "RIGHT_ATTRS": {"TAG": "VBG", "DEP": {"IN": ["nsubj", "csubj"]}}}]
_GER_POSS_FORM = [
    {"RIGHT_ID": "g", "RIGHT_ATTRS": {"TAG": {"IN": ["VBG", "NN"]}}},
    {"LEFT_ID": "g", "REL_OP": ">", "RIGHT_ID": "p",
     "RIGHT_ATTRS": {"DEP": "poss", "TAG": "PRP$"}},
]
_GER_BE_FORM = [{"RIGHT_ID": "g", "RIGHT_ATTRS": {"TAG": "VBG",
                                                  "DEP": {"IN": ["attr", "acomp"]}}}]


def build(nlp, client=None):
    dets = []

    # ---- To-infinitive tier ------------------------------------------------
    dets.append(PhraseLexiconDetector(
        nlp, "NFV-02", ["in order not to", "so as not to",
                        "in order to not", "so as to not"],
        version="nfv02-negpurpose@0.1"))
    dets.append(DependencyRuleDetector(
        nlp, "NFV-03", _CONTROL_FORM, version="nfv03-control@0.1"))
    dets.append(DependencyRuleDetector(
        nlp, "NFV-04", _ADJ_TO, exclude=_adj_to_exclude, version="nfv04-adj@0.1"))
    dets.append(DependencyRuleDetector(
        nlp, "NFV-05", _NOUN_TO, version="nfv05-noun@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NFV-06"], _TO_INF, _classify_too_enough, version="nfv06-degree@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NFV-07"], _TO_INF, _classify_wh_inf, version="nfv07-wh@0.1"))

    # ---- Bare-infinitive tier ----------------------------------------------
    dets.append(DependencyRuleDetector(
        nlp, "NFV-09", _MODAL, exclude=_modal_exclude, version="nfv09-modal@0.1"))
    dets.append(LexiconRuleDetector(
        nlp, "NFV-10", _CAUSATIVE_FORM, {"let", "make"}, _causative_key,
        version="nfv10-causative@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NFV-12"], [_HAD_BETTER, _WHY_INF], _classify_nfv12,
        version="nfv12-hadbetter@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NFV-13"], _RATHER_THAN, _classify_nfv13,
        version="nfv13-ratherthan@0.1"))
    dets.append(DependencyRuleDetector(
        nlp, "NFV-14", _HELP_FORM, version="nfv14-help@0.1"))

    # ---- Gerund tier -------------------------------------------------------
    dets.append(DependencyRuleDetector(
        nlp, "NFV-16", _GER_OBJ, version="nfv16-gerobj@0.1"))
    dets.append(DependencyRuleDetector(
        nlp, "NFV-17", _GER_PREP, version="nfv17-gerprep@0.1"))
    dets.append(_FixedGerundDetector())

    # ---- Infinitive-vs-ing verb selection (fragment lexicons) --------------
    dets.append(PhraseLexiconDetector(
        nlp, "NFV-21", sorted(_INF_ONLY), attr="LEMMA", version="nfv21-inf@0.1"))
    dets.append(PhraseLexiconDetector(
        nlp, "NFV-22", sorted(_ING_ONLY), attr="LEMMA", version="nfv22-ing@0.1"))
    dets.append(PhraseLexiconDetector(
        nlp, "NFV-23", sorted(_BOTH_SAME), attr="LEMMA", version="nfv23-both@0.1"))

    # ---- Perfect / passive / participial non-finites -----------------------
    dets.append(RuleRoutingDetector(
        nlp, ["NFV-26"], _PERF_FORM, _classify_perf_inf,
        version="nfv26-perfinf@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NFV-27"], _PASS_FORM, _classify_pass_inf,
        version="nfv27-passinf@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["NFV-30"], _PERF_FORM, _classify_perf_part,
        version="nfv30-perfpart@0.1"))

    # ---- LLM tier (registered, skipped offline) ----------------------------
    dets.append(LLMReadingDetector(
        nlp, ["NFV-01", "NFV-08"], _PURP_FORM, _PURPOSE_SYS,
        client=client, version="nfv-purpose@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["NFV-28", "NFV-29"], _PART_FORM, _PART_SYS,
        client=client, version="nfv-participle@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["NFV-11"], _PERCEPTION_FORM, _PERCEPTION_SYS,
        client=client, version="nfv11-perception@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["NFV-15"], _GER_SUBJ_FORM, _GER_SUBJ_SYS,
        client=client, version="nfv15-gersubj@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["NFV-18"], _GER_POSS_FORM, _GER_POSS_SYS,
        client=client, version="nfv18-gerposs@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["NFV-19"], _GER_BE_FORM, _GER_BE_SYS,
        client=client, version="nfv19-gerbe@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["NFV-24"], _MEANING_FORM, _MEANING_SYS,
        client=client, version="nfv24-meaning@0.1"))
    dets.append(LLMStandaloneDetector(
        "NFV-25", _LIKE_SYS, client=client, version="nfv25-like@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["NFV-31"], _WOULD_LIKE_FORM, _WOULD_LIKE_SYS,
        client=client, version="nfv31-wouldlike@0.1"))

    return dets
