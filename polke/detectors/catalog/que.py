"""Category QUE - questions (QUE-01..17).

Rule/lexicon tier:
  QUE-01/02  yes-no questions, split by operator: be/have/modal vs do-support.
  QUE-03     short answers (Yes/No + subject + operator, no '?').
  QUE-04..08 wh-questions, one router keyed on the wh word's grammatical role:
             object (04), adjunct where/when/why (05), how+adj/adv/quantity (06),
             subject / no inversion (07), which/what/whose + noun (08).
  QUE-09     preposition placement (wh as object of a preposition; stranded/pied).
  QUE-15     indirect/embedded questions (if/whether, or embedded wh, no inversion).
  QUE-16     how come / what if / what about / how about (fixed openers).
LLM tier (registered, skipped offline):
  QUE-10 copular/'what...like'; QUE-11/12/13 the tag-question family;
  QUE-14 negative questions; QUE-17 alternative/echo/rhetorical.

spaCy en_core_web_sm labels verified empirically: wh words are WP/WDT/WP$/WRB;
do-support and be/have/modal operators are ``aux``/``auxpass``; a fronted subject
wh has ``nsubj`` and no inversion; embedded questions sit under ``ccomp`` with an
``if``/``whether`` ``mark`` or an in-situ wh word.
"""
from __future__ import annotations
from ..rules import DependencyRuleDetector
from ..routing import RuleRoutingDetector, LexiconRuleDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector

_WH_TAGS = {"WP", "WDT", "WP$", "WRB"}
_YESNO_WORDS = {"yes", "no", "yeah", "yep", "nope", "nah", "sure"}
_HOW_QUANTITY = {"much", "many", "long", "often", "far", "old", "tall",
                 "big", "deep", "wide", "high", "fast", "soon", "late"}
_OBJECT_DEPS = {"dobj", "dative", "attr", "oprd", "nsubjpass"}
_PREP_WH = {"who", "whom", "what", "which", "whose"}
# QUE-16 fixed question openers: (wh word, following head word).
_OPENERS = {("what", "about"), ("how", "about"),
            ("how", "come"), ("what", "if")}


def _is_question(doc):
    for t in reversed(doc):
        if t.is_punct:
            if t.text == "?":
                return True
            continue
        break
    return False


def _has_wh(doc):
    return any(t.tag_ in _WH_TAGS for t in doc)


# --------------------------------------------------------------------------- #
# QUE-01/02: yes-no questions (be/have/modal vs do-support).
# --------------------------------------------------------------------------- #
_YESNO_FORM = [
    {"RIGHT_ID": "h", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "h", "REL_OP": ">", "RIGHT_ID": "s",
     "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
]


def _classify_yesno(doc, token_ids):
    if not _is_question(doc) or _has_wh(doc):
        return None
    head, subj = doc[token_ids[0]], doc[token_ids[1]]
    auxes = [c for c in head.children if c.dep_ in ("aux", "auxpass")]
    if auxes:
        op = min(auxes, key=lambda t: t.i)
    elif head.lemma_ == "be":
        op = head
    else:
        return None
    if op.i >= subj.i:          # need subject-operator inversion
        return None
    return "QUE-02" if op.lemma_ == "do" else "QUE-01"


# --------------------------------------------------------------------------- #
# QUE-03: short answers ("Yes, I do." / "No, she hasn't.").
# --------------------------------------------------------------------------- #
_SHORT_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "ROOT"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "i",
     "RIGHT_ATTRS": {"DEP": "intj"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "s",
     "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
]


def _classify_short(doc, token_ids):
    v, i = doc[token_ids[0]], doc[token_ids[1]]
    if i.lower_ not in _YESNO_WORDS:
        return None
    if _is_question(doc):
        return None
    if v.tag_ != "MD" and v.lemma_ not in ("do", "be", "have"):
        return None
    return "QUE-03"


# --------------------------------------------------------------------------- #
# QUE-04..08: wh-questions routed by the wh word's role.
# --------------------------------------------------------------------------- #
_WH_FORM = [{"RIGHT_ID": "w", "RIGHT_ATTRS": {"TAG": {"IN": list(_WH_TAGS)}}}]


def _classify_wh(doc, token_ids):
    w = doc[token_ids[0]]
    if not _is_question(doc):
        return None
    if w.dep_ == "pobj":
        return None                                   # -> QUE-09
    if w.dep_ in ("det", "poss"):
        return "QUE-08"                               # which/what/whose + noun
    if w.lower_ == "how":
        h = w.head
        if h.tag_ in ("JJ", "JJR", "JJS", "RB", "RBR", "RBS"):
            return "QUE-06"
        nxt = doc[w.i + 1] if w.i + 1 < len(doc) else None
        if nxt is not None and nxt.lower_ in _HOW_QUANTITY:
            return "QUE-06"
        return "QUE-05"                               # how + verb (manner)
    if w.dep_ in ("nsubj", "nsubjpass"):
        return "QUE-07"                               # subject wh, no inversion
    if w.tag_ == "WRB":
        return "QUE-05"                               # where/when/why adjunct
    if w.dep_ in _OBJECT_DEPS:
        return "QUE-04"                               # object/complement wh
    return None


# --------------------------------------------------------------------------- #
# QUE-09: preposition placement (wh as pobj; stranded or pied-piped).
# --------------------------------------------------------------------------- #
_PREP_FORM = [
    {"RIGHT_ID": "p", "RIGHT_ATTRS": {"TAG": "IN"}},
    {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "w",
     "RIGHT_ATTRS": {"TAG": {"IN": ["WP", "WDT"]}}},
]


def _classify_prep(doc, token_ids):
    w = doc[token_ids[1]]
    if not _is_question(doc) or w.lower_ not in _PREP_WH:
        return None
    return "QUE-09"


# --------------------------------------------------------------------------- #
# QUE-15: indirect / embedded questions.
# --------------------------------------------------------------------------- #
_EMBED_DEPS = ["ccomp", "advcl", "acl", "pcomp", "xcomp", "relcl"]
_EMBED_IF = [
    {"RIGHT_ID": "emb", "RIGHT_ATTRS": {"DEP": {"IN": _EMBED_DEPS}}},
    {"LEFT_ID": "emb", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"DEP": "mark", "LEMMA": {"IN": ["if", "whether"]}}},
]
_EMBED_WH = [
    {"RIGHT_ID": "emb", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "advcl",
                                                       "acl", "pcomp"]}}},
    {"LEFT_ID": "emb", "REL_OP": ">", "RIGHT_ID": "w",
     "RIGHT_ATTRS": {"TAG": {"IN": list(_WH_TAGS)}}},
]


def _classify_embed(doc, token_ids):
    return "QUE-15"


# --------------------------------------------------------------------------- #
# QUE-16: how come / what if / what about / how about.
# --------------------------------------------------------------------------- #
_OPENER_FORM = [
    {"RIGHT_ID": "h", "RIGHT_ATTRS": {"LOWER": {"IN": ["about", "come", "if"]}}},
    {"LEFT_ID": "h", "REL_OP": ">", "RIGHT_ID": "w",
     "RIGHT_ATTRS": {"TAG": {"IN": ["WP", "WDT", "WRB"]}}},
]


def _opener_key(doc, token_ids):
    h, w = doc[token_ids[0]], doc[token_ids[1]]
    return (w.lower_, h.lower_)


# --------------------------------------------------------------------------- #
# LLM tier - registered, skipped offline.
# --------------------------------------------------------------------------- #
_TAG_SYS = (
    "The sentence ends in a tag question (a short operator+pronoun after a "
    "comma). Choose one:\n"
    "QUE-11 ordinary reversed-polarity tag (positive clause -> negative tag or "
    "vice versa) | QUE-12 same-polarity or special tag (e.g. 'are you?', "
    "'aren't I', imperative 'will you', 'shall we') | QUE-13 tag after a "
    "negative/indefinite word (nothing/nobody/everyone -> polarity flips)."
)
_NEGQ_SYS = (
    "The bracketed form is a question. Return QUE-14 only if it is a negative "
    "question (contract negation on the operator, e.g. \"Don't you agree?\", "
    "\"Why didn't you call?\")."
)
_WHATLIKE_SYS = (
    "Return QUE-10 if this is a copular wh pattern such as 'What is X like?', "
    "'How are you?', 'What is it like?' (asking for a description/state)."
)
_ALT_ECHO_SYS = (
    "Return QUE-17 if this is an alternative question ('Tea or coffee?'), an "
    "echo question ('You did what?'), or a rhetorical question ('Who cares?')."
)

_TAG_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "ROOT",
                                      "POS": {"IN": ["AUX", "VERB"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "c",
     "RIGHT_ATTRS": {"DEP": "ccomp"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "s",
     "RIGHT_ATTRS": {"DEP": "nsubj", "TAG": "PRP"}},
]
_NEGQ_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "n",
     "RIGHT_ATTRS": {"DEP": "neg"}},
]
_WHATLIKE_FORM = [
    {"RIGHT_ID": "w", "RIGHT_ATTRS": {"TAG": {"IN": ["WP", "WRB"]}}},
]


def build(nlp, client=None):
    dets = []

    dets.append(RuleRoutingDetector(
        nlp, ["QUE-01", "QUE-02"], _YESNO_FORM, _classify_yesno,
        version="que-yesno@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["QUE-03"], _SHORT_FORM, _classify_short,
        version="que03-short@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["QUE-04", "QUE-05", "QUE-06", "QUE-07", "QUE-08"],
        _WH_FORM, _classify_wh, version="que-wh@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["QUE-09"], _PREP_FORM, _classify_prep,
        version="que09-prep@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["QUE-15"], [_EMBED_IF, _EMBED_WH], _classify_embed,
        version="que15-embed@0.1"))
    dets.append(LexiconRuleDetector(
        nlp, "QUE-16", _OPENER_FORM, _OPENERS, _opener_key,
        version="que16-opener@0.1"))

    # LLM tier (skipped offline; registered).
    dets.append(LLMReadingDetector(
        nlp, ["QUE-11", "QUE-12", "QUE-13"], _TAG_FORM, _TAG_SYS,
        client=client, version="que-tags@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["QUE-14"], _NEGQ_FORM, _NEGQ_SYS,
        client=client, version="que14-negq@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["QUE-10"], _WHATLIKE_FORM, _WHATLIKE_SYS,
        client=client, version="que10-whatlike@0.1"))
    dets.append(LLMStandaloneDetector(
        "QUE-17", _ALT_ECHO_SYS, client=client, version="que17-alt-echo@0.1"))

    return dets
