"""Three worked reference detectors, one per core tier.

The coding agent implements the remaining constructs in data/detectors.json by
following these patterns and making each construct's contract test cases pass.
"""
from __future__ import annotations
from .base import register
from .rules import DependencyRuleDetector
from .lexicon import LexiconDetector
from .llm import LLMReadingDetector
from ..registry import lexicons


# ---- Tier P: PAS-01 present/past simple passive -----------------------------
def _pas01(nlp):
    pattern = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "aux",
         "RIGHT_ATTRS": {"DEP": "auxpass"}},
    ]

    def exclude(doc, token_ids):
        for t in token_ids:
            tok = doc[t]
            if tok.dep_ == "auxpass" and tok.lemma_ == "get":
                return True  # get-passive -> route to PAS-15
        return False

    return DependencyRuleDetector(nlp, "PAS-01", pattern,
                                  exclude=exclude, version="pas01-rule@0.1")


# ---- Tier L: PREP-18 verb + dependent preposition ---------------------------
def _prep18(nlp):
    pattern = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep",
         "RIGHT_ATTRS": {"DEP": "prep"}},
    ]
    lex = {tuple(p) for p in lexicons()["verb_prep"]["pairs"]}

    def key_fn(doc, token_ids):
        verb = prep = None
        for t in token_ids:
            tok = doc[t]
            if tok.pos_ == "VERB":
                verb = tok
            elif tok.dep_ == "prep":
                prep = tok
        if verb is None or prep is None:
            return None
        return (verb.lemma_.lower(), prep.lower_)

    return LexiconDetector(nlp, "PREP-18", pattern, lex, key_fn,
                           version="prep18-lex@0.1")


# ---- Tier hybrid: present-perfect reading VTA-29..34 ------------------------
_PP_SYSTEM = (
    "You label the USE of an English present perfect (the verb group is marked "
    "with [[ ]]). Choose exactly one:\n"
    "VTA-29 experience | VTA-30 unfinished time-span | "
    "VTA-31 recent event with present result | VTA-32 resultative state change | "
    "VTA-33 superlative/first-time frame | VTA-34 news reporting.\n"
    'Return only JSON: {"construct_id": "...", "confidence": 0.0-1.0, '
    '"rationale": "..."}'
)


def _pp_reading(nlp, client=None):
    form = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "aux",
         "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "have",
                         "TAG": {"IN": ["VBP", "VBZ"]}}},
    ]
    ids = ["VTA-29", "VTA-30", "VTA-31", "VTA-32", "VTA-33", "VTA-34"]
    return LLMReadingDetector(nlp, ids, form, _PP_SYSTEM,
                              client=client, version="pp-reading@0.1")


def build_reference_detectors(nlp, llm_client=None):
    dets = [_pas01(nlp), _prep18(nlp), _pp_reading(nlp, client=llm_client)]
    for d in dets:
        register(d)
    return dets
