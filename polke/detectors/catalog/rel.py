"""Category REL - relative clauses (REL-01..18).

Finite relatives split by the relativiser word:
  who/whom/which/that (REL-01/02/03/06) share one form -> a RuleRoutingDetector
  routes a wh word that is a direct child of a ``relcl`` verb by its lemma.
  whose (REL-05) and the zero relative (REL-04) get their own shapes.
Preposition handling: stranded (REL-11) vs pied-piped (REL-12) vs relative
adverbs where/when/why (REL-13) vs the formal connective whereby (REL-14).
Reading/meaning siblings that share a surface form stay in the LLM tier:
  that-preference (REL-07), the non-defining/sentential/quantifier group
  (REL-08/09/10), and the reduced-relative group (REL-15/16/17/18).

spaCy en_core_web_sm labels verified empirically: relative pronouns are
WDT/WP/WP$, relative adverbs WRB; the relative clause is ``relcl`` (a
sentential ``which`` is ``advcl``); non-defining clauses are comma-set.
"""
from __future__ import annotations
from ..rules import DependencyRuleDetector
from ..routing import RuleRoutingDetector, LexiconRuleDetector
from ..llm import LLMReadingDetector

# Prepositions that can strand at the end of a relative clause.
_PREP_WORDS = {
    "in", "on", "at", "for", "with", "to", "about", "from", "of", "by",
    "into", "onto", "over", "under", "through", "after", "before", "against",
    "between", "among", "towards", "toward", "upon", "within", "without",
    "around", "off", "out", "up", "down", "across", "behind", "beyond",
}
_WH_TAGS = {"WDT", "WP", "WP$", "WRB"}
_REL_ADVERBS = {"where", "when", "why"}
_WHEREBY_WORDS = {"whereby", "wherein", "whereupon", "whereof", "whereat"}

# Route table for the finite relativiser router (REL-01/02/03/06).
_FINITE_REL = {"who": "REL-01", "which": "REL-02",
               "that": "REL-03", "whom": "REL-06"}


# --------------------------------------------------------------------------- #
# REL-01/02/03/06: finite relativiser (wh directly under a relcl verb).
# --------------------------------------------------------------------------- #
_FINITE_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "relcl"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "w",
     "RIGHT_ATTRS": {"TAG": {"IN": ["WP", "WDT"]}}},
]


def _classify_finite(doc, token_ids):
    w = doc[token_ids[1]]
    return _FINITE_REL.get(w.lower_)


# --------------------------------------------------------------------------- #
# REL-04: zero (object) relative - no overt relativiser.
# --------------------------------------------------------------------------- #
_ZERO_RELCL = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "relcl"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "subj",
     "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
]
# Fronted-object fragment ("the film (Ø) I saw"): the parser leaves the
# antecedent as the object of the verb, preceding the subject.
_ZERO_FRONTED = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "obj",
     "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "nsubjpass"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "subj",
     "RIGHT_ATTRS": {"DEP": "nsubj"}},
]


def _classify_zero(doc, token_ids):
    v = doc[token_ids[0]]
    # No overt relativiser anywhere in the clause.
    for t in v.subtree:
        if t.tag_ in _WH_TAGS:
            return None
    subj = [c for c in v.children if c.dep_ in ("nsubj", "nsubjpass")]
    if not subj or subj[0].tag_ in _WH_TAGS:
        return None
    if v.dep_ == "relcl":
        return "REL-04"
    objs = [c for c in v.children if c.dep_ in ("dobj", "nsubjpass")]
    if objs and objs[0].i < subj[0].i:      # object fronted before subject
        return "REL-04"
    return None


# --------------------------------------------------------------------------- #
# REL-05: whose (possessive relative).
# --------------------------------------------------------------------------- #
_WHOSE_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "relcl"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "n",
     "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "n", "REL_OP": ">", "RIGHT_ID": "w",
     "RIGHT_ATTRS": {"TAG": "WP$"}},
]


# --------------------------------------------------------------------------- #
# REL-11: stranded preposition in a relative clause.
# --------------------------------------------------------------------------- #
_STRAND_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "p",
     "RIGHT_ATTRS": {"DEP": {"IN": ["advmod", "prt", "prep"]}}},
]


def _classify_strand(doc, token_ids):
    v, p = doc[token_ids[0]], doc[token_ids[1]]
    if p.lower_ not in _PREP_WORDS or p.i <= v.i:
        return None
    if any(c.dep_ in ("pobj", "pcomp") for c in p.children):
        return None                       # has an object -> not stranded
    return "REL-11"


# --------------------------------------------------------------------------- #
# REL-12: pied-piped preposition ("... in which ...").
# --------------------------------------------------------------------------- #
_PIED_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "relcl"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "p",
     "RIGHT_ATTRS": {"DEP": "prep", "TAG": "IN"}},
    {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "w",
     "RIGHT_ATTRS": {"TAG": {"IN": ["WDT", "WP"]}}},
]


# --------------------------------------------------------------------------- #
# REL-13: relative adverbs where/when/why (needs a nominal antecedent).
# --------------------------------------------------------------------------- #
_WRB_FORM = [{"RIGHT_ID": "w", "RIGHT_ATTRS": {"TAG": "WRB"}}]


def _classify_reladv(doc, token_ids):
    w = doc[token_ids[0]]
    if w.lower_ not in _REL_ADVERBS:
        return None
    # An antecedent noun must precede it (distinguishes it from a fronted
    # interrogative where/when/why).
    if not any(t.pos_ in ("NOUN", "PROPN") and t.i < w.i for t in doc):
        return None
    return "REL-13"


# --------------------------------------------------------------------------- #
# REL-14: whereby / wherein ... (formal connective) - lexicon gated.
# --------------------------------------------------------------------------- #
def _whereby_key(doc, token_ids):
    return doc[token_ids[0]].lower_


# --------------------------------------------------------------------------- #
# LLM tier - registered, skipped offline.
# --------------------------------------------------------------------------- #
_REL07_SYS = (
    "The bracketed relative clause uses 'that'. Decide whether this is a "
    "'that-preference' context (antecedent is a superlative, an ordinal, "
    "all/only/every/any, or an indefinite/quantified head such as "
    "everything/something/the only). Return REL-07 if so."
)
_ND_SYS = (
    "The bracketed relative is a non-defining (comma-set) clause. Choose one:\n"
    "REL-08 ordinary non-defining relative (who/which adds extra info about a "
    "noun) | REL-09 sentential relative (which refers to the whole preceding "
    "clause) | REL-10 quantifier + of + which/whom (e.g. 'many of whom')."
)
_RED_SYS = (
    "The bracketed phrase is a reduced (non-finite) relative postmodifying a "
    "noun. Choose one:\n"
    "REL-15 -ing participle (active) | REL-16 -ed participle (passive) | "
    "REL-17 to-infinitive relative | REL-18 adjective/prepositional reduced "
    "relative (e.g. 'the people responsible', 'the woman in the corner')."
)

_REL07_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "relcl"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "w",
     "RIGHT_ATTRS": {"TAG": {"IN": ["WDT", "WP"]}, "LOWER": "that"}},
]
_ND_FORM = [{"RIGHT_ID": "w", "RIGHT_ATTRS": {"TAG": {"IN": ["WP", "WDT"]}}}]
# Post-nominal modifiers only (>++ = child to the RIGHT of the noun): acl/
# relcl participial and infinitive relatives, plus postposed adjectives (the
# people responsible), which parse as amod but only ever follow the noun in
# this reduced-relative use. Plain prenominal amod stays excluded.
_RED_FORM = [
    {"RIGHT_ID": "n", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "n", "REL_OP": ">++", "RIGHT_ID": "a",
     "RIGHT_ATTRS": {"DEP": {"IN": ["acl", "relcl", "amod"]}}},
]


def build(nlp, client=None):
    dets = []

    # REL-01/02/03/06: who / which / that / whom.
    dets.append(RuleRoutingDetector(
        nlp, ["REL-01", "REL-02", "REL-03", "REL-06"],
        _FINITE_FORM, _classify_finite, version="rel-finite@0.1"))

    # REL-04: zero (object) relative.
    dets.append(RuleRoutingDetector(
        nlp, ["REL-04"], [_ZERO_RELCL, _ZERO_FRONTED], _classify_zero,
        version="rel04-zero@0.1"))

    # REL-05: whose.
    dets.append(RuleRoutingDetector(
        nlp, ["REL-05"], _WHOSE_FORM,
        lambda d, t: "REL-05", version="rel05-whose@0.1"))

    # REL-11: stranded preposition.
    dets.append(RuleRoutingDetector(
        nlp, ["REL-11"], _STRAND_FORM, _classify_strand,
        version="rel11-strand@0.1"))

    # REL-12: pied-piped preposition.
    dets.append(DependencyRuleDetector(
        nlp, "REL-12", _PIED_FORM, version="rel12-pied@0.1"))

    # REL-13: relative adverbs where/when/why.
    dets.append(RuleRoutingDetector(
        nlp, ["REL-13"], _WRB_FORM, _classify_reladv,
        version="rel13-adv@0.1"))

    # REL-14: whereby (formal connective).
    dets.append(LexiconRuleDetector(
        nlp, "REL-14", _WRB_FORM, _WHEREBY_WORDS, _whereby_key,
        version="rel14-whereby@0.1"))

    # LLM tier (skipped offline; registered).
    dets.append(LLMReadingDetector(
        nlp, ["REL-07"], _REL07_FORM, _REL07_SYS,
        client=client, version="rel07-that-pref@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["REL-08", "REL-09", "REL-10"], _ND_FORM, _ND_SYS,
        client=client, version="rel-nondef@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["REL-15", "REL-16", "REL-17", "REL-18"], _RED_FORM, _RED_SYS,
        client=client, version="rel-reduced@0.1"))

    return dets
