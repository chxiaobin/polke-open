"""Category REP - reported speech (REP-01..11).

Rule/lexicon tier (tested offline):
  REP-07 reported commands/requests: reporting verb + object + (not) to-inf.
  REP-08 say vs tell (the base-form report frames themselves).
  REP-09 illocutionary reporting verbs (offer/promise/suggest/admit/deny/accuse/
         congratulate/warn/insist ...).
LLM tier (registered, skipped offline) - backshift/deixis/reading are meaning:
  REP-01 reported statements + backshift; REP-02 no backshift; REP-03 pronoun/
  possessive shift; REP-04 deixis shift; REP-05 reported yes/no (if/whether);
  REP-06 reported wh-questions; REP-10 modal backshift; REP-11 present-tense
  reporting (news/summaries).

REP-08/09 positives are metalinguistic frame lists (base forms), so a surface/
lemma phrase matcher is the right tool; REP-08 is a degenerate row (see NOTES).
spaCy en_core_web_sm labels verified: to-infinitive complements are ``xcomp``
with a ``TO`` ``aux``; the reported object is ``dobj``/``dative``.
"""
from __future__ import annotations
from ..lexical import PhraseLexiconDetector
from ..routing import LexiconRuleDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector

# Verbs that take Object + to-infinitive as a reported directive.
_DIRECTIVE_VERBS = {
    "tell", "ask", "order", "command", "advise", "warn", "beg", "urge",
    "remind", "instruct", "invite", "encourage", "forbid", "persuade",
    "request", "implore", "recommend", "expect", "want",
}

_REP07_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "o",
     "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "dative"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp"}},
]


def _rep07_key(doc, token_ids):
    v = doc[token_ids[0]]
    x = doc[token_ids[2]]
    if not any(g.tag_ == "TO" for g in x.children):
        return None
    return v.lemma_.lower()


# REP-09 illocutionary verbs (excludes say/tell/ask/agree -> other rows).
_ILLOC_VERBS = ["offer", "promise", "threaten", "suggest", "admit", "deny",
                "accuse", "congratulate", "warn", "insist", "propose",
                "recommend", "apologize", "apologise", "complain", "boast",
                "refuse", "demand", "confess", "announce", "claim", "beg",
                "urge", "remind", "invite", "encourage", "advise"]

# --------------------------------------------------------------------------- #
# LLM tier.
# --------------------------------------------------------------------------- #
_STMT_SYS = (
    "A reported statement (reporting verb + that-clause). Choose:\n"
    "REP-01 backshift of tense (She said she was tired) | REP-03 pronoun/"
    "possessive shift (I->he, my->his) | REP-10 modal backshift (will->would, "
    "can->could, must->had to, may->might).\n"
    "Reported questions (asked whether/if ...) are a different construct - "
    "return NONE for those.")
_QUES_SYS = (
    "A reported question (statement word order, no inversion). Choose:\n"
    "REP-05 reported yes/no question with if/whether (He asked whether I "
    "agreed) | REP-06 reported wh-question (She asked where I lived).")
_REP02_SYS = ("Return REP-02 when a reported statement keeps the original tense "
              "(no backshift) because it is still true / just said / a general "
              "truth (He said the Earth is round).")
_REP04_SYS = ("Return REP-04 for deictic shift in reported speech (this->that, "
              "here->there, now->then, today->that day, tomorrow->the next day, "
              "ago->before).")
_REP11_SYS = ("Return REP-11 for reporting with a present-tense reporting verb, "
              "typical of news/summaries (The report says that ...).")

_REPORT_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "c",
     "RIGHT_ATTRS": {"DEP": "ccomp"}},
]
_ROOT_FORM = [{"RIGHT_ID": "r", "RIGHT_ATTRS": {"DEP": "ROOT"}}]


def build(nlp, client=None):
    return [
        LexiconRuleDetector(nlp, "REP-07", _REP07_FORM, _DIRECTIVE_VERBS,
                            _rep07_key, version="rep07-command@0.1"),
        PhraseLexiconDetector(nlp, "REP-08", ["say", "tell"], attr="LOWER",
                              version="rep08-saytell@0.1"),
        PhraseLexiconDetector(nlp, "REP-09", _ILLOC_VERBS, attr="LEMMA",
                              version="rep09-illoc@0.1"),
        # LLM tier
        LLMReadingDetector(nlp, ["REP-01", "REP-03", "REP-10"], _REPORT_FORM,
                           _STMT_SYS, client=client, version="rep-stmt@0.1"),
        LLMReadingDetector(nlp, ["REP-05", "REP-06"], _REPORT_FORM, _QUES_SYS,
                           client=client, version="rep-ques@0.1"),
        LLMStandaloneDetector("REP-02", _REP02_SYS, client=client,
                              version="rep02-nobackshift@0.1"),
        LLMStandaloneDetector("REP-04", _REP04_SYS, client=client,
                              version="rep04-deixis@0.1"),
        LLMStandaloneDetector("REP-11", _REP11_SYS, client=client,
                              version="rep11-present@0.1"),
    ]
