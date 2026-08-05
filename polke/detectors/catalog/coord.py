"""Category COORD - coordination (COORD-01..09).

Rule/lexicon tier (tested offline):
  COORD-02 but / yet (contrast).
  COORD-03 or / nor (alternative/exclusion), excluding correlative either/whether.
  COORD-05 both ... and (the correlative preconjunction "both").
  COORD-07 not only ... (but also) - lexicon.
  COORD-08 whether ... (or not) - lexicon.
LLM tier (registered, skipped offline):
  COORD-01 and (addition/sequence/result/condition reading);
  COORD-04 so (result) / for (reason); COORD-06 either ... or;
  COORD-09 ellipsis/gapping in coordination.

The three conjunction ids share one form (a ``cc``/``preconj`` token) and are
routed by the conjunction word; "or" is gated so correlative either.../whether...
go to their own (LLM/lexicon) ids.
"""
from __future__ import annotations
from ..lexical import PhraseLexiconDetector
from ..routing import RuleRoutingDetector
from ..llm import LLMReadingDetector

_CC_FORM = [{"RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": {"IN": ["cc", "preconj"]}}}]


def _classify_coord(doc, token_ids):
    c = doc[token_ids[0]]
    w = c.lower_
    if w == "both" and c.dep_ == "preconj":
        return "COORD-05"
    if w in ("but", "yet") and c.dep_ == "cc":
        return "COORD-02"
    if w in ("or", "nor") and c.dep_ == "cc":
        # correlative either.../whether... -> COORD-06 / COORD-08
        if any(t.lower_ in ("either", "whether") for t in doc[:c.i]):
            return None
        return "COORD-03"
    return None


_COORD07 = ["not only"]
_COORD08 = ["whether"]

# --------------------------------------------------------------------------- #
# LLM tier.
# --------------------------------------------------------------------------- #
_COORD01_SYS = ("Return COORD-01 for 'and' coordination and specify nothing "
                "further; distinguish its reading (addition/sequence/result/"
                "condition, e.g. 'Touch it and you'll regret it').")
_COORD04_SYS = ("Return COORD-04 for clause coordination with 'so' (result) or "
                "'for' (reason, formal).")
_COORD06_SYS = ("Return COORD-06 for the correlative 'either ... or'.")
_COORD09_SYS = ("Return COORD-09 for ellipsis/gapping inside coordination "
                "(I had tea and she [had] coffee).")

_AND_FORM = [{"RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": "cc", "LOWER": "and"}}]
_SOFOR_FORM = [{"RIGHT_ID": "c",
                "RIGHT_ATTRS": {"LOWER": {"IN": ["so", "for"]}}}]
_EITHER_FORM = [{"RIGHT_ID": "c",
                 "RIGHT_ATTRS": {"LOWER": "either", "DEP": "preconj"}}]
_CONJ_FORM = [{"RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": "conj"}}]


def build(nlp, client=None):
    return [
        RuleRoutingDetector(nlp, ["COORD-02", "COORD-03", "COORD-05"],
                            _CC_FORM, _classify_coord, span="match",
                            version="coord-conj@0.1"),
        PhraseLexiconDetector(nlp, "COORD-07", _COORD07,
                              version="coord07-notonly@0.1"),
        PhraseLexiconDetector(nlp, "COORD-08", _COORD08,
                              version="coord08-whether@0.1"),
        LLMReadingDetector(nlp, ["COORD-01"], _AND_FORM, _COORD01_SYS,
                           client=client, version="coord01-and@0.1"),
        LLMReadingDetector(nlp, ["COORD-04"], _SOFOR_FORM, _COORD04_SYS,
                           client=client, version="coord04-sofor@0.1"),
        LLMReadingDetector(nlp, ["COORD-06"], _EITHER_FORM, _COORD06_SYS,
                           client=client, version="coord06-either@0.1"),
        LLMReadingDetector(nlp, ["COORD-09"], _CONJ_FORM, _COORD09_SYS,
                           client=client, version="coord09-gapping@0.1"),
    ]
