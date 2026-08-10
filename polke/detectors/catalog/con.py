"""Category CON - conditionals (CON-01..17).

Rule/lexicon tier (tested offline):
  CON-08 as/so long as, provided (that), providing, on condition that.
  CON-09 even if / only if / whether (... or not).
  CON-11 suppose / supposing / imagine / what if (clause-initial).
  CON-12 but for / if it weren't/hadn't been for / if not for / otherwise.
  CON-13 inverted conditional with no "if" (Had I known; Were she to ask;
         Should you need anything).
LLM tier (registered, skipped offline) - the type/reading of the conditional is
a tense-pairing/meaning judgement:
  CON-01..07 zero/first/second/third/mixed/unless; CON-10 in case; CON-14
  implied/incomplete; CON-15 imperative-as-condition; CON-16 if + would
  (politeness); CON-17 tentative (were to / should / happen to).

spaCy en_core_web_sm labels verified: if/unless/provided/as-long-as introduce an
``advcl`` via a ``mark``; the inverted conditional fronts had/were/should before
the subject with no ``mark``.
"""
from __future__ import annotations
from ...schema import Annotation, Span
from ..base import Detector
from .common import SentenceScoped
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector

_INVERT_HEADS = {"had", "were", "should"}


def _sent_initial(doc, start, end):
    return start == 0 or doc[start - 1].is_punct


class _InvertedConditional(SentenceScoped):
    """CON-13: Had/Were/Should + subject (no 'if')."""
    construct_ids = ["CON-13"]
    detector_type = "rule"
    version = "con13-inversion@0.1"

    def match_sent(self, doc, text_id="doc"):
        if len(doc) < 2:
            return []
        t0 = doc[0]
        if t0.lower_ not in _INVERT_HEADS:
            return []
        if any(t.dep_ == "mark" and t.lower_ == "if" for t in doc):
            return []
        # a real question ("Were you there?") is not a conditional
        last = next((t for t in reversed(doc) if not t.is_space), None)
        if last is not None and last.text == "?":
            return []
        subj = next((t for t in doc[1:4] if t.pos_ in ("PRON", "NOUN", "PROPN")),
                    None)
        if subj is None or subj.dep_ not in ("nsubj", "nsubjpass"):
            return []
        hi = len(doc) - 1
        while hi > 0 and doc[hi].is_punct:
            hi -= 1
        span = doc[0:hi + 1]
        return [Annotation(
            text_id=text_id, construct_id="CON-13",
            span=Span(span.start_char, span.end_char, 0, hi),
            detector_type=self.detector_type, detector_version=self.version,
            confidence=1.0, evidence={"tokens": list(range(0, hi + 1)),
                                      "matched": span.text})]


_CON08 = ["as long as", "so long as", "provided that", "provided", "providing",
          "providing that", "on condition that"]
_CON09 = ["even if", "only if", "whether or not", "whether"]
_CON11 = ["suppose", "supposing", "imagine", "what if"]
_CON12 = ["but for", "if it weren't for", "if it wasn't for",
          "if it hadn't been for", "if not for", "otherwise"]


# CON-11 gate: clause-initial only (suppose/imagine are common verbs elsewhere).
def _initial_gate(doc, start, end):
    return _sent_initial(doc, start, end)


# --------------------------------------------------------------------------- #
# LLM tier.
# --------------------------------------------------------------------------- #
_TYPE_SYS = (
    "The sentence is a conditional. Classify by tense pairing / meaning:\n"
    "CON-01 zero (if + present, present; general truth) | CON-02 first (if + "
    "present, will) | CON-03 second (if + past, would; unreal present, incl. "
    "'if I were you') | CON-04 third (if + past perfect, would have; unreal "
    "past) | CON-05 mixed past->present (if + past perfect, would now) | "
    "CON-06 mixed present->past (if + past/were, would have) | CON-07 unless "
    "(= if not).")
_CON10_SYS = ("Return CON-10 for 'in case' expressing a precaution (Take an "
              "umbrella in case it rains) - not a plain if-conditional.")
_CON14_SYS = ("Return CON-14 for an implied/incomplete conditional (If so; if "
              "not; in that case; then) standing in for a full if-clause.")
_CON15_SYS = ("Return CON-15 for an imperative/coordination read as a condition "
              "(Do that again and you're out).")
_CON16_SYS = ("Return CON-16 for if + would/could used for politeness or "
              "insistence (If you would wait here, please).")
_CON17_SYS = ("Return CON-17 for a tentative conditional with were to / should / "
              "happen to (If you should see her; If she were to ask).")

_IF_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "advcl"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "m",
     "RIGHT_ATTRS": {"DEP": "mark", "LEMMA": {"IN": ["if", "unless"]}}},
]
_INCASE_FORM = [{"RIGHT_ID": "c", "RIGHT_ATTRS": {"LOWER": "case"}}]
_ROOT_FORM = [{"RIGHT_ID": "r", "RIGHT_ATTRS": {"DEP": "ROOT"}}]
_IFANY_FORM = [{"RIGHT_ID": "m", "RIGHT_ATTRS": {"DEP": "mark", "LEMMA": "if"}}]


def build(nlp, client=None):
    return [
        PhraseLexiconDetector(nlp, "CON-08", _CON08, version="con08-longas@0.1"),
        PhraseLexiconDetector(nlp, "CON-09", _CON09, version="con09-evenif@0.1"),
        PhraseLexiconDetector(nlp, "CON-11", _CON11, gate=_initial_gate,
                              version="con11-suppose@0.1"),
        PhraseLexiconDetector(nlp, "CON-12", _CON12, version="con12-butfor@0.1"),
        _InvertedConditional(),
        # LLM tier
        LLMReadingDetector(nlp, ["CON-01", "CON-02", "CON-03", "CON-04",
                                 "CON-05", "CON-06", "CON-07"], _IF_FORM,
                           _TYPE_SYS, client=client, version="con-type@0.1"),
        LLMReadingDetector(nlp, ["CON-10"], _INCASE_FORM, _CON10_SYS,
                           client=client, version="con10-incase@0.1"),
        LLMReadingDetector(nlp, ["CON-14"], _ROOT_FORM, _CON14_SYS,
                           client=client, version="con14-implied@0.1"),
        LLMReadingDetector(nlp, ["CON-15"], _ROOT_FORM, _CON15_SYS,
                           client=client, version="con15-imp@0.1"),
        LLMReadingDetector(nlp, ["CON-16"], _IFANY_FORM, _CON16_SYS,
                           client=client, version="con16-would@0.1"),
        LLMReadingDetector(nlp, ["CON-17"], _IFANY_FORM, _CON17_SYS,
                           client=client, version="con17-tentative@0.1"),
    ]
