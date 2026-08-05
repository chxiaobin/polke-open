"""Category FUT - future reference (FUT-01..21).

Future FORMS are aux-chain routable; future READINGS (prediction vs intention vs
arrangement vs offer/promise) are LLM.

Rule/lexicon tier (tested offline):
  FUT-14/16/17  future continuous / future perfect / future perfect continuous
                (one 'will+...' aux-chain router)
  FUT-18        be to (formal arrangement): be + to-infinitive, no acomp
  FUT-19        be about to (imminence): 'about' acomp of be + to-infinitive
  FUT-20        be due/set to, be on the point/verge of
LLM tier (registered, skipped offline):
  FUT-01..07  the 'will' readings (prediction/decision/offer/promise/refusal/
              request/habitual) share the modal-will form
  FUT-08..10  'be going to' readings (intention/evidence/unfulfilled)
  FUT-11/12/13 present forms for the future
  FUT-15      future continuous as-a-matter-of-course / polite (question form)
  FUT-21      future in the past (would / was to / was about to)
"""
from __future__ import annotations
from ..routing import RuleRoutingDetector, CompositeDetector
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector


# --------------------------------------------------------------------------- #
# FORM patterns.
# --------------------------------------------------------------------------- #
_MODAL_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "md",
     "RIGHT_ATTRS": {"TAG": "MD"}},
]
_WILL_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "md",
     "RIGHT_ATTRS": {"TAG": "MD", "LEMMA": {"IN": ["will", "shall"]}}},
]
_GOING_TO = [
    {"RIGHT_ID": "go", "RIGHT_ATTRS": {"LEMMA": "go", "TAG": "VBG"}},
    {"LEFT_ID": "go", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp"}},
]
_BE_TO = [
    {"RIGHT_ID": "be", "RIGHT_ATTRS": {"LEMMA": "be",
                                       "TAG": {"IN": ["VBZ", "VBP", "VBD"]}}},
    {"LEFT_ID": "be", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp", "TAG": "VB"}},
    {"LEFT_ID": "x", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"TAG": "TO"}},
]
_BE_ABOUT_TO = [
    {"RIGHT_ID": "ab", "RIGHT_ATTRS": {"LOWER": "about", "DEP": "acomp"}},
    {"LEFT_ID": "ab", "REL_OP": ">", "RIGHT_ID": "x",
     "RIGHT_ATTRS": {"DEP": "xcomp"}},
]
_BE_DUE_TO = [
    {"RIGHT_ID": "adj", "RIGHT_ATTRS": {"LOWER": {"IN": ["due", "set"]},
                                        "DEP": "acomp"}},
    {"LEFT_ID": "adj", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"LOWER": "to"}},
]

# Present-continuous future (arrangement) form for FUT-11.
_PRES_CONT = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBG"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "be",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "be", "TAG": {"IN": ["VBZ", "VBP"]}}},
]
_PRES_SIMPLE = [{"RIGHT_ID": "v",
                 "RIGHT_ATTRS": {"TAG": {"IN": ["VBZ", "VBP"]}, "DEP": "ROOT"}}]


# --------------------------------------------------------------------------- #
def _is_question(doc):
    return any(t.text == "?" for t in doc)


def classify_will_forms(doc, tids):
    """Route the will/shall aux chain: future cont / perfect / perfect cont."""
    v = doc[tids[0]]
    auxes = [c for c in v.children if c.dep_ in ("aux", "auxpass")]
    if not any(a.tag_ == "MD" and a.lemma_ in ("will", "shall") for a in auxes):
        return None
    have = [a for a in auxes if a.lemma_ == "have"]
    been = [a for a in auxes if a.lemma_ == "be" and a.tag_ == "VBN"]
    be_vb = [a for a in auxes if a.lemma_ == "be" and a.tag_ == "VB"]
    if v.tag_ == "VBG":
        if have and been:
            return "FUT-17"                  # will have been V-ing
        if be_vb and not have:
            if _is_question(doc):
                return None                  # -> FUT-15 (LLM)
            return "FUT-14"                  # will be V-ing
        return None
    if v.tag_ == "VBN" and have:
        return "FUT-16"                      # will have V-en
    return None


def _be_to_ok(doc, tids):
    be = doc[tids[0]]
    if any(c.dep_ == "acomp" for c in be.children):
        return None                          # about/due/going -> 19/20
    return "FUT-18"


def _be_about_ok(doc, tids):
    ab = doc[tids[0]]
    return "FUT-19" if ab.head.lemma_ == "be" else None


def _be_due_ok(doc, tids):
    adj = doc[tids[0]]
    return "FUT-20" if adj.head.lemma_ == "be" else None


# --------------------------------------------------------------------------- #
# LLM reading prompts.
# --------------------------------------------------------------------------- #
_WILL_SYS = (
    "The bracketed 'will/'ll/won't' verb group has one of these USES. Choose "
    "one:\nFUT-01 prediction/future fact (It'll rain tomorrow) | FUT-02 "
    "spontaneous decision at the moment of speaking (I'll get it!) | FUT-03 offer "
    "(I'll carry that for you) | FUT-04 promise/threat (I'll always love you; "
    "you'll regret this) | FUT-05 willingness/refusal, esp. won't (She won't "
    "answer the phone) | FUT-06 request (Will you close the door?) | FUT-07 "
    "habitual/characteristic will (He'll sit there for hours)."
)
_GOING_SYS = (
    "The bracketed 'be going to' verb group has one of these USES. Choose one:\n"
    "FUT-08 prior intention/plan (I'm going to apply) | FUT-09 prediction from "
    "present evidence (Look out - it's going to fall!) | FUT-10 was/were going to "
    "= unfulfilled intention (I was going to call but forgot)."
)
_PRES_FUT_SYS = (
    "The bracketed present-tense verb group refers to the FUTURE. Choose one:\n"
    "FUT-12 present simple for a timetable/schedule (The show starts at eight) | "
    "FUT-13 present simple after a time/conditional conjunction (I'll call when I "
    "get there)."
)
_FUT11_SYS = ("Return FUT-11 if a present continuous refers to a fixed future "
              "ARRANGEMENT (We're flying on Monday).")
_FUT15_SYS = ("Return FUT-15 if a future continuous expresses the future as a "
              "matter of course, or a polite enquiry (Will you be using the car "
              "tonight?).")
_FUT21_SYS = ("Return FUT-21 if this expresses FUTURE IN THE PAST (would / was to "
              "/ was about to): a future viewpoint from a past reference time "
              "(He didn't know he would never return).")


# --------------------------------------------------------------------------- #
def build(nlp, client=None):
    dets = []

    # --- 'will' readings (LLM). ---------------------------------------------
    dets.append(LLMReadingDetector(
        nlp, ["FUT-01", "FUT-02", "FUT-03", "FUT-04", "FUT-05", "FUT-06",
              "FUT-07"], _WILL_FORM, _WILL_SYS, client=client,
        version="fut-will@0.1"))

    # --- 'be going to' readings (LLM). --------------------------------------
    dets.append(LLMReadingDetector(
        nlp, ["FUT-08", "FUT-09", "FUT-10"], _GOING_TO, _GOING_SYS,
        client=client, version="fut-going-to@0.1"))

    # --- present forms for the future (LLM). --------------------------------
    dets.append(LLMStandaloneDetector("FUT-11", _FUT11_SYS, client=client,
                                      version="fut11-arrangement@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["FUT-12", "FUT-13"], _PRES_SIMPLE, _PRES_FUT_SYS, client=client,
        version="fut-pres-future@0.1"))

    # --- future continuous / perfect / perfect continuous (rule router). ----
    dets.append(RuleRoutingDetector(
        nlp, ["FUT-14", "FUT-16", "FUT-17"], _MODAL_FORM, classify_will_forms,
        version="fut-future-forms@0.1"))
    dets.append(LLMStandaloneDetector("FUT-15", _FUT15_SYS, client=client,
                                      version="fut15-matter-of-course@0.1"))

    # --- be to / be about to / be due to (lexicon/rule). --------------------
    dets.append(RuleRoutingDetector(
        nlp, ["FUT-18"], _BE_TO, _be_to_ok, version="fut18-be-to@0.1"))
    dets.append(RuleRoutingDetector(
        nlp, ["FUT-19"], _BE_ABOUT_TO, _be_about_ok,
        version="fut19-about-to@0.1"))

    fut20_phrases = ["on the point of", "on the verge of"]
    dets.append(CompositeDetector(
        "FUT-20",
        [RuleRoutingDetector(nlp, ["FUT-20"], _BE_DUE_TO, _be_due_ok,
                             version="fut20-due-to@0.1"),
         PhraseLexiconDetector(nlp, "FUT-20", fut20_phrases,
                               version="fut20-phrase@0.1")],
        detector_type="lexicon", version="fut20-composite@0.1"))

    # --- future in the past (LLM). ------------------------------------------
    dets.append(LLMStandaloneDetector("FUT-21", _FUT21_SYS, client=client,
                                      version="fut21-future-in-past@0.1"))

    return dets
