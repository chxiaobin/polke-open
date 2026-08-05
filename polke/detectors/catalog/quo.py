"""Category QUO — conversational reporting & quotatives, Part VI.

Phase 2 implements the two lexicon-gated members. The formal trigger for
both is the quote frame: quotative verb + (comma) + opening quote mark.
QUO-01 quotative go ("And she goes, 'No way.'") — motion "she goes to
school" has no following quote. QUO-02 be like / be all — discourse like
without a quote (DMG-05) and comparative like (COM-16) lack the be + quote
frame. QUO-03 (hybrid), QUO-04 (rule), QUO-05 (LLM) -> Phases 3/4.
"""
from __future__ import annotations
from ...registry import lexicons
from ...schema import Annotation, Span
from ..base import Detector

_OPEN_QUOTES = {"'", '"', "‘", "“", "`", "``"}


def _quote_follows(doc, i, max_skip=2):
    """An opening quote mark within ``max_skip`` tokens right of i (an
    intervening comma allowed)."""
    j = i + 1
    skipped = 0
    while j < len(doc) and skipped <= max_skip:
        t = doc[j]
        if t.tag_ == "``" or t.text in _OPEN_QUOTES:
            return True
        if t.text == ",":
            j += 1
            skipped += 1
            continue
        return False
    return False


def _mk(det, doc, cid, lo, hi, text_id):
    span = doc[lo:hi + 1]
    return Annotation(
        text_id=text_id, construct_id=cid,
        span=Span(span.start_char, span.end_char, lo, hi),
        detector_type=det.detector_type, detector_version=det.version,
        confidence=1.0,
        evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text})


class _QuotativeGo(Detector):
    detector_type = "lexicon"
    version = "quo01-go@0.1"
    construct_ids = ["QUO-01"]

    def __init__(self):
        self._lemmas = set(lexicons()["quotative_markers"]["go_lemmas"])

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lemma_.lower() not in self._lemmas or t.pos_ not in ("VERB", "AUX"):
                continue
            if not any(k.dep_ in ("nsubj", "nsubjpass") for k in t.children):
                continue
            if _quote_follows(doc, t.i):
                out.append(_mk(self, doc, "QUO-01", t.i, t.i, text_id))
        return out


class _QuotativeBeLikeAll(Detector):
    detector_type = "lexicon"
    version = "quo02-be-like@0.1"
    construct_ids = ["QUO-02"]

    def __init__(self):
        self._comps = set(lexicons()["quotative_markers"]["be_complements"])

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ not in self._comps:
                continue
            be_left = (t.i > 0 and doc[t.i - 1].lemma_ == "be") or \
                      t.head.lemma_ == "be"
            if not be_left:
                continue
            if _quote_follows(doc, t.i):
                lo = t.i - 1 if t.i > 0 and doc[t.i - 1].lemma_ == "be" else t.i
                out.append(_mk(self, doc, "QUO-02", lo, t.i, text_id))
        return out


class _PastProgressiveReport(Detector):
    """QUO-04 (rule tier): reporting verb in the past progressive with a
    clausal complement ("She was saying (that) they might move."). Non-
    reporting progressives ("I was working") fail the lemma set; simple-past
    reporting ("She said that ...") is not progressive."""
    detector_type = "rule"
    version = "quo04-past-prog@0.1"
    construct_ids = ["QUO-04"]

    def __init__(self):
        self._lemmas = set(lexicons()["reporting_verbs"]["lemmas"]) | {"tell"}

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.tag_ != "VBG" or t.lemma_.lower() not in self._lemmas:
                continue
            was = next((c for c in t.children if c.dep_ == "aux"
                        and c.lemma_ == "be" and c.tag_ == "VBD"), None)
            if was is None:
                continue
            if not any(c.dep_ == "ccomp" for c in t.children):
                continue
            out.append(_mk(self, doc, "QUO-04", was.i, t.i, text_id))
        return out


# QUO-03 (hybrid): present-tense 'say' is one FORM with two register
# readings; the LLM routes between conversational narrative (QUO-03) and
# news/summary reporting (REP-11, whose own detector in rep.py registers
# later and keeps ownership of the id).
_QUO03_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"LEMMA": "say",
                                      "TAG": {"IN": ["VBZ", "VBP"]}}},
]

_QUO03_SYS = (
    "The sentence reports speech with PRESENT-tense 'say(s)' (marked "
    "[[ ]]). Decide which construct it realises:\n\n"
    "QUO-03 — CONVERSATIONAL HISTORIC-PRESENT reporting: informal narrative "
    "of a past conversation, personal subject, often a direct quote and a "
    "narrative connective (so / and then); includes vernacular 'I says'.\n"
    "Examples: \"So he says, 'Come back Monday.'\" | \"Then she says she's "
    "not interested.\"\n\n"
    "REP-11 — reporting with present tense in NEWS/SUMMARY register: "
    "institutional or text source summarised with timeless present.\n"
    "Examples: \"The report says that inflation is rising.\" | \"The law "
    "says you must register.\"\n\n"
    "If the sentence is a general truth-attribution (\"People say it's "
    "haunted\") or otherwise neither, return NONE.\n"
    'Return only JSON: {"construct_id": "QUO-03"|"REP-11"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')

_QUO05_SYS = (
    "Decide whether the sentence is FREE direct or FREE indirect "
    "speech/thought (QUO-05): a character's words or thoughts rendered as "
    "narrative WITHOUT any reporting frame ('she said/thought that ...'), "
    "typically with backshifted tense, third person, and questions or "
    "deictics anchored to the character.\n"
    "Positive examples: \"He'd be back tomorrow, he was sure.\" | \"Would "
    "she come?\" (as narrative) | \"Why had nobody told him?\"\n\n"
    "NOT QUO-05 (return NONE):\n"
    "- REP-01 framed indirect report: \"She said that she was tired.\"\n"
    "- Framed direct speech: \"He said, 'I'll be back tomorrow.'\"\n"
    "- Framed thought: \"He wondered whether she would come.\"\n"
    "- Plain narration or dialogue with quotation marks.\n"
    'Return only JSON: {"construct_id": "QUO-05"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')


def build(nlp, client=None):
    from ..llm import LLMReadingDetector, LLMStandaloneDetector
    return [
        _QuotativeGo(), _QuotativeBeLikeAll(), _PastProgressiveReport(),
        LLMReadingDetector(nlp, ["QUO-03", "REP-11"], _QUO03_FORM, _QUO03_SYS,
                           client=client, version="quo03-historic@0.1"),
        LLMStandaloneDetector("QUO-05", _QUO05_SYS, client=client,
                              version="quo05-free-speech@0.1"),
    ]
