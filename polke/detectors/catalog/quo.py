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
                continue
            # spoken path (no quote marks): motion go has a complement to its
            # right (go WITH X, go ROUND); quotative go opens directly into
            # speech — an interjection/pronoun clause or a capitalised word.
            if any(k.dep_ in ("prep", "prt", "dobj", "attr", "oprd")
                   and k.i > t.i for k in t.children):
                continue
            nxt = next((x for x in doc[t.i + 1:]
                        if not x.is_space and not x.is_punct), None)
            titled = nxt is not None and nxt.is_title and nxt.i > 0
            if _speechy_follows(doc, t.i) or titled:
                out.append(_mk(self, doc, "QUO-01", t.i, t.i, text_id))
        return out


_QUOTE_OPENERS = {"oh", "no", "yeah", "yes", "what", "why", "how", "wow",
                  "whoa", "hey", "ooh", "nah", "okay", "ok", "right", "well"}


_ACC_PRONOUNS = {"me", "him", "her", "us", "them", "it"}
_PERSON_SUBJ = {"i", "you", "he", "she", "we", "they", "everyone", "everybody"}


def _speechy_follows(doc, i):
    """Speech path for the quotative (transcripts carry no quote marks):
    be + like/all counts as quotative when what follows opens direct speech
    — a pronoun-subject clause or an interjection ("I was like oh my god",
    "he's like no way"). Comparative "he's like his dad" (PRP$), "be like
    THEM" (accusative pronoun) and approximator "it's like really strange"
    (RB) do not match."""
    # scan within the same LINE (utterance) — the parser often opens a new
    # sentence exactly at the quote ("I was like | oh my god"), so the
    # sentence boundary must not stop the scan, but a line break must
    for t in doc[i + 1:]:
        if "\n" in t.text:
            break
        if t.is_space or t.is_punct:
            continue
        if t.tag_ == "PRP" and t.lower_ in _ACC_PRONOUNS:
            return False              # "be like THEM" = comparison, not quote
        return (t.tag_ in ("PRP", "UH", "WP", "WRB")
                or t.lower_ in _QUOTE_OPENERS)
    return False


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
            be = None
            if t.i > 0 and doc[t.i - 1].lemma_ == "be":
                be = doc[t.i - 1]
            elif t.head.lemma_ == "be":
                be = t.head
            if be is None:
                continue
            # the quoter must be a person: "it's like she's moody" (similative)
            # and "that's all I remember" have non-person subjects
            subj = next((k for k in be.children
                         if k.dep_ in ("nsubj", "nsubjpass")), None)
            if subj is None and be.head is not be:
                subj = next((k for k in be.head.children
                             if k.dep_ in ("nsubj", "nsubjpass")), None)
            if subj is not None and subj.pos_ != "PROPN" \
                    and subj.lower_ not in _PERSON_SUBJ:
                continue
            if _quote_follows(doc, t.i) or _speechy_follows(doc, t.i):
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
    "- In casual conversation, the speaker's OWN current statements, "
    "opinions or questions ('I'd rather her spend time with Sarah', 'is it "
    "just two?') are ordinary dialogue, NOT free indirect speech. QUO-05 "
    "requires a narrated past scene whose words/thoughts are re-enacted "
    "without a frame.\n"
    'Return only JSON: {"construct_id": "QUO-05"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')


def _quo05_gate(sent):
    """Free indirect speech needs a backshift cue: a past-tense verb or a
    past-oriented modal somewhere in the sentence."""
    return any(t.tag_ == "VBD"
               or (t.tag_ == "MD" and t.lower_ in ("would", "could", "might"))
               for t in sent)


def build(nlp, client=None):
    from ..llm import LLMReadingDetector, LLMStandaloneDetector
    return [
        _QuotativeGo(), _QuotativeBeLikeAll(), _PastProgressiveReport(),
        LLMReadingDetector(nlp, ["QUO-03", "REP-11"], _QUO03_FORM, _QUO03_SYS,
                           client=client, version="quo03-historic@0.1"),
        LLMStandaloneDetector("QUO-05", _QUO05_SYS, client=client,
                              version="quo05-free-speech@0.2",
                              gate=_quo05_gate),
    ]
