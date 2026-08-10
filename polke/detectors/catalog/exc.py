"""Category EXC — exclamatives, Part VI.

EXC-01 what-exclamative: initial *what* functioning as a determiner of an NP
("What a day (it was)!", "What nonsense!") — interrogative *what* is a
pronoun/object followed by an operator ("What did you do?") and questions
end in "?", which is the exclude. EXC-02 how-exclamative: initial *how* +
adjective/adverb without subject–operator inversion and without "?"
("How lovely!", "How quickly time passes!" vs "How old are you?").
EXC-03 (verbless/phrasal) is hybrid tier -> Phase 4.
"""
from __future__ import annotations
from ...schema import Annotation, Span
from ..base import Detector
from ..llm import DummyClient
from .common import Scan, ends_with

_EXC03_SYS = (
    "The utterance is an exclamation without a canonical finite clause. "
    "Decide which construct it realises:\n\n"
    "EXC-03 — verbless / phrasal EXCLAMATIVE: an evaluative judgement of "
    "something, expressed as an NP/AdjP fragment or a fronted-NP frame.\n"
    "Examples: \"Nice one!\" | \"The cheek of it!\" | \"Some help you were!\" "
    "| \"Brilliant!\"\n\n"
    "INS-01 — pure INTERJECTION: an emotion noise with no evaluative "
    "predication of a referent.\nExamples: \"Oh!\" | \"Wow!\" | \"Ouch!\" "
    "(likewise expletive interjections \"Damn!\", \"Bloody hell!\" -> "
    "INS-01 here).\n\n"
    'Return only JSON: {"construct_id": "EXC-03"|"INS-01"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')


class _VerblessExclamative(Detector):
    """EXC-03 (hybrid): the rule proposes '!'-final units that are not
    what/how exclamatives (EXC-01/02) and not imperatives (base-verb root);
    the LLM routes the survivors between EXC-03 and INS-01."""
    detector_type = "hybrid_rule_llm"
    version = "exc03-hybrid@0.1"
    construct_ids = ["EXC-03"]

    def __init__(self, client=None):
        self._client = client or DummyClient()

    def llm_tasks(self, doc, text_id="doc"):
        tasks = []
        for sent in doc.sents:
            toks = [t for t in sent if not t.is_punct and not t.is_space]
            if not toks:
                continue
            first = toks[0]
            if first.lower_ in ("what", "how"):
                continue                    # EXC-01 / EXC-02
            if sent.root.tag_ == "VB":
                continue                    # imperative ("Listen!")
            bang = ends_with(sent, "!")
            # Speech path: transcripts have no "!", so also propose short
            # verbless evaluative fragments ("nice one", "brilliant").
            fragment = (not bang and len(toks) <= 6
                        and not ends_with(sent, "?")
                        and not any(t.pos_ in ("VERB", "AUX") for t in toks)
                        and any(t.pos_ == "ADJ" for t in toks))
            if not (bang or fragment):
                continue
            sp = Span(sent.start_char, sent.end_char, sent.start, sent.end - 1)
            tasks.append(lambda user=sent.text, sp=sp:
                         self._judge(user, sp, text_id))
        return tasks

    def _judge(self, user, sp, text_id):
        res = self._client.classify(_EXC03_SYS, user, ["EXC-03", "INS-01"])
        if res.get("construct_id") != "EXC-03":
            return None
        return Annotation(
            text_id=text_id, construct_id="EXC-03", span=sp,
            detector_type=self.detector_type, detector_version=self.version,
            confidence=float(res.get("confidence", 0.0)),
            model=getattr(self._client, "model", None),
            evidence={"rationale": res.get("rationale", "")})

    def match(self, doc, text_id="doc"):
        anns = (t() for t in self.llm_tasks(doc, text_id=text_id))
        return [a for a in anns if a is not None]


def _exc01(doc):
    for sent in doc.sents:
        if ends_with(sent, "?"):
            continue
        first = next((t for t in sent if not t.is_punct), None)
        if first is None or first.lower_ != "what":
            continue
        if first.i + 1 >= sent.end:
            continue
        nxt = doc[first.i + 1]
        # exclamative what determines an NP: what + a/an or bare noun/adj
        if not (first.dep_ == "det" or nxt.lower_ in ("a", "an")
                or nxt.pos_ in ("NOUN", "ADJ")):
            continue
        yield (first.i, first.head.i if first.dep_ == "det" else nxt.i)


def _exc02(doc):
    for sent in doc.sents:
        if ends_with(sent, "?"):
            continue
        first = next((t for t in sent if not t.is_punct), None)
        if first is None or first.lower_ != "how":
            continue
        if first.i + 1 >= sent.end:
            continue
        nxt = doc[first.i + 1]
        if nxt.pos_ not in ("ADJ", "ADV"):
            continue
        # no subject-operator inversion after the how-phrase ("How old are you")
        j = first.i + 2
        if j < sent.end and doc[j].pos_ == "AUX" and j + 1 < sent.end \
                and doc[j + 1].pos_ in ("PRON", "PROPN", "NOUN", "DET"):
            continue
        yield (first.i, nxt.i)


def build(nlp, client=None):
    return [
        Scan("EXC-01", _exc01, version="exc01-what@0.1"),
        Scan("EXC-02", _exc02, version="exc02-how@0.1"),
        _VerblessExclamative(client=client),
    ]
