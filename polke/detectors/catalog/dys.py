"""Category DYS — performance phenomena (optional annotation layer), Part VI.

DYS-01 repeats: a run of identical adjacent tokens with no punctuation
between them ("I I I don't know."). Deliberate reduplication is excluded two
ways: intervening punctuation ("It was a very, very long day.") and a small
whitelist of lexicalised reduplicatives ("Bye bye.", "now now").
DYS-02/03/04 (retrace-and-repair, abandoned utterances, syntactic blends)
need meaning-level judgement -> LLM tier, Phase 4.
"""
from __future__ import annotations
from .common import Scan

_REDUPLICATIVES = {"bye", "ha", "blah", "ho", "la", "tut", "now", "there",
                   "hush", "boo", "night", "hear"}


def _dys01(doc):
    i = 0
    while i < len(doc) - 1:
        t = doc[i]
        if t.is_punct or t.lower_ in _REDUPLICATIVES:
            i += 1
            continue
        j = i
        # same surface form AND same tag: "had had" (VBD+VBN perfect) and
        # "that that" (IN+DT) are grammatical doubles, not repeats
        while j + 1 < len(doc) and doc[j + 1].lower_ == t.lower_ \
                and doc[j + 1].tag_ == doc[j].tag_ \
                and not doc[j + 1].is_punct:
            j += 1
        if j > i:
            yield (i, j)
            i = j + 1
        else:
            i += 1


_DYS02_SYS = (
    "Decide whether the utterance shows RETRACE-AND-REPAIR (DYS-02): the "
    "speaker breaks off mid-construction and restarts, replacing or "
    "correcting part of what was begun (often marked with a dash in "
    "transcription).\n"
    "Positive examples: \"She was — he was already there.\" | \"I went to — "
    "we went to the market.\"\n\n"
    "NOT DYS-02 (return NONE):\n"
    "- Complete fluent utterances: \"She was already there.\"\n"
    "- A dash coordinating two complete clauses without any repair: \"She "
    "was tired — he was too.\"\n"
    "- DYS-03 abandoned utterances that trail off WITHOUT restarting: \"I "
    "just thought maybe —\"\n"
    'Return only JSON: {"construct_id": "DYS-02"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')

_DYS03_SYS = (
    "Decide whether the utterance is INCOMPLETE / ABANDONED (DYS-03): the "
    "speaker trails off before completing the construction, leaving it "
    "syntactically unfinished (transcribed with a final dash or ellipsis).\n"
    "Positive examples: \"I just thought maybe —\" | \"If you could just "
    "...\" (trailing off)\n\n"
    "NOT DYS-03 (return NONE):\n"
    "- Complete utterances: \"I just thought maybe we could leave early.\"\n"
    "- DYS-02 repairs where the speaker restarts and finishes: \"She was — "
    "he was already there.\"\n"
    "- Deliberate rhetorical aposiopesis in polished writing.\n"
    'Return only JSON: {"construct_id": "DYS-03"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')

_DYS04_SYS = (
    "Decide whether the utterance is a SYNTACTIC BLEND / anacoluthon "
    "(DYS-04): it starts in one construction and finishes in another, so no "
    "single parse covers it — e.g. a missing relativiser because two frames "
    "were merged.\n"
    "Positive examples: \"That's the one thing is important.\" | \"Most of "
    "the people, when you get there, most are gone.\"\n\n"
    "NOT DYS-04 (return NONE):\n"
    "- The standard construction: \"That's the one thing that is "
    "important.\"\n"
    "- FOC-18 thing-is focus formulae (including double is): \"The thing "
    "is, is that we're broke.\"\n"
    "- Grammatical contact-clauses: \"There's a man wants to see you.\" "
    "(vernacular but systematic)\n"
    'Return only JSON: {"construct_id": "DYS-04"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')


def build(nlp, client=None):
    from ..llm import LLMStandaloneDetector
    return [
        Scan("DYS-01", _dys01, version="dys01-repeats@0.1"),
        LLMStandaloneDetector("DYS-02", _DYS02_SYS, client=client,
                              version="dys02-repair@0.1"),
        LLMStandaloneDetector("DYS-03", _DYS03_SYS, client=client,
                              version="dys03-abandoned@0.1"),
        LLMStandaloneDetector("DYS-04", _DYS04_SYS, client=client,
                              version="dys04-blend@0.1"),
    ]
