"""Category DMG — discourse markers at item grain, Part VI.

One InitialMarkerDetector serves DMG-01/02/03/04/06/07/08 from the
``discourse_markers`` lexicon table: the marker must open the utterance and
be followed by more material (else it is a freestanding insert, INS-*).
Items whose surface form has heavy non-marker competition (well, you know,
i mean, so, right, okay) additionally require a following comma or another
discourse-marker word — that is what keeps "The well in the garden", "You
know the answer", "I mean it" and "She was so tired" out. COH-11 stays
registered as the coarse residual detector; per-item DMG hits refine it.

DMG-05 (discourse like) is LLM tier and DMG-09 (turn-initial and/but) rule
tier -> Phases 3/4.
"""
from __future__ import annotations
from ...registry import lexicons
from ..llm import LLMStandaloneDetector
from ..spoken import InitialMarkerDetector

_DMG05_SYS = (
    "Decide whether the sentence contains DISCOURSE 'like' (DMG-05): 'like' "
    "used as a filler, focus marker or approximator, contributing no "
    "propositional content and removable without changing truth conditions.\n"
    "Positive examples: \"It was like really strange.\" | \"There were like "
    "fifty people.\" | \"He was like so annoyed.\"\n\n"
    "NOT DMG-05 (return NONE):\n"
    "- COM-16 comparison/preposition: \"He runs like the wind.\" | \"like "
    "his father\"\n"
    "- QUO-02 quotative BE like introducing reported speech: \"I'm like, "
    "'What?'\"\n"
    "- Lexical verb: \"I like tea.\"\n"
    "- Suffixal/conjunction uses: \"It looks like rain.\"\n"
    'Return only JSON: {"construct_id": "DMG-05"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')


def _has_like(sent):
    return any(t.lower_ == "like" for t in sent)


def build(nlp, client=None):
    table = lexicons()["discourse_markers"]
    entries = [(e["phrase"], e["id"], e["comma"]) for e in table["entries"]]
    # DMG-09 turn-initial and/but (rule tier): same positional grammar —
    # utterance-initial coordinator with a continuation; clause-medial
    # coordination ("I came and saw") is never sentence-initial.
    dmg09 = [("and", "DMG-09", "none"), ("but", "DMG-09", "none")]
    return [
        InitialMarkerDetector(nlp, entries, table["dm_followers"],
                              version="dmg-initial@0.1"),
        InitialMarkerDetector(nlp, dmg09, table["dm_followers"],
                              version="dmg09-and-but@0.2", line_initial=True),
        LLMStandaloneDetector("DMG-05", _DMG05_SYS, client=client,
                              gate=_has_like, version="dmg05-like@0.1"),
    ]
