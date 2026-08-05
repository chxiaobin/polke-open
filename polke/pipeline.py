"""Run detectors over text and collect annotations.

`nlp` is any callable text -> doc (a spaCy Language, or an identity function for
the offline mock tests).

`context` optionally carries the preceding utterance(s) of a dialogue (oldest
first). Discourse-level detectors (e.g. follow-up questions) opt in by setting
`wants_context = True`; all other detectors keep the plain match(doc, text_id)
interface and never see it.
"""
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence
from .schema import Annotation


def annotate(text: str, nlp, detectors: Iterable, text_id: str = "doc",
             context: Optional[Sequence[str]] = None) -> List[Annotation]:
    doc = nlp(text)
    out: List[Annotation] = []
    for det in detectors:
        if context is not None and getattr(det, "wants_context", False):
            out.extend(det.match(doc, text_id=text_id, context=context))
        else:
            out.extend(det.match(doc, text_id=text_id))
    return out
