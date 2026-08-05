"""High-level annotation API.

Wraps the detector kit for library, CLI, and server use:

    from polke.annotate import Annotator
    ann = Annotator()                       # loads spaCy, builds all detectors
    ann.annotate("She has lived here for years.")

The heavy pieces (spaCy pipeline, detector build, LLM probe) happen once in the
constructor; annotate() is then cheap per text.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from . import llm as llm_mod
from .env import load_env, spacy_model
from .registry import constructs


def load_nlp(model: Optional[str] = None):
    """Load the spaCy pipeline, with an actionable error if the model is
    missing."""
    import spacy
    name = model or spacy_model()
    try:
        return spacy.load(name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {name!r} is not installed — run: "
            f"python -m spacy download {name}") from exc


def resolve_ids(selection: Optional[Sequence[str]]) -> set:
    """Expand category prefixes and/or construct ids to a set of ids.

    'PAS' -> all PAS-*; 'PAS-01' -> {'PAS-01'}; None/empty -> ALL constructs.
    Unknown entries are ignored.
    """
    all_ids = set(constructs().keys())
    if not selection:
        return all_ids
    by_cat: dict = {}
    for cid in all_ids:
        by_cat.setdefault(cid.split("-")[0], set()).add(cid)
    out: set = set()
    for item in selection:
        item = str(item).strip()
        if item in all_ids:
            out.add(item)
        elif item.upper() in by_cat:
            out |= by_cat[item.upper()]
    return out


def catalog() -> list:
    """The construct inventory, one row per construct (for pickers/docs)."""
    rows = []
    for c in constructs().values():
        rows.append({
            "id": c["id"],
            "category": c["category"],
            "category_name": c.get("category_name", c["category"]),
            "part": c.get("part", ""),
            "family": c.get("family", ""),
            "name": c["name"],
            "example": c.get("example", ""),
            "detector_type": c.get("detector_type", ""),
            "needs_llm": c.get("detector_type", "") in llm_mod.LLM_TYPES,
        })
    rows.sort(key=lambda r: (r["category"], r["id"]))
    return rows


class Annotator:
    """One spaCy pipeline + the full detector registry + one LLM client."""

    def __init__(self, spacy_model_name: Optional[str] = None,
                 no_llm: bool = False, llm_status: Optional[dict] = None):
        load_env()
        self.nlp = load_nlp(spacy_model_name)
        client, status = llm_mod.build_client(no_llm=no_llm, status=llm_status)
        self.llm_status = status
        from .build import build_all
        build_all(self.nlp, llm_client=client)

    def annotate(self, text: str, selection: Optional[Sequence[str]] = None,
                 text_id: str = "doc",
                 context: Optional[Sequence[str]] = None) -> List[dict]:
        """Annotate one text. Returns Annotation dicts sorted by span.

        `selection` — construct ids and/or category prefixes (None = all).
        `context` — preceding dialogue utterances (oldest first) for the
        discourse-level detectors.
        """
        wanted = resolve_ids(selection)
        if not wanted:
            return []
        from .detectors.base import registry
        reg = registry()
        seen, dets = set(), []
        for cid in wanted:
            det = reg.get(cid)
            if det is not None and id(det) not in seen:
                seen.add(id(det))
                dets.append(det)
        doc = self.nlp(text)
        out = []
        for det in dets:
            if context is not None and getattr(det, "wants_context", False):
                anns = det.match(doc, text_id=text_id, context=context)
            else:
                anns = det.match(doc, text_id=text_id)
            for a in anns:
                # A group router emits for its whole group; keep what was asked.
                if a.construct_id in wanted:
                    out.append(a.to_dict())
        out.sort(key=lambda a: (a["span"]["token_start"], a["construct_id"]))
        return out
