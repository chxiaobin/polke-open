"""High-level annotation API.

Wraps the detector kit for library, CLI, and server use:

    from polke.annotate import Annotator
    ann = Annotator()                       # loads spaCy, builds all detectors
    ann.annotate("She has lived here for years.")

The heavy pieces (spaCy pipeline, detector build, LLM probe) happen once in the
constructor; annotate() is then cheap per text.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence

from . import llm as llm_mod
from .env import llm_concurrency, load_env, segment_mode, spacy_model
from .registry import constructs


def _line_senter(doc):
    """Sentence boundaries at line breaks ONLY: each input line is one
    sentence. For transcribed speech (one utterance per line) where the
    parser's punctuation-driven segmentation is unreliable."""
    for i, tok in enumerate(doc):
        tok.is_sent_start = i == 0 or "\n" in doc[i - 1].text
    return doc


def load_nlp(model: Optional[str] = None, segment: Optional[str] = None):
    """Load the spaCy pipeline, with an actionable error if the model is
    missing.

    `segment` — "parser" (default) or "line" (each input line = one
    sentence); falls back to the POLKE_SEGMENT environment variable.
    """
    import spacy
    name = model or spacy_model()
    try:
        nlp = spacy.load(name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {name!r} is not installed — run: "
            f"python -m spacy download {name}") from exc
    if (segment or segment_mode()) == "line":
        from spacy.language import Language
        if not Language.has_factory("polke_line_senter"):
            Language.component("polke_line_senter", func=_line_senter)
        nlp.add_pipe("polke_line_senter",
                     before="parser" if nlp.has_pipe("parser") else None)
    return nlp


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
            "definition": c.get("definition_raw", ""),
            "use_notes": (c.get("use_notes") or "").strip(),
        })
    rows.sort(key=lambda r: (r["category"], r["id"]))
    return rows


class Annotator:
    """One spaCy pipeline + the full detector registry + one LLM client."""

    def __init__(self, spacy_model_name: Optional[str] = None,
                 no_llm: bool = False, llm_status: Optional[dict] = None,
                 segment: Optional[str] = None):
        load_env()
        self.nlp = load_nlp(spacy_model_name, segment=segment)
        client, status = llm_mod.build_client(no_llm=no_llm, status=llm_status)
        self.llm_status = status
        from .build import build_all
        build_all(self.nlp, llm_client=client)

    def annotate(self, text: str, selection: Optional[Sequence[str]] = None,
                 text_id: str = "doc",
                 context: Optional[Sequence[str]] = None,
                 progress: Optional[Callable[[int, int], None]] = None
                 ) -> List[dict]:
        """Annotate one text. Returns Annotation dicts sorted by span.

        `selection` — construct ids and/or category prefixes (None = all).
        `context` — preceding dialogue utterances (oldest first) for the
        discourse-level detectors.
        `progress` — called as progress(done, total) after each LLM judgment
        (total = number of LLM calls this text needs; never called when 0).
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
        # Offline detectors run inline; LLM-stage detectors only contribute
        # their pending judgments here, which then run on a thread pool (each
        # task is one classify() call over pre-extracted strings — no doc
        # access off the main thread).
        tasks = []
        for det in dets:
            wants_ctx = (context is not None
                         and getattr(det, "wants_context", False))
            if hasattr(det, "llm_tasks"):
                if wants_ctx:
                    tasks.extend(det.llm_tasks(doc, text_id=text_id,
                                               context=context))
                else:
                    tasks.extend(det.llm_tasks(doc, text_id=text_id))
                continue
            if wants_ctx:
                anns = det.match(doc, text_id=text_id, context=context)
            else:
                anns = det.match(doc, text_id=text_id)
            for a in anns:
                # A group router emits for its whole group; keep what was asked.
                if a.construct_id in wanted:
                    out.append(a.to_dict())
        if tasks:
            workers = min(llm_concurrency(), len(tasks))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(t) for t in tasks]
                for done, fut in enumerate(as_completed(futures), 1):
                    a = fut.result()
                    if progress is not None:
                        progress(done, len(tasks))
                    if a is not None and a.construct_id in wanted:
                        out.append(a.to_dict())
        out.sort(key=lambda a: (a["span"]["token_start"], a["construct_id"]))
        return out
