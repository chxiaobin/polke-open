"""CEFR level verifier: vocabulary + grammar levels, sentence by sentence.

    from polke.cefr import LevelAnalyzer
    an = LevelAnalyzer()                  # spaCy + detectors + word lists, once
    for rec in an.analyze_iter("I have lived here for years.\\nIt was built in 1900."):
        ...

Two sources of levels:

* **Words** — the Open Language Profiles word lists in data/ (CEFR-J
  Vocabulary Profile 1.5 for A1–B2, Octanove Vocabulary Profile 1.0 for
  C1–C2), looked up by lemma + POS; see polke.vocab.
* **Constructions** — every POLKE annotation carries the CEFR level assigned
  to its construct in data/cefr_levels.csv (EGP / CEFR-J-informed estimates;
  edit the file to recalibrate).

Input is split into sentences by spaCy's parser with every line break forced
to be a boundary, so "one sentence per line" input is honoured exactly. Each
sentence is annotated independently (preceding sentences are passed as
dialogue context for the discourse-level detectors) and tagged for vocabulary.
Exposed as `polke level`, `GET /verify` (UI) and `POST /verify/analyze`.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Optional, Sequence

from . import __version__
from . import llm as llm_mod
from .annotate import Annotator, catalog, resolve_ids
from .env import load_env, max_chars, max_sentences, sentence_concurrency
from .registry import CEFR_LEVELS, UNRATED
from .vocab import LEVEL_RANK, VocabProfile

log = logging.getLogger("polke")


def max_level(levels) -> Optional[str]:
    lv = [l for l in levels if l in LEVEL_RANK]
    return max(lv, key=LEVEL_RANK.__getitem__) if lv else None


class LevelAnalyzer:
    """Wraps an Annotator with the word lists and per-sentence driver.

    Pass an existing `annotator` to share its spaCy pipeline and detectors
    (the server does); otherwise one is built (segment="line").
    """

    def __init__(self, annotator: Optional[Annotator] = None,
                 no_llm: bool = False, llm_status: Optional[dict] = None):
        load_env()
        if annotator is None:
            if llm_status is None and not no_llm:
                llm_status = llm_mod.check_llm()
                if not llm_status["ready"]:
                    log.warning(llm_mod.warning_text(llm_status))
            annotator = Annotator(no_llm=no_llm, llm_status=llm_status,
                                  segment="line")
        self.annotator = annotator
        self.llm_status = annotator.llm_status
        self.nlp = annotator.nlp
        self.vocab = VocabProfile(self.nlp.tokenizer)
        self.catalog = {r["id"]: r for r in catalog()}

    # -- segmentation ----------------------------------------------------
    def split_sentences(self, text: str) -> List[str]:
        """Line breaks always split; within a line the parser decides."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if len(text) > max_chars():
            text = text[:max_chars()]
        out: List[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for s in self.nlp(line).sents:
                st = s.text.strip()
                if st:
                    out.append(st)
        return out[:max_sentences()]

    # -- one sentence ----------------------------------------------------
    def analyze_sentence(self, sentence: str, index: int = 0,
                         context: Optional[Sequence[str]] = None,
                         selection: Optional[Sequence[str]] = None,
                         use_llm: bool = True) -> dict:
        doc = self.nlp(sentence)
        vocab = self.vocab.tag(doc)
        sel = list(selection) if selection else None
        if not use_llm:                      # offline-tier constructs only
            wanted = resolve_ids(sel)
            sel = [cid for cid, r in self.catalog.items()
                   if not r["needs_llm"] and cid in wanted]
        anns = self.annotator.annotate(
            sentence, selection=sel, text_id=f"s{index}",
            context=list(context) if context else None)
        cons = []
        for a in anns:
            row = self.catalog.get(a["construct_id"], {})
            sp = a["span"]
            ev = a.get("evidence") or {}
            cons.append({
                "id": a["construct_id"],
                "name": row.get("name", a["construct_id"]),
                "category": row.get("category", a["construct_id"].split("-")[0]),
                "category_name": row.get("category_name", ""),
                "part": row.get("part", ""),
                "family": row.get("family", ""),
                "level": row.get("cefr_level", UNRATED),
                "tier": a["detector_type"],
                "confidence": a.get("confidence", 1.0),
                "start": sp["start_char"], "end": sp["end_char"],
                "token_start": sp["token_start"], "token_end": sp["token_end"],
                "matched": ev.get("matched")
                           or sentence[sp["start_char"]:sp["end_char"]],
                "rationale": ev.get("rationale"),
                "model": a.get("model"),
            })
        cons.sort(key=lambda c: (c["start"], c["id"]))
        return {
            "type": "sentence",
            "index": index,
            "text": sentence,
            "vocab": vocab.to_dict(),
            "constructions": cons,
            "max_vocab_level": vocab.max_level,
            "max_grammar_level": max_level(c["level"] for c in cons),
        }

    # -- whole text ------------------------------------------------------
    def analyze_iter(self, text: str, selection: Optional[Sequence[str]] = None,
                     use_llm: bool = True, context_window: int = 3
                     ) -> Iterator[dict]:
        """Yield a `meta` record, one `sentence` record per sentence (in
        order), then a `done` record with totals."""
        sentences = self.split_sentences(text)
        llm = dict(self.llm_status)
        if use_llm and not llm.get("ready"):
            use_llm = False
        elif not use_llm and llm.get("ready"):
            llm["reason"] = "disabled for this run"
        llm["used"] = use_llm
        yield {"type": "meta", "version": __version__,
               "sentence_count": len(sentences), "sentences": sentences,
               "llm": llm}
        records: List[dict] = []

        def job(i: int) -> dict:
            ctx = sentences[max(0, i - context_window):i]
            try:
                return self.analyze_sentence(sentences[i], i, context=ctx,
                                             selection=selection,
                                             use_llm=use_llm)
            except Exception as exc:  # noqa: BLE001 — report, keep going
                log.exception("sentence %d failed", i)
                return {"type": "sentence", "index": i, "text": sentences[i],
                        "error": f"{type(exc).__name__}: {exc}",
                        "vocab": {"tokens": [], "phrases": [], "max_level": None},
                        "constructions": [], "max_vocab_level": None,
                        "max_grammar_level": None}

        workers = sentence_concurrency() if use_llm else 1
        if workers <= 1:
            for i in range(len(sentences)):
                rec = job(i)
                records.append(rec)
                yield rec
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(job, i) for i in range(len(sentences))]
                for fut in futs:                 # emit in order
                    rec = fut.result()
                    records.append(rec)
                    yield rec
        yield {"type": "done", "summary": summarize(records)}


def summarize(records: Sequence[dict]) -> dict:
    """Totals over sentence records: words per vocabulary level (multi-word
    entries counted once), constructions per grammar level."""
    vocab_counts: dict = {}
    gram_counts: dict = {}
    words = 0
    for r in records:
        seen = set()
        for t in r["vocab"]["tokens"]:
            if t["kind"] == "skip":
                continue
            if t["phrase"] is not None:
                if t["phrase"] in seen:
                    continue
                seen.add(t["phrase"])
            words += 1
            key = t["level"] or t["kind"]
            vocab_counts[key] = vocab_counts.get(key, 0) + 1
        for c in r["constructions"]:
            gram_counts[c["level"]] = gram_counts.get(c["level"], 0) + 1
    return {"sentences": len(records), "words": words,
            "vocab_levels": vocab_counts, "grammar_levels": gram_counts,
            "errors": sum(1 for r in records if r.get("error"))}


__all__ = ["LevelAnalyzer", "summarize", "max_level", "CEFR_LEVELS", "UNRATED"]
