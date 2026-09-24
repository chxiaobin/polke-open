"""Vocabulary CEFR levelling with the Open Language Profiles word lists.

Part of the level verifier (see polke.cefr).

Two lists (shipped in ``data/``):

* CEFR-J Vocabulary Profile 1.5 — A1–B2 (Tono Lab, TUFS)
* Octanove Vocabulary Profile 1.0 — C1–C2 (Octanove Labs, CC BY-SA 4.0)

Each row is ``headword,pos,CEFR``.  Headwords may list spelling variants
separated by ``/`` (``a.m./A.M./am/AM``) and may be multi-word (``bus stop``,
``according to``).  Lookup is lemma+POS first, then surface form, then any
POS (flagged ``pos_match=False``).  Multi-word entries are matched greedily
over token lemma / lowercase sequences before single-token lookup.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .registry import DATA as DATA_DIR
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_RANK = {lv: i for i, lv in enumerate(LEVELS)}

LISTS = [
    ("cefrj", "cefrj-vocabulary-profile-1.5.csv"),
    ("octanove", "octanove-vocabulary-profile-c1c2-1.0.csv"),
]

# spaCy universal POS -> CEFR-J pos labels, in order of preference.
POS_MAP: Dict[str, Tuple[str, ...]] = {
    "NOUN": ("noun",),
    "PROPN": ("noun",),
    "VERB": ("verb", "be-verb", "have-verb", "do-verb"),
    "AUX": ("modal auxiliary", "be-verb", "have-verb", "do-verb", "verb"),
    "ADJ": ("adjective", "noun"),
    "ADV": ("adverb", "adjective", "preposition"),
    "PRON": ("pronoun", "determiner", "noun"),
    "DET": ("determiner", "pronoun", "adjective"),
    "ADP": ("preposition", "adverb", "conjunction"),
    "SCONJ": ("conjunction", "preposition", "adverb"),
    "CCONJ": ("conjunction", "adverb"),
    "NUM": ("number", "noun", "determiner"),
    "INTJ": ("interjection", "adverb", "noun"),
    "PART": ("infinitive-to", "adverb", "preposition"),
}
SKIP_POS = {"PUNCT", "SPACE", "SYM", "X"}


@dataclass
class Entry:
    headword: str
    pos: str
    level: str
    source: str


@dataclass
class TokenTag:
    i: int
    text: str
    ws: str            # trailing whitespace
    start: int         # char offsets within the sentence
    end: int
    lemma: str
    pos: str
    tag: str
    kind: str          # word | skip | proper | unlisted
    level: Optional[str] = None
    headword: Optional[str] = None
    list_pos: Optional[str] = None
    source: Optional[str] = None
    pos_match: bool = True
    phrase: Optional[int] = None   # index into VocabResult.phrases

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Phrase:
    tokens: List[int]
    headword: str
    level: str
    source: str
    list_pos: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class VocabResult:
    tokens: List[TokenTag] = field(default_factory=list)
    phrases: List[Phrase] = field(default_factory=list)

    @property
    def max_level(self) -> Optional[str]:
        lv = [t.level for t in self.tokens if t.level]
        lv += [p.level for p in self.phrases]
        return max(lv, key=LEVEL_RANK.__getitem__) if lv else None

    def to_dict(self) -> dict:
        return {"tokens": [t.to_dict() for t in self.tokens],
                "phrases": [p.to_dict() for p in self.phrases],
                "max_level": self.max_level}


def _variants(headword: str) -> List[str]:
    """``a.m./A.M./am/AM`` -> [a.m., A.M., am, AM]; keeps ``check-in``."""
    parts = [p.strip() for p in headword.split("/")]
    return [p for p in parts if p]


def _lower_tokens(tokenizer, text: str) -> Tuple[str, ...]:
    return tuple(t.text.lower() for t in tokenizer(text) if not t.is_space)


class VocabProfile:
    """In-memory index over the word lists.

    ``tokenizer`` (spaCy ``nlp.tokenizer`` or ``nlp.make_doc``) is used so that
    multi-word headwords are split exactly like the input text.
    """

    def __init__(self, tokenizer, data_dir: Path = DATA_DIR):
        self.tokenizer = tokenizer
        self.single: Dict[str, List[Entry]] = {}
        self.multi: Dict[Tuple[str, ...], List[Entry]] = {}
        self.max_phrase_len = 1
        self.counts: Dict[str, int] = {}
        for source, fname in LISTS:
            self._load(source, data_dir / fname)

    def _load(self, source: str, path: Path) -> None:
        n = 0
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                hw = (row.get("headword") or "").strip()
                level = (row.get("CEFR") or "").strip().upper()
                pos = (row.get("pos") or "").strip().lower()
                if not hw or level not in LEVEL_RANK:
                    continue
                n += 1
                for v in _variants(hw):
                    e = Entry(v, pos, level, source)
                    toks = _lower_tokens(self.tokenizer, v)
                    if len(toks) > 1:
                        self.multi.setdefault(toks, []).append(e)
                        self.max_phrase_len = max(self.max_phrase_len, len(toks))
                    self.single.setdefault(v.lower(), []).append(e)
        self.counts[source] = n

    # -- lookups ---------------------------------------------------------
    @staticmethod
    def _best(entries: Iterable[Entry]) -> Optional[Entry]:
        entries = list(entries)
        if not entries:
            return None
        return min(entries, key=lambda e: LEVEL_RANK[e.level])

    def lookup(self, form: str, upos: Optional[str] = None
               ) -> Tuple[Optional[Entry], bool]:
        """Return (entry, pos_matched). Tries POS-compatible entries first,
        then any entry (lowest level)."""
        entries = self.single.get(form.lower())
        if not entries:
            return None, False
        if upos:
            for want in POS_MAP.get(upos, ()):
                hit = self._best(e for e in entries if e.pos == want)
                if hit:
                    return hit, True
        return self._best(entries), False

    def lookup_all(self, form: str) -> List[Entry]:
        return list(self.single.get(form.lower(), []))

    # -- tagging ---------------------------------------------------------
    def tag(self, doc) -> VocabResult:
        """Tag a spaCy Doc/Span (one sentence). Char offsets are relative to
        ``doc`` start."""
        base = doc[0].idx if len(doc) else 0
        toks = list(doc)
        res = VocabResult()
        for t in toks:
            res.tokens.append(TokenTag(
                i=t.i - toks[0].i, text=t.text, ws=t.whitespace_,
                start=t.idx - base, end=t.idx - base + len(t.text),
                lemma=t.lemma_, pos=t.pos_, tag=t.tag_, kind="word"))

        # multi-word entries: greedy longest match, lemma seq or lower seq
        n = len(toks)
        i = 0
        lowers = [t.text.lower() for t in toks]
        lemmas = [t.lemma_.lower() for t in toks]
        while i < n:
            found = None
            for L in range(min(self.max_phrase_len, n - i), 1, -1):
                for seq in (tuple(lowers[i:i + L]), tuple(lemmas[i:i + L])):
                    entries = self.multi.get(seq)
                    if entries:
                        found = (L, self._best(entries))
                        break
                if found:
                    break
            if found:
                L, e = found
                idx = len(res.phrases)
                res.phrases.append(Phrase(
                    tokens=list(range(i, i + L)), headword=e.headword,
                    level=e.level, source=e.source, list_pos=e.pos))
                for k in range(i, i + L):
                    tt = res.tokens[k]
                    tt.phrase = idx
                    tt.level = e.level
                    tt.headword = e.headword
                    tt.list_pos = e.pos
                    tt.source = e.source
                i += L
            else:
                i += 1

        # single tokens
        for t, tt in zip(toks, res.tokens):
            if tt.phrase is not None:
                continue
            if t.pos_ in SKIP_POS or t.is_space:
                tt.kind = "skip"
                continue
            if t.pos_ == "NUM" and re.fullmatch(r"[\d.,:%/-]+", t.text):
                tt.kind = "skip"
                continue
            if t.tag_ == "POS":          # possessive 's
                tt.kind = "skip"
                continue
            e, ok = self.lookup(t.lemma_, t.pos_)
            if e is None:
                e, ok = self.lookup(t.text, t.pos_)
            if e is None and t.text.lower() != t.lemma_.lower():
                # e.g. spaCy lemma "U.S." vs list "US"
                e, ok = self.lookup(t.text.replace(".", ""), t.pos_)
            if e is None:
                tt.kind = "proper" if t.pos_ == "PROPN" else "unlisted"
                continue
            tt.level = e.level
            tt.headword = e.headword
            tt.list_pos = e.pos
            tt.source = e.source
            tt.pos_match = ok
        return res
