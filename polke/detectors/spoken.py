"""Position-aware lexicon detectors for the Part VI spoken/interactional tier.

Membership in a lexicon is necessary but not sufficient for these constructs:
an insert must be a freestanding C-unit ("Oh!"), a discourse marker must be
utterance-initial with a continuation ("Well, let's see." but not "The well in
the garden"), an interactive parenthetical medial/final ("He was, you know,
upset" but not "You know the answer"), a vocative comma-peripheral, an
invariant tag clause-final before "?". Each class here encodes one of those
positional grammars over a phrase lexicon.

Phrases are normalised through ``nlp.tokenizer`` at build time, so items that
spaCy splits ("ain't" -> ai + n't, "you're welcome" -> you + 're + welcome)
match the same way they tokenise in running text.
"""
from __future__ import annotations
from .catalog.common import ends_with
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from ..schema import Annotation, Span
from .base import Detector

_OPENERS = {"(", "[", "-", "--", "—", "–", '"', "'", "‘", "“", "``"}


def tokenize_phrase(nlp, phrase: str) -> Tuple[str, ...]:
    """The phrase as the tuple of lower-cased tokens en_core_web_sm produces."""
    return tuple(t.lower_ for t in nlp.tokenizer(phrase))


def _norm_entries(nlp, phrases, default_cid):
    """[(token-tuple, construct_id), ...] from strings or (phrase, cid) pairs."""
    out = []
    for p in phrases:
        if isinstance(p, (tuple, list)):
            out.append((tokenize_phrase(nlp, p[0]), p[1]))
        else:
            out.append((tokenize_phrase(nlp, p), default_cid))
    return out


def _content_start(doc, sent):
    """Index of the first non-punct token of the sentence."""
    for t in sent:
        if not (t.is_punct or t.is_space):
            return t.i
    return sent.start


def _mk(det, doc, cid, lo, hi, text_id, **evidence):
    span = doc[lo:hi + 1]
    return Annotation(
        text_id=text_id, construct_id=cid,
        span=Span(span.start_char, span.end_char, lo, hi),
        detector_type=det.detector_type, detector_version=det.version,
        confidence=1.0,
        evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text,
                  **evidence})


class FreestandingUnitDetector(Detector):
    """Fires when a whole C-unit (sentence) consists of one lexicon phrase.

    ``question=True`` additionally requires the unit to end in "?",
    ``question=False`` requires it NOT to (separates response elicitors
    "Right?" from response tokens "Right."). ``None`` = don't care.
    """
    detector_type = "lexicon"

    def __init__(self, nlp, construct_ids, phrases, question: Optional[bool] = None,
                 version: str = "0.1"):
        self.construct_ids = [construct_ids] if isinstance(construct_ids, str) else list(construct_ids)
        self.version = version
        self._entries = {toks: cid for toks, cid
                         in _norm_entries(nlp, phrases, self.construct_ids[0])}
        self._question = question

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        for sent in doc.sents:
            content = [t for t in sent if not (t.is_punct or t.is_space)]
            if not content:
                continue
            key = tuple(t.lower_ for t in content)
            cid = self._entries.get(key)
            if cid is None or cid not in self.construct_ids:
                continue
            is_q = "?" in doc[content[-1].i + 1:sent.end].text
            if self._question is True and not is_q:
                continue
            if self._question is False and is_q:
                continue
            out.append(_mk(self, doc, cid, content[0].i, content[-1].i, text_id))
        return out


class InitialMarkerDetector(Detector):
    """Utterance-initial discourse marker + continuation (DMG-01..08).

    entries: [(phrase, construct_id, comma_mode)] where comma_mode is
    "require" (a comma or another discourse-marker word must follow) or
    "none". A marker with no continuation is a freestanding insert (INS-*),
    not a discourse marker, so it never fires here. Longest entry wins
    ("oh right" over "oh").
    """
    detector_type = "lexicon"

    def __init__(self, nlp, entries, dm_followers: Iterable[str],
                 version: str = "0.1"):
        self.version = version
        self._entries = []  # (token-tuple, cid, comma_mode) longest first
        cids = []
        for phrase, cid, comma in entries:
            self._entries.append((tokenize_phrase(nlp, phrase), cid, comma))
            cids.append(cid)
        self._entries.sort(key=lambda e: -len(e[0]))
        self.construct_ids = sorted(set(cids))
        self._followers = {f.lower() for f in dm_followers}

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        for sent in doc.sents:
            start = _content_start(doc, sent)
            for toks, cid, comma in self._entries:
                end = start + len(toks)  # exclusive
                if end > sent.end:
                    continue
                if tuple(t.lower_ for t in doc[start:end]) != toks:
                    continue
                # continuation: some non-punct token after the marker
                rest = [t for t in doc[end:sent.end]
                        if not (t.is_punct or t.is_space)]
                if not rest:
                    break  # freestanding -> INS territory
                if comma == "require":
                    nxt = doc[end]
                    if not (nxt.text == "," or nxt.lower_ in self._followers):
                        break
                out.append(_mk(self, doc, cid, start, end - 1, text_id))
                break  # longest match only, one marker per utterance start
        return out


class DelimitedInsertDetector(Detector):
    """A hesitator / filled pause anywhere in the utterance: er, erm, um, uh
    (INS-06). The items are lexically unambiguous, so no punctuation
    delimitation is required — transcribed speech has none ("erm yes in
    terms of...", a bare "erm" line). The one exclusion is the
    metalinguistic MENTION, where the item fills an argument slot ("Um is a
    common hesitation marker" — 'um' as subject)."""
    detector_type = "lexicon"

    _ARG_DEPS = {"nsubj", "nsubjpass", "dobj", "pobj", "attr", "poss",
                 "dative", "oprd"}

    def __init__(self, nlp, construct_id: str, items: Iterable[str],
                 version: str = "0.1"):
        self.construct_ids = [construct_id]
        self.version = version
        self._items = {i.lower() for i in items}

    @staticmethod
    def _is_mention(t, doc):
        """'Um is a common hesitation marker' — the item fills the subject
        slot of the following root verb (which then has no other subject)."""
        for nxt in doc[t.i + 1:t.sent.end]:
            if nxt.is_space or nxt.is_punct:
                continue
            return (nxt.pos_ in ("AUX", "VERB") and nxt.dep_ == "ROOT"
                    and nxt.tag_ in ("VBZ", "VBP", "VBD")   # finite, not "let"
                    and not any(c.dep_ in ("nsubj", "nsubjpass")
                                for c in nxt.children))
        return False

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        for t in doc:
            if t.lower_ not in self._items:
                continue
            if t.dep_ in self._ARG_DEPS or self._is_mention(t, doc):
                continue                  # mention, not use
            out.append(_mk(self, doc, self.construct_ids[0],
                           t.i, t.i, text_id))
        return out


class MedialFinalParentheticalDetector(Detector):
    """Comma-preceded interactive parenthetical in medial/final position
    (CMT-02): "He was, you know, quite upset." Utterance-initial use is the
    discourse-marker reading (DMG) and integrated use ("You know the answer")
    has no comma, so both are excluded by construction.
    """
    detector_type = "lexicon"

    def __init__(self, nlp, construct_id: str, phrases, version: str = "0.1"):
        self.construct_ids = [construct_id]
        self.version = version
        self._phrases = sorted({tokenize_phrase(nlp, p) for p in phrases},
                               key=len, reverse=True)

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        for sent in doc.sents:
            start = _content_start(doc, sent)
            i = sent.start
            while i < sent.end:
                hit = None
                for toks in self._phrases:
                    j = i + len(toks)
                    if j <= sent.end and tuple(t.lower_ for t in doc[i:j]) == toks:
                        hit = (i, j)
                        break
                if hit is None:
                    i += 1
                    continue
                s, e = hit
                medial_or_final = s > start
                comma_before = s > sent.start and doc[s - 1].text == ","
                delimited_after = (e >= sent.end or doc[e].is_punct
                                   or doc[e].is_space)
                # Written path: comma-delimited. Spoken path (transcripts
                # have no commas): the phrase's verb takes no complement of
                # its own — integrated "you know the answer" has a dobj/
                # ccomp, parenthetical "he was you know quite upset" does
                # not.
                integrated = any(
                    c.dep_ in ("dobj", "ccomp", "xcomp", "attr", "acomp")
                    and not s <= c.i < e
                    for w in doc[s:e] if w.pos_ in ("VERB", "AUX")
                    for c in w.children)
                if medial_or_final and ((comma_before and delimited_after)
                                        or not integrated):
                    out.append(_mk(self, doc, self.construct_ids[0],
                                   s, e - 1, text_id))
                i = e
        return out


class VocativeDetector(Detector):
    """Comma-peripheral vocative NP from a closed lexicon (VOC-02/03).

    A window equal to a lexicon phrase fires iff it is set off by commas at
    an utterance edge (initial + following comma, or preceded by a comma at
    the end, or comma on both sides). A determiner/possessive before the
    window breaks the comma adjacency, so referential uses ("My mum is a
    nurse", "The folks next door") never match.
    """
    detector_type = "lexicon"

    def __init__(self, nlp, construct_id: str, phrases, version: str = "0.1"):
        self.construct_ids = [construct_id]
        self.version = version
        self._phrases = sorted({tokenize_phrase(nlp, p) for p in phrases},
                               key=len, reverse=True)

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        for sent in doc.sents:
            start = _content_start(doc, sent)
            i = sent.start
            while i < sent.end:
                hit = None
                for toks in self._phrases:
                    j = i + len(toks)
                    if j <= sent.end and tuple(t.lower_ for t in doc[i:j]) == toks:
                        hit = (i, j)
                        break
                if hit is None:
                    i += 1
                    continue
                s, e = hit
                initial = s == start and e < sent.end and doc[e].text == ","
                after_comma = s > sent.start and doc[s - 1].text == ","
                final = after_comma and all(t.is_punct for t in doc[e:sent.end])
                medial = after_comma and e < sent.end and doc[e].text == ","
                if initial or final or medial:
                    out.append(_mk(self, doc, self.construct_ids[0],
                                   s, e - 1, text_id))
                i = e
        return out


class FinalTagDetector(Detector):
    """Invariant tag: clause + comma + tag item + "?" ending the utterance
    (QIN-06). Freestanding "Eh?" (INS-04) has no host clause and reversed
    tags ("aren't you?") are not lexicon items, so neither fires.
    """
    detector_type = "lexicon"

    def __init__(self, nlp, construct_id: str, items, version: str = "0.1"):
        self.construct_ids = [construct_id]
        self.version = version
        self._items = sorted({tokenize_phrase(nlp, i) for i in items},
                             key=len, reverse=True)

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        for sent in doc.sents:
            if not ends_with(sent, "?"):
                continue
            # tokens of the tag candidate: between the last comma and the "?"
            tail = [t for t in sent if not t.is_punct]
            for toks in self._items:
                n = len(toks)
                if len(tail) <= n:  # need a host clause before the tag
                    continue
                cand = tail[-n:]
                if tuple(t.lower_ for t in cand) != toks:
                    continue
                if doc[cand[0].i - 1].text != ",":
                    continue
                if any(not t.is_punct for t in doc[cand[-1].i + 1:sent.end]):
                    continue
                out.append(_mk(self, doc, self.construct_ids[0],
                               cand[0].i, cand[-1].i, text_id))
                break
        return out


class TokenSequenceDetector(Detector):
    """Fires on any occurrence of one of the given token sequences (surface
    forms, lower-cased) — for transcribed reduced/vernacular forms whose
    tokenisation is fixed but non-obvious (ai + n't, gon + na, wanna).
    """
    detector_type = "lexicon"

    def __init__(self, construct_id: str, sequences: Sequence[Sequence[str]],
                 version: str = "0.1"):
        self.construct_ids = [construct_id]
        self.version = version
        self._seqs = sorted((tuple(s.lower() for s in seq) for seq in sequences),
                            key=len, reverse=True)

    def match(self, doc, text_id: str = "doc") -> List[Annotation]:
        out = []
        i = 0
        while i < len(doc):
            for seq in self._seqs:
                j = i + len(seq)
                if j <= len(doc) and tuple(t.lower_ for t in doc[i:j]) == seq:
                    out.append(_mk(self, doc, self.construct_ids[0],
                                   i, j - 1, text_id))
                    i = j - 1
                    break
            i += 1
        return out
