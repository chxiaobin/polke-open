"""Surface / lemma phrase-lexicon detector (spaCy Matcher based).

For closed-class constructs whose trigger is membership of a fixed set of words
or multiword expressions (spatial/temporal prepositions, complex prepositions,
linking adverbials, determiners, quantifiers, pronouns, fixed phrases). Unlike
the DependencyMatcher tiers, this matches by surface form, so it also fires on
the inventory's illustrative *fragment* examples (bare lists like
"above/over, below/under, ...") that have no clean parse.

Longest-match, non-overlapping: a phrase fully covered by a longer accepted
match is suppressed (so "in front of" wins over "in"). An optional ``gate``
callable can require e.g. a real prepositional-object for extra precision.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence, Union
from ..schema import Annotation, Span
from .base import Detector


class PhraseLexiconDetector(Detector):
    detector_type = "lexicon"

    def __init__(self, nlp, construct_ids: Union[str, Sequence[str]],
                 phrases, attr: str = "LOWER", version: str = "0.1",
                 gate: Optional[Callable] = None):
        """phrases: an iterable of phrase strings (single construct id) OR of
        (phrase, construct_id) pairs (routing table). Each phrase is
        whitespace-tokenised; each token becomes a `{attr: token}` pattern dict.
        Use attr="LOWER" for function words, "LEMMA" for inflected content words.
        gate(doc, start, end) -> bool may veto a match."""
        from spacy.matcher import Matcher
        self.construct_ids = [construct_ids] if isinstance(construct_ids, str) else list(construct_ids)
        self.version = version
        self._m = Matcher(nlp.vocab)
        self._route = {}   # match string-id -> (construct_id, n_tokens)
        self._gate = gate
        items = []
        for p in phrases:
            if isinstance(p, (tuple, list)):
                items.append((p[0], p[1]))
            else:
                items.append((p, self.construct_ids[0]))
        for i, (phrase, cid) in enumerate(items):
            toks = str(phrase).split()
            if not toks:
                continue
            pat = [{attr: t.lower() if attr == "LOWER" else t} for t in toks]
            label = f"pl_{i}"
            self._m.add(label, [pat])
            self._route[nlp.vocab.strings[label]] = (cid, len(toks))

    def match(self, doc, text_id: str = "doc"):
        raw = []
        for mid, start, end in self._m(doc):
            cid, _n = self._route[mid]
            if cid not in self.construct_ids:
                continue
            if self._gate and not self._gate(doc, start, end):
                continue
            raw.append((start, end, cid))
        # longest-first greedy, drop matches covered by an accepted longer span
        raw.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        accepted, out, seen = [], [], set()
        for start, end, cid in raw:
            if any(start >= a and end <= b for a, b, _ in accepted):
                continue
            accepted.append((start, end, cid))
            key = (cid, start, end)
            if key in seen:
                continue
            seen.add(key)
            span = doc[start:end]
            out.append(Annotation(
                text_id=text_id, construct_id=cid,
                span=Span(span.start_char, span.end_char, start, end - 1),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": list(range(start, end)), "matched": span.text}))
        return out
