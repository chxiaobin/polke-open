"""Category COH - cohesion & linking adverbials.

Linking adverbials are a closed class -> one PhraseLexiconDetector with a
(phrase -> construct_id) routing table serves COH-01..11 and COH-13 at once,
which is precise because each linker belongs to exactly one semantic group and
the negatives are sibling groups. COH-12 (anaphoric reference) is the LLM tier.
COH-14 (punctuation of connectors) is a degenerate contract row - see NOTES.
"""
from __future__ import annotations
from ..lexical import PhraseLexiconDetector
from ..llm import LLMStandaloneDetector
from ...schema import Annotation, Span

# phrase -> COH id. Extended well beyond the seed examples with the standard
# Ambiguous single-word linkers: they are connectives only when they OPEN the
# utterance ("So we left." / "Then she rang.") — degree 'so good', temporal
# 'back then', phase 'not yet', copular 'that is', intensifier 'well' and
# response 'right' all occur medially and are different constructs.
_INITIAL_ONLY = {
    "so": "COH-03", "then": "COH-04", "yet": "COH-02",
    "well": "COH-11", "right": "COH-11", "certainly": "COH-09",
}

# linking-adverbial inventory (Halliday/Biber), still one group per item.
_LINKERS = {
    # COH-01 additive
    "moreover": "COH-01", "furthermore": "COH-01", "in addition": "COH-01",
    "additionally": "COH-01", "besides": "COH-01", "what's more": "COH-01",
    "likewise": "COH-01", "similarly": "COH-01", "also": "COH-01",
    "as well": "COH-01", "on top of that": "COH-01", "not only that": "COH-01",
    # COH-02 adversative / contrastive
    "however": "COH-02", "nevertheless": "COH-02", "nonetheless": "COH-02",
    "on the other hand": "COH-02", "in contrast": "COH-02", "by contrast": "COH-02",
    "conversely": "COH-02", "even so": "COH-02", "instead": "COH-02",
    "on the contrary": "COH-02", "then again": "COH-02", "all the same": "COH-02",
    "whereas": "COH-02",
    # COH-03 causal / resultative
    "therefore": "COH-03", "thus": "COH-03", "hence": "COH-03",
    "consequently": "COH-03", "as a result": "COH-03", "accordingly": "COH-03",
    "for this reason": "COH-03", "as a consequence": "COH-03",
    # COH-04 temporal / sequencing
    "first": "COH-04", "firstly": "COH-04", "second": "COH-04",
    "secondly": "COH-04", "next": "COH-04",
    "afterwards": "COH-04", "subsequently": "COH-04", "finally": "COH-04",
    "meanwhile": "COH-04", "eventually": "COH-04", "to begin with": "COH-04",
    "in the end": "COH-04", "lastly": "COH-04", "beforehand": "COH-04",
    "at the same time": "COH-04", "later": "COH-04",
    # COH-05 exemplifying
    "for example": "COH-05", "for instance": "COH-05", "such as": "COH-05",
    "namely": "COH-05", "in particular": "COH-05", "to illustrate": "COH-05",
    "as an illustration": "COH-05", "including": "COH-05",
    # COH-06 reformulating / clarifying
    "in other words": "COH-06", "that is to say": "COH-06",
    "or rather": "COH-06", "to put it another way": "COH-06", "i.e.": "COH-06",
    "put differently": "COH-06",
    # COH-07 summarising / concluding
    "in conclusion": "COH-07", "to sum up": "COH-07", "in short": "COH-07",
    "overall": "COH-07", "on the whole": "COH-07", "all in all": "COH-07",
    "to conclude": "COH-07", "in summary": "COH-07", "in brief": "COH-07",
    # COH-08 emphasising
    "indeed": "COH-08", "in fact": "COH-08", "as a matter of fact": "COH-08",
    "above all": "COH-08", "importantly": "COH-08", "clearly": "COH-08",
    "notably": "COH-08", "significantly": "COH-08",
    # COH-09 conceding
    "admittedly": "COH-09", "of course": "COH-09", "naturally": "COH-09",
    "granted": "COH-09", "it is true that": "COH-09",
    "to be sure": "COH-09", "no doubt": "COH-09",
    # COH-10 conditional / consequence
    "otherwise": "COH-10", "in that case": "COH-10", "if so": "COH-10",
    "if not": "COH-10", "under the circumstances": "COH-10",
    "in which case": "COH-10",
    # COH-11 discourse markers (spoken / informal)
    "anyway": "COH-11", "by the way": "COH-11",
    "you know": "COH-11", "i mean": "COH-11",
    "actually": "COH-11",
    # COH-13 the former/latter/above/following/aforementioned
    "the former": "COH-13", "the latter": "COH-13", "the above": "COH-13",
    "the following": "COH-13", "the aforementioned": "COH-13",
    "aforementioned": "COH-13",
}

_COH_IDS = sorted(set(_LINKERS.values()))
_CONNECTOR_WORDS = {p for p in _LINKERS}


class _ConnectorPunctuation:
    """COH-14 best-effort: a linking adverbial set off by a comma in running
    text, NOT one item of a metalinguistic enumeration of connectors. Flagged in
    NOTES as a degenerate contract row (positive 'however, ...' vs the negatives
    which are comma-separated *lists* of connectors)."""
    construct_ids = ["COH-14"]
    detector_type = "rule"
    version = "coh14-punct@0.1"

    @staticmethod
    def _starts_connector(doc, i):
        for n in (3, 2, 1):
            if i + n <= len(doc):
                phrase = " ".join(doc[i:i + n].text.lower().split())
                if phrase in _LINKERS:
                    return True
        return False

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.i + 1 >= len(doc):
                continue
            nxt = doc[t.i + 1]
            if nxt.text != ",":
                continue
            if t.lower_ not in ("however", "therefore", "moreover",
                                "nevertheless", "furthermore", "consequently",
                                "thus", "hence", "meanwhile", "instead"):
                continue
            # skip an item that is part of a metalinguistic list of connectors
            if self._starts_connector(doc, t.i + 2):
                continue
            out.append(Annotation(
                text_id=text_id, construct_id="COH-14",
                span=Span(t.idx, nxt.idx + 1, t.i, nxt.i),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0, evidence={"tokens": [t.i, nxt.i], "matched": t.text + ","}))
        return out


def _initial_gate(doc, start, end):
    """Ambiguous single-word linkers must be the first content token of their
    sentence AND the sentence must open the utterance (line); unambiguous
    phrases pass through."""
    word = " ".join(doc[start:end].text.lower().split())
    if word not in _INITIAL_ONLY:
        return True
    sent = doc[start].sent
    for t in sent:
        if t.is_punct or t.is_space:
            continue
        if t.i != start:
            return False
        break
    j = sent.start - 1
    while j >= 0 and (doc[j].is_space or doc[j].is_punct):
        if "\n" in doc[j].text:
            return True
        j -= 1
    return j < 0


def build(nlp, client=None):
    dets = []
    phrases = [(p, cid) for p, cid in _LINKERS.items()] + \
              [(p, cid) for p, cid in _INITIAL_ONLY.items()]
    dets.append(PhraseLexiconDetector(nlp, _COH_IDS, phrases,
                                      gate=_initial_gate,
                                      version="coh-linkers@0.2"))
    dets.append(_ConnectorPunctuation())
    # COH-12 anaphoric reference (pronominal/demonstrative/the-): LLM tier.
    dets.append(LLMStandaloneDetector(
        "COH-12",
        'Decide whether the sentence uses anaphoric reference for cohesion '
        '(a pronoun, demonstrative "this/that/these/those/such", or a definite '
        'NP) pointing back to something earlier in the discourse. Return JSON '
        '{"construct_id":"COH-12"|"NONE","confidence":0..1,"rationale":"..."}.',
        client=client, version="coh12-llm@0.1"))
    return dets
