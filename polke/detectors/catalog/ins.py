"""Category INS — inserts (freestanding interactional forms), Part VI.

INS-01..08 are freestanding C-units: the whole utterance is the insert
(GSWE 14.3.3). Integrated uses ("I said hi to her", "Damn the torpedoes")
and turn-initial discourse-marker uses ("Oh, I see." -> DMG-02) therefore
never fire — position is the discriminator, not just lexicon membership.
INS-06 hesitators additionally occur punctuation-delimited inside the
utterance. INS-09 is the taboo intensifier in attributive position (amod/
advmod), excluding predicative "his shirt was bloody" (acomp).
"""
from __future__ import annotations
from ...registry import lexicons
from ...schema import Annotation, Span
from ..base import Detector
from ..spoken import DelimitedInsertDetector, FreestandingUnitDetector


def _lx(name, key="entries"):
    return lexicons()[name][key]


class _TabooIntensifier(Detector):
    """INS-09: taboo/expletive item premodifying an adjective or noun."""
    detector_type = "lexicon"
    version = "ins09-taboo-intens@0.1"
    construct_ids = ["INS-09"]

    def __init__(self):
        self._items = set(_lx("taboo", "intensifiers"))
        self._interjections = {i for i in _lx("taboo", "interjections")}

    def match(self, doc, text_id="doc"):
        out = []
        for sent in doc.sents:
            content = " ".join(t.lower_ for t in sent if not t.is_punct)
            if content in self._interjections:
                continue  # freestanding "Bloody hell!" is INS-08, not INS-09
            for t in sent:
                if t.lower_ not in self._items:
                    continue
                if t.dep_ not in ("amod", "advmod", "intj"):
                    continue  # predicative ("was bloody") or other use
                hi = max(t.i, t.head.i) if t.head.sent == t.sent else t.i
                span = doc[t.i:hi + 1]
                out.append(Annotation(
                    text_id=text_id, construct_id="INS-09",
                    span=Span(span.start_char, span.end_char, t.i, hi),
                    detector_type=self.detector_type,
                    detector_version=self.version, confidence=1.0,
                    evidence={"tokens": list(range(t.i, hi + 1)),
                              "matched": span.text}))
        return out


def build(nlp, client=None):
    taboo = list(_lx("taboo", "interjections"))
    moderated = list(_lx("inserts_expletives_moderated"))
    return [
        FreestandingUnitDetector(nlp, "INS-01", _lx("inserts_interjections"),
                                 version="ins01-interj@0.1"),
        FreestandingUnitDetector(nlp, "INS-02", _lx("inserts_greetings"),
                                 version="ins02-greet@0.1"),
        FreestandingUnitDetector(nlp, "INS-03", _lx("inserts_attention"),
                                 version="ins03-attention@0.1"),
        # elicitors end in "?", response tokens do not — same items split by
        # final punctuation ("Right?" INS-04 vs "Right." INS-05)
        FreestandingUnitDetector(nlp, "INS-04", _lx("inserts_response_elicitors"),
                                 question=True, version="ins04-elicitor@0.1"),
        FreestandingUnitDetector(nlp, "INS-05", _lx("inserts_response_tokens"),
                                 question=False, version="ins05-response@0.1"),
        DelimitedInsertDetector(nlp, "INS-06", _lx("inserts_hesitators"),
                                version="ins06-hesitator@0.1"),
        FreestandingUnitDetector(nlp, "INS-07", _lx("inserts_politeness"),
                                 version="ins07-polite@0.1"),
        FreestandingUnitDetector(nlp, "INS-08", moderated + taboo,
                                 version="ins08-expletive@0.1"),
        _TabooIntensifier(),
    ]
