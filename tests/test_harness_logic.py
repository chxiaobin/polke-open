"""Offline smoke test of the detect -> annotate -> assert mechanics, WITHOUT
spaCy, using a mock detector over raw strings. This proves the harness wiring
the coding agent reuses for real detectors.
"""
from polke.pipeline import annotate
from polke.schema import Annotation, Span


class MockDetector:
    detector_type = "rule"
    version = "mock@0"

    def __init__(self, construct_id, needle):
        self.construct_ids = [construct_id]
        self.needle = needle

    def match(self, doc, text_id="doc"):
        # here `doc` is the raw string (identity nlp)
        i = doc.lower().find(self.needle.lower())
        if i < 0:
            return []
        return [Annotation(text_id, self.construct_ids[0],
                           Span(i, i + len(self.needle), 0, 0),
                           self.detector_type, self.version)]


def _identity(t):
    return t


def _fires(detector, sentence):
    anns = annotate(sentence, _identity, [detector])
    return any(a.construct_id == detector.construct_ids[0] for a in anns)


def test_positive_fires_negative_does_not():
    det = MockDetector("PAS-01", "is signed")
    assert _fires(det, "The form is signed here.")
    assert not _fires(det, "The door is open.")


def test_annotation_shape():
    det = MockDetector("X-1", "hello")
    anns = annotate("well hello there", _identity, [det])
    assert anns and anns[0].to_dict()["construct_id"] == "X-1"


class MockContextDetector(MockDetector):
    """Fires only when the needle occurs in a preceding utterance."""
    wants_context = True

    def match(self, doc, text_id="doc", context=None):
        if not context or not any(self.needle.lower() in u.lower() for u in context):
            return []
        return [Annotation(text_id, self.construct_ids[0], Span(0, len(doc), 0, 0),
                           self.detector_type, self.version)]


def test_context_reaches_opted_in_detector():
    det = MockContextDetector("QIN-04", "drawer")
    assert _fires(det, "Which drawer?") is False  # no context given
    anns = annotate("Which drawer?", _identity, [det],
                    context=["In the drawer."])
    assert {a.construct_id for a in anns} == {"QIN-04"}


def test_context_ignored_by_plain_detector():
    det = MockDetector("PAS-01", "is signed")
    anns = annotate("The form is signed here.", _identity, [det],
                    context=["Some earlier turn."])
    assert {a.construct_id for a in anns} == {"PAS-01"}
