"""Reference detectors validated against their contract. Auto-skips if spaCy or
the model is not installed (see the nlp fixture in conftest.py).
"""
import pytest
from polke.pipeline import annotate
from polke.detectors.reference import build_reference_detectors


@pytest.fixture(scope="module")
def refs(nlp):
    return build_reference_detectors(nlp)  # DummyClient for the hybrid tier


def _ids(anns):
    return {a.construct_id for a in anns}


def test_pas01_positive(nlp, refs):
    assert "PAS-01" in _ids(annotate("The form is signed by the manager.", nlp, refs))


def test_pas01_excludes_get_passive(nlp, refs):
    assert "PAS-01" not in _ids(annotate("He got promoted last year.", nlp, refs))


def test_prep18_positive(nlp, refs):
    assert "PREP-18" in _ids(annotate("It depends on the weather.", nlp, refs))


def test_prep18_negative_pair_not_in_lexicon(nlp, refs):
    assert "PREP-18" not in _ids(annotate("She sat on the chair.", nlp, refs))


def test_pp_hybrid_fires_on_present_perfect(nlp, refs):
    ids = _ids(annotate("I have visited Rome twice.", nlp, refs))
    assert ids & {"VTA-29", "VTA-30", "VTA-31", "VTA-32", "VTA-33", "VTA-34"}
