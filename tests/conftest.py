import json
import pytest
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "polke" / "data"


@pytest.fixture(scope="session")
def detector_contract():
    return json.loads((DATA / "detectors.json").read_text(encoding="utf-8"))["detectors"]


@pytest.fixture(scope="session")
def nlp():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("run: python -m spacy download en_core_web_sm")
