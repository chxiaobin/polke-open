import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "polke" / "data"


def test_every_construct_has_a_detector_entry():
    con = json.loads((DATA / "constructs.json").read_text(encoding="utf-8"))["constructs"]
    det = json.loads((DATA / "detectors.json").read_text(encoding="utf-8"))["detectors"]
    cids = {c["id"] for c in con}
    dids = {d["construct_id"] for d in det}
    assert cids == dids, "missing=%s extra=%s" % (cids - dids, dids - cids)


def test_contract_has_positive_test_cases(detector_contract):
    missing = [d["construct_id"] for d in detector_contract
               if not d["test_cases"]["positive"]]
    # a few constructs carry illustrative (non-sentential) examples; keep loose
    assert len(missing) <= 6, missing
