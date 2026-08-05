"""The contract as a burn-down suite: one test per construct in
data/detectors.json. A construct's test SKIPS until a detector is registered for
it (via build_all), then asserts its positive cases fire and its negative cases
do not. LLM/hybrid tiers only run when a real client is configured
(POLKE_LLM=1), since DummyClient cannot judge readings.
"""
import os
import pytest
from polke.registry import detectors as _contract
from polke.pipeline import annotate

CONTRACT = _contract()
LLM_TYPES = {"llm", "hybrid_rule_llm", "hybrid_lexicon_llm"}


def _case(case):
    """A test case is either a plain string or, for discourse-level constructs,
    {"text": ..., "context": [preceding utterances, oldest first]}."""
    if isinstance(case, dict):
        return case["text"], case.get("context")
    return case, None


def _load_root_env():
    """Best-effort OPENAI_API_KEY from the project root .env."""
    from polke.env import load_env
    load_env()


@pytest.fixture(scope="module")
def built(nlp):
    from polke.build import build_all
    client = None
    if os.getenv("POLKE_LLM") == "1":
        _load_root_env()
        from polke.detectors.openai_client import OpenAIClassifier
        client = OpenAIClassifier()  # real client replaces DummyClient
    build_all(nlp, llm_client=client)
    from polke.detectors.base import implemented_ids
    return implemented_ids()


@pytest.mark.parametrize("cid", sorted(CONTRACT.keys()))
def test_contract(cid, nlp, built):
    spec = CONTRACT[cid]
    if spec["detector_type"] in LLM_TYPES and os.getenv("POLKE_LLM") != "1":
        pytest.skip("LLM tier: set POLKE_LLM=1 with a real client to test %s" % cid)
    if cid not in built:
        pytest.skip("not implemented yet: %s" % cid)
    from polke.detectors.base import registry
    det = registry()[cid]
    for s in spec["test_cases"]["positive"]:
        text, context = _case(s)
        ids = {a.construct_id for a in annotate(text, nlp, [det], context=context)}
        assert cid in ids, "positive missed (%s): %r" % (cid, s)
    for s in spec["test_cases"]["negative"]:
        text, context = _case(s)
        ids = {a.construct_id for a in annotate(text, nlp, [det], context=context)}
        assert cid not in ids, "negative fired (%s): %r" % (cid, s)
