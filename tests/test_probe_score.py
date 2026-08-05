"""Probe (mocked LLM) and scorer: candidate generation, dedupe, P/R/F1."""
import json

import pytest

from polke.probe import Prober, run
from polke.score import compute, key_of, load_verdicts


TEXT = "The window was broken. She lives here."


def _record():
    return {
        "file": "corpus/a.txt", "text_id": "a",
        "spacy_model": "en_core_web_sm",
        "llm": {"ready": False, "model": None, "reason": "off"},
        "text": TEXT,
        "annotations": [
            {"text_id": "a", "construct_id": "PAS-02",
             "span": {"start_char": 11, "end_char": 21,
                      "token_start": 2, "token_end": 3},
             "detector_type": "rule", "detector_version": "v",
             "confidence": 1.0, "model": None,
             "evidence": {"matched": "was broken"}, "source": "system"},
        ],
    }


class FakeProber(Prober):
    """Claims PAS-02 (already annotated -> deduped), PAS-01 (a new
    candidate), and a bogus id, on every probed sentence."""

    def __init__(self):   # no OpenAI client
        self.model = "fake-model"

    def sentence(self, category, block, text, prev=None):
        return [
            {"construct_id": "PAS-02", "confidence": 0.9, "evidence": text},
            {"construct_id": "PAS-01", "confidence": 0.8, "evidence": text},
            {"construct_id": "XXX-99", "confidence": 1.0, "evidence": "?"},
        ]


@pytest.fixture()
def ann_file(tmp_path):
    f = tmp_path / "a.annotations.json"
    f.write_text(json.dumps(_record()), encoding="utf-8")
    return f


def test_probe_dedupes_and_validates(ann_file, nlp):
    stats = run(ann_file, nlp, FakeProber(), {"PAS-01", "PAS-02"})
    rec = json.loads(ann_file.read_text(encoding="utf-8"))
    cands = rec["probe"]["candidates"]
    assert stats["calls"] == 2                      # 2 sentences x 1 category
    ids = [c["construct_id"] for c in cands]
    assert "XXX-99" not in ids                      # off-menu id dropped
    # PAS-02 deduped only in sentence 1 (where the system annotated it)
    s1 = [c for c in cands if c["span"]["start_char"] == 0]
    assert [c["construct_id"] for c in s1] == ["PAS-01"]
    assert all(c["source"] == "probe" for c in cands)
    assert rec["probe"]["model"] == "fake-model"


def test_probe_rerun_adds_no_duplicates(ann_file, nlp):
    run(ann_file, nlp, FakeProber(), {"PAS-01"})
    n1 = len(json.loads(ann_file.read_text())["probe"]["candidates"])
    run(ann_file, nlp, FakeProber(), {"PAS-01"})
    n2 = len(json.loads(ann_file.read_text())["probe"]["candidates"])
    assert n1 == n2


def test_probe_skips_records_without_text(tmp_path, nlp):
    rec = _record()
    del rec["text"]
    f = tmp_path / "a.annotations.json"
    f.write_text(json.dumps(rec), encoding="utf-8")
    stats = run(f, nlp, FakeProber(), {"PAS-01"})
    assert stats["calls"] == 0 and stats["candidates"] == 0


def test_score_math(tmp_path):
    rec = _record()
    rec["annotations"].append(
        {"text_id": "a", "construct_id": "PRO-01",
         "span": {"start_char": 23, "end_char": 26,
                  "token_start": 5, "token_end": 5},
         "detector_type": "rule", "detector_version": "v", "confidence": 1.0,
         "model": None, "evidence": {}, "source": "system"})
    rec["probe"] = {"model": "m", "candidates": [
        {"text_id": "a", "construct_id": "CLS-06",
         "span": {"start_char": 23, "end_char": 38,
                  "token_start": 5, "token_end": 8},
         "detector_type": "llm_probe", "detector_version": "p",
         "confidence": 0.7, "model": "m", "evidence": {"quoted": "lives"},
         "source": "probe"},
    ]}
    verdicts = {
        key_of("sys", "a", "PAS-02", 11, 21): {"verdict": "tp"},
        key_of("sys", "a", "PRO-01", 23, 26): {"verdict": "wrong",
                                               "corrected_id": "PRO-02"},
        key_of("probe", "a", "CLS-06", 23, 38): {"verdict": "fn"},
        "sys|a|GONE-01|0|1": {"verdict": "tp"},   # stale
    }
    res = compute([rec], verdicts)
    by = {r["construct_id"]: r for r in res["constructs"]}
    assert by["PAS-02"]["tp"] == 1 and by["PAS-02"]["precision"] == 1.0
    assert by["PAS-02"]["recall"] == 1.0 and by["PAS-02"]["support"] == 1
    assert by["PRO-01"]["fp"] == 1 and by["PRO-01"]["precision"] == 0.0
    assert by["PRO-02"]["fn"] == 1 and by["PRO-02"]["recall"] == 0.0
    assert by["PRO-02"]["precision"] is None       # no system hits to judge
    assert by["CLS-06"]["fn"] == 1 and by["CLS-06"]["support"] == 1
    assert res["totals"]["tp"] == 1 and res["totals"]["fp"] == 1
    assert res["totals"]["fn"] == 2
    assert res["stale_verdicts"] == ["sys|a|GONE-01|0|1"]
    assert res["totals"]["pending_sys"] == 0


def test_score_pending_and_span(tmp_path):
    rec = _record()
    verdicts = {}
    res = compute([rec], verdicts)
    assert res["totals"]["pending_sys"] == 1
    verdicts = {key_of("sys", "a", "PAS-02", 11, 21): {"verdict": "span"}}
    res = compute([rec], verdicts)
    by = {r["construct_id"]: r for r in res["constructs"]}
    assert by["PAS-02"]["tp"] == 1 and by["PAS-02"]["span"] == 1


def test_load_verdicts_shapes(tmp_path):
    f = tmp_path / "v.json"
    f.write_text(json.dumps({"verdicts": {"k": {"verdict": "tp"}}}))
    assert load_verdicts(f)["k"]["verdict"] == "tp"
    f.write_text(json.dumps({"k": "fp"}))          # bare mapping also accepted
    assert load_verdicts(f)["k"]["verdict"] == "fp"


def test_backend_routing():
    from polke.probe import _backend_for
    assert _backend_for("claude-opus-5") == "anthropic"
    assert _backend_for("Claude-Haiku-4-5") == "anthropic"
    assert _backend_for("gpt-4o-mini") == "openai"


def test_anthropic_prober_missing_sdk():
    pytest.importorskip("polke")   # guard: only meaningful when anthropic absent
    try:
        import anthropic  # noqa: F401
        pytest.skip("anthropic installed; missing-SDK path not testable")
    except ImportError:
        pass
    from polke.probe import make_prober
    with pytest.raises(RuntimeError, match="anthropic"):
        make_prober("claude-opus-5")


class _FakeBlock:
    def __init__(self, type_, text=""):
        self.type, self.text = type_, text


class _FakeResponse:
    def __init__(self, stop_reason, blocks):
        self.stop_reason, self.content = stop_reason, blocks


def _anthropic_prober_with(resp):
    from polke.probe import AnthropicProber
    p = AnthropicProber.__new__(AnthropicProber)   # skip __init__ (no SDK)
    p.model = "claude-opus-5"

    class _Messages:
        @staticmethod
        def create(**kwargs):
            _Messages.last = kwargs
            return resp

    class _Client:
        messages = _Messages()

    p._client = _Client()
    return p, _Messages


def test_anthropic_sentence_parses_structured_output():
    resp = _FakeResponse("end_turn", [
        _FakeBlock("thinking"),
        _FakeBlock("text", json.dumps({"present": [
            {"construct_id": "PAS-01", "confidence": 0.9,
             "evidence": "was broken"}]})),
    ])
    p, msgs = _anthropic_prober_with(resp)
    out = p.sentence("PAS", "PAS-01: ...", "The window was broken.")
    assert out == [{"construct_id": "PAS-01", "confidence": 0.9,
                    "evidence": "was broken"}]
    assert msgs.last["model"] == "claude-opus-5"
    assert msgs.last["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_sentence_handles_refusal():
    p, _ = _anthropic_prober_with(_FakeResponse("refusal", []))
    assert p.sentence("PAS", "block", "text") == []
