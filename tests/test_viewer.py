"""Viewer: record loading, sentence alignment, HTML generation (offline)."""
import json

import pytest

from polke.viewer import (_heuristic_sents, build_payload, default_out,
                          viewer_html, write_viewer)


def _record(text="The window was broken. She lives here.", with_text=True):
    rec = {
        "file": "corpus/essay1.txt",
        "text_id": "essay1",
        "spacy_model": "en_core_web_sm",
        "llm": {"ready": False, "model": None, "reason": "off"},
        "annotations": [
            {"text_id": "essay1", "construct_id": "PAS-02",
             "span": {"start_char": 11, "end_char": 21,
                      "token_start": 2, "token_end": 3},
             "detector_type": "rule", "detector_version": "v", "confidence": 1.0,
             "model": None, "evidence": {"matched": "was broken"},
             "source": "system"},
            {"text_id": "essay1", "construct_id": "PRO-01",
             "span": {"start_char": 23, "end_char": 26,
                      "token_start": 5, "token_end": 5},
             "detector_type": "rule", "detector_version": "v", "confidence": 1.0,
             "model": None, "evidence": {"matched": "She"}, "source": "system"},
        ],
    }
    if with_text:
        rec["text"] = text
    return rec


def test_heuristic_sents_offsets_cover_text():
    text = "One sentence here. Another one!\nA third (no punct)"
    spans = _heuristic_sents(text)
    assert [text[a:b] for a, b in spans] == [
        "One sentence here.", "Another one!", "A third (no punct)"]


def test_build_payload_from_json_file(tmp_path):
    f = tmp_path / "essay1.annotations.json"
    f.write_text(json.dumps(_record()), encoding="utf-8")
    payload = build_payload(f)
    (rec,) = payload["records"]
    assert rec["text"].startswith("The window")
    assert len(rec["sentences"]) == 2
    # both annotations fall inside the first/second sentence respectively
    (s0, s1) = rec["sentences"]
    assert s0[0] <= 11 < 21 <= s0[1]
    assert s1[0] <= 23 < 26 <= s1[1]
    assert "PAS-02" in payload["constructs"]
    meta = payload["constructs"]["PAS-02"]
    assert meta["cat"] == "PAS"
    # detail fields feeding the hover tooltip
    assert meta["name"] == "Past simple passive"
    assert meta["family"] == "Passive across the paradigm"
    assert meta["part"] == "I. Verb Phrase"
    assert meta["example"]
    assert meta["tier"] == "rule"
    assert "note" in meta


def test_build_payload_from_jsonl_folder(tmp_path):
    (tmp_path / "out.jsonl").write_text(
        json.dumps(_record()) + "\n" + json.dumps(_record()) + "\n",
        encoding="utf-8")
    payload = build_payload(tmp_path)
    assert len(payload["records"]) == 2


def test_text_fallback_reads_recorded_file(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "essay1.txt").write_text("The window was broken. She lives here.",
                                       encoding="utf-8")
    ann_dir = corpus / "annotations"
    ann_dir.mkdir()
    rec = _record(with_text=False)
    rec["file"] = str(corpus / "essay1.txt")
    (ann_dir / "essay1.annotations.json").write_text(json.dumps(rec),
                                                    encoding="utf-8")
    payload = build_payload(ann_dir)
    assert payload["records"][0]["text"].startswith("The window")


def test_missing_text_degrades_to_null(tmp_path):
    rec = _record(with_text=False)
    rec["file"] = "does/not/exist.txt"
    f = tmp_path / "x.annotations.json"
    f.write_text(json.dumps(rec), encoding="utf-8")
    payload = build_payload(f)
    assert payload["records"][0]["text"] is None
    assert payload["records"][0]["sentences"] == []


def test_write_viewer_html(tmp_path):
    f = tmp_path / "essay1.annotations.json"
    f.write_text(json.dumps(_record()), encoding="utf-8")
    out = write_viewer(f)
    assert out == tmp_path / "essay1.view.html"
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "PAS-02" in page and "annotation viewer" in page
    # embedded payload must not be able to close the script tag early
    assert "</script>" not in json.dumps(
        build_payload(f), ensure_ascii=False).replace("</", "<\\/")


def test_no_records_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        build_payload(tmp_path)


def test_default_out_for_folder(tmp_path):
    assert default_out(tmp_path) == tmp_path / "view.html"


def test_viewer_html_escapes_text():
    payload = {"version": "x", "records": [], "constructs": {}}
    assert "<script>" in viewer_html(payload)
