"""CEFR level verifier: construct levels, sentence analysis, CLI."""
import json

import pytest

pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def analyzer(nlp):
    from polke.cefr import LevelAnalyzer
    return LevelAnalyzer(no_llm=True)


def test_every_construct_has_a_level():
    from polke.registry import cefr_levels, constructs
    lv = cefr_levels()
    assert not [cid for cid in constructs() if cid not in lv]
    unrated = [cid for cid, v in lv.items() if v["level"] == "unrated"]
    assert unrated and all(cid.split("-")[0] in ("VER", "DYS") for cid in unrated)


def test_catalog_carries_levels():
    from polke.annotate import catalog
    from polke.registry import CEFR_LEVELS
    rows = {r["id"]: r for r in catalog()}
    assert rows["CON-04"]["cefr_level"] in CEFR_LEVELS   # third conditional
    assert rows["DYS-01"]["cefr_level"] == "unrated"


def test_split_sentences(analyzer):
    s = analyzer.split_sentences("One. Two two.\n\n  three\r\nfour? five!")
    assert s == ["One.", "Two two.", "three", "four?", "five!"]


def test_analyze_iter_records(analyzer):
    recs = list(analyzer.analyze_iter(
        "The bridge was built in 1900.\nCould you pass the salt?"))
    assert recs[0]["type"] == "meta" and recs[0]["sentence_count"] == 2
    assert recs[0]["llm"]["used"] is False
    assert recs[-1]["type"] == "done"
    sents = [r for r in recs if r["type"] == "sentence"]
    assert [r["index"] for r in sents] == [0, 1]
    c = next(c for c in sents[0]["constructions"] if c["id"] == "PAS-02")
    assert c["matched"] == "was built" and c["level"] == "A2"
    assert sents[0]["text"][c["start"]:c["end"]] == "was built"
    assert sents[0]["max_vocab_level"] == "A1"
    assert sents[0]["max_grammar_level"] is not None
    json.dumps(recs)


def test_selection_restricts(analyzer):
    recs = list(analyzer.analyze_iter("The bridge was built in 1900.",
                                      selection=["PAS"]))
    cons = recs[1]["constructions"]
    assert cons and all(c["category"] == "PAS" for c in cons)


def test_summary_counts_phrases_once(analyzer):
    from polke.cefr import summarize
    recs = [r for r in analyzer.analyze_iter("I waited at the bus stop.")
            if r["type"] == "sentence"]
    s = summarize(recs)
    assert s["words"] == 5                         # I waited at the [bus stop]
    assert sum(s["vocab_levels"].values()) == 5


def test_cli_level(tmp_path, capsys):
    from polke.cli import main
    f = tmp_path / "t.txt"
    f.write_text("It was built in 1900.\nGo away!\n", encoding="utf-8")
    assert main(["level", str(f), "--no-llm"]) == 0
    out = capsys.readouterr().out
    assert "It was built in 1900." in out and "Go away!" in out
    assert "2 sentences" in out
    assert main(["level", str(f), "--no-llm", "--json"]) == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l]
    assert [l["type"] for l in lines] == ["meta", "sentence", "sentence", "done"]
