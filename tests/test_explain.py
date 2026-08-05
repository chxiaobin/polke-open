"""CLI: `polke explain` — detection-mechanism introspection."""
import json

from polke.cli import main


def test_explain_rule_router(capsys):
    assert main(["explain", "PAS-01"]) == 0
    out = capsys.readouterr().out
    assert "RuleRoutingDetector @ pas-paradigm@0.1" in out
    assert "deterministic routing" in out
    assert "PAS-01…PAS-10" in out
    assert '"RIGHT_ATTRS": {"TAG": "VBN"}' in out       # dependency pattern
    assert "+ The form is signed here." in out          # contract cases


def test_explain_llm_hybrid_shows_prompt(capsys):
    assert main(["explain", "VTA-29"]) == 0
    out = capsys.readouterr().out
    assert "LLMReadingDetector" in out
    assert "LLM picks one reading" in out
    assert "| You label the USE of an English present perfect" in out


def test_explain_lexicon_shows_entries(capsys):
    assert main(["explain", "ADV-06"]) == 0
    out = capsys.readouterr().out
    assert "PhraseLexiconDetector" in out
    assert "entries:" in out and "abroad" in out


def test_explain_json(capsys):
    assert main(["explain", "PAS-01", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["PAS-01"]["class"] == "RuleRoutingDetector"
    assert data["PAS-01"]["patterns"]


def test_explain_unknown_errors(capsys):
    assert main(["explain", "NOPE"]) == 2
    assert "no constructs match" in capsys.readouterr().err
