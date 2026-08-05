"""CLI: catalog listing, --full details, -c selection."""
import json

import pytest

from polke.cli import main


def test_catalog_brief(capsys):
    assert main(["catalog"]) == 0
    out = capsys.readouterr().out
    assert "PAS — Voice (passives)" in out
    assert "PAS-01" in out
    assert "e.g." not in out


def test_catalog_full_shows_details(capsys):
    assert main(["catalog", "--full", "-c", "VTA-03"]) == 0
    out = capsys.readouterr().out
    assert "VTA-03" in out and "[rule+LLM]" in out
    assert "family : Present simple" in out
    assert "e.g.   : I get up at seven." in out
    assert "note   : often with frequency adverbs." in out
    assert "PAS-01" not in out          # selection respected


def test_catalog_selection_mixes_ids_and_categories(capsys):
    assert main(["catalog", "-c", "PAS,REL-01"]) == 0
    out = capsys.readouterr().out
    assert "PAS-01" in out and "REL-01" in out
    assert "REL-02" not in out


def test_catalog_json_respects_selection(capsys):
    assert main(["catalog", "--json", "-c", "PAS"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 20
    assert all(r["category"] == "PAS" for r in rows)


def test_catalog_unknown_selection_errors(capsys):
    assert main(["catalog", "-c", "NOPE"]) == 2
    assert "no constructs match" in capsys.readouterr().err
