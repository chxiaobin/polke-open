"""Spoken BNC2014 preparation script + line-based sentence segmentation."""
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "spoken_bnc_prepare",
    Path(__file__).resolve().parents[1] / "scripts" / "spoken_bnc_prepare.py")
sbp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sbp)


def _u(xml):
    return sbp.utterance_text(ET.fromstring(xml))


def test_clean_keeps_unclear_drops_nonlexical():
    assert _u('<u who="S1">oh <unclear>that is nice</unclear> '
              '<vocal desc="laugh"/> <pause dur="short"/>yeah</u>'
              ) == "oh that is nice yeah"


def test_clean_drops_trunc_and_foreign():
    assert _u('<u who="S1"><trunc>wha</trunc> what <foreign lang="jpn">'
              'itadakimasu</foreign> thank you</u>') == "what thank you"


def test_clean_anon_placeholders():
    assert _u('<u who="S1">thanks <anon type="name" nameType="f"/></u>'
              ) == "thanks Sarah"
    # tail with no leading space stays attached (possessive)
    assert _u('<u who="S1">at <anon type="name" nameType="m"/>\'s house</u>'
              ) == "at James's house"
    assert _u('<u who="S1">near <anon type="place"/></u>') == "near Milltown"


def test_clean_empty_utterance():
    assert _u('<u who="S1"><vocal desc="laugh"/></u>') == ""
    assert _u('<u who="S1"><unclear/></u>') == ""


_MINI = """<text id="{tid}">
<header><rec_length>0:01</rec_length></header>
<body>
<u n="1" who="S0001">hello there</u>
<u n="2" who="S0002"><vocal desc="laugh"/></u>
<event desc="door slams"/>
<u n="3" who="S0001">did you see <anon type="name" nameType="n"/> today</u>
</body>
</text>"""


def _mini_corpus(tmp_path, n_texts=3, utts_extra=0):
    src = tmp_path / "dl"
    (src / "spoken" / "untagged").mkdir(parents=True)
    for i in range(n_texts):
        extra = "".join(
            f'<u n="{k + 4}" who="S0001">utterance number {k + 4}</u>\n'
            for k in range(utts_extra))
        xml = _MINI.format(tid=f"T{i:03d}").replace("</body>",
                                                    extra + "</body>")
        (src / "spoken" / "untagged" / f"T{i:03d}.xml").write_text(
            xml, encoding="utf-8")
    return src


def test_convert_writes_text_and_sidecar(tmp_path):
    src = _mini_corpus(tmp_path)
    out = tmp_path / "corpus"
    assert sbp.main(["convert", "--src", str(src), "--out", str(out)]) == 0
    txt = (out / "texts" / "T000.txt").read_text(encoding="utf-8")
    # empty (laughter-only) utterance dropped; anon substituted
    assert txt == "hello there\ndid you see Sam today\n"
    side = json.loads((out / "texts" / "T000.utt.json").read_text("utf-8"))
    assert side["utterances"] == [
        {"line": 1, "n": "1", "who": "S0001"},
        {"line": 2, "n": "3", "who": "S0001"}]


def test_sample_deterministic_windows(tmp_path):
    src = _mini_corpus(tmp_path, n_texts=5, utts_extra=20)
    out = tmp_path / "corpus"
    sbp.main(["convert", "--src", str(src), "--out", str(out)])
    for run in ("s1", "s2"):
        assert sbp.main(["sample", "--corpus", str(out),
                         "--out", str(tmp_path / run), "--chunks", "3",
                         "--utterances", "5", "--seed", "7"]) == 0
    names = sorted(p.name for p in (tmp_path / "s1").glob("*.txt"))
    assert names == sorted(p.name for p in (tmp_path / "s2").glob("*.txt"))
    chunk = (tmp_path / "s1" / names[0])
    lines = chunk.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    side = json.loads(
        chunk.with_name(chunk.stem + ".utt.json").read_text("utf-8"))
    assert [u["line"] for u in side["utterances"]] == [1, 2, 3, 4, 5]
    assert all(u["who"] == "S0001" for u in side["utterances"][1:])


def test_line_segmentation_component():
    spacy = pytest.importorskip("spacy")
    from polke.annotate import _line_senter, load_nlp  # registers nothing yet

    nlp = spacy.blank("en")
    from spacy.language import Language
    if not Language.has_factory("polke_line_senter"):
        Language.component("polke_line_senter", func=_line_senter)
    nlp.add_pipe("polke_line_senter")
    doc = nlp("no punctuation here\nso the parser would merge these\nyeah\n")
    sents = [s.text.strip() for s in doc.sents]
    assert sents == ["no punctuation here",
                     "so the parser would merge these", "yeah"]


def test_load_nlp_line_mode_full_pipeline():
    pytest.importorskip("spacy")
    from polke.annotate import load_nlp
    try:
        nlp = load_nlp(segment="line")
    except RuntimeError:
        pytest.skip("spaCy model not installed")
    assert nlp.has_pipe("polke_line_senter")
    doc = nlp("it was broken\nwas it him\n")
    assert [s.text.strip() for s in doc.sents] == ["it was broken",
                                                   "was it him"]
    # default mode must NOT carry the component
    assert not load_nlp().has_pipe("polke_line_senter")
