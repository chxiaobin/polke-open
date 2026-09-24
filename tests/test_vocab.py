"""Word-list vocabulary levelling (polke.vocab)."""
import pytest

pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def vp(nlp):
    from polke.vocab import VocabProfile
    return VocabProfile(nlp.tokenizer)


def test_lists_loaded(vp):
    assert vp.counts["cefrj"] > 7000 and vp.counts["octanove"] > 2000
    assert len(vp.multi) > 100                     # multi-word entries


def test_levels_by_lemma_and_pos(vp, nlp):
    r = vp.tag(nlp("The reluctant cats were running quickly."))
    by = {t.text: t for t in r.tokens}
    assert by["The"].level == "A1" and by["cats"].level == "A1"
    assert by["reluctant"].level == "B2"
    assert by["running"].headword == "run" and by["running"].level == "A1"
    assert by["."].kind == "skip"
    assert r.max_level == "B2"


def test_octanove_c_levels(vp, nlp):
    by = {t.text: t for t in vp.tag(nlp("He was avid and timid.")).tokens}
    assert by["avid"].level == "C1" and by["avid"].source == "octanove"


def test_multiword_entries(vp, nlp):
    r = vp.tag(nlp("According to him, the bus stops are far."))
    assert [p.headword for p in r.phrases] == ["according to", "bus stop"]
    assert r.phrases[1].tokens == [5, 6]
    assert all(r.tokens[i].phrase == 1 for i in (5, 6))


def test_unlisted_and_proper(vp, nlp):
    by = {t.text: t for t in vp.tag(nlp("Xylophonists visited Berlin.")).tokens}
    assert by["Berlin"].kind == "proper"
    assert by["Xylophonists"].kind in ("unlisted", "proper")
    assert by["visited"].level == "A1"


def test_lookup_pos_fallback(vp):
    e, ok = vp.lookup("abandon", "VERB")
    assert e.level == "B1" and ok
    e, ok = vp.lookup("abandon", "NOUN")          # only a verb entry exists
    assert e is not None and not ok
    assert vp.lookup("zzzz", "NOUN") == (None, False)
