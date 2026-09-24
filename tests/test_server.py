"""HTTP API smoke tests. Startup loads spaCy + builds every detector, so the
TestClient context is entered once per module."""
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def client():
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("run: python -m spacy download en_core_web_sm")
    from fastapi.testclient import TestClient
    from polke.server import app
    with TestClient(app) as c:  # enters lifespan: spaCy + detector build
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "llm" in body


def test_catalog_is_complete_and_enriched(client):
    rows = client.get("/catalog").json()["constructions"]
    assert len(rows) == 670
    row = {r["id"]: r for r in rows}["VTA-01"]
    assert row["needs_llm"] is True
    assert "definition" in row and row["definition"]
    assert "use_notes" in row


def test_reference_page(client):
    resp = client.get("/reference")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    page = resp.text
    assert 'id="PAS-01"' in page          # stable anchors
    assert 'id="DYS-04"' in page          # last construct present too
    assert "cats-by-part" in page         # filter data embedded


def test_annotate_rule_tier(client):
    resp = client.post("/annotate", json={
        "text": "The window was broken by the storm.",
        "constructions": ["PAS-02"],
    })
    ids = [a["construct_id"] for a in resp.json()["annotations"]]
    assert ids == ["PAS-02"]


# ---- CEFR level verifier ---------------------------------------------------

def test_root_redirects_to_verify_ui(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307) and resp.headers["location"] == "/verify"
    page = client.get("/verify")
    assert page.status_code == 200 and "level verifier" in page.text
    assert client.get("/static/verify/app.js").status_code == 200


def test_health_reports_word_lists(client):
    body = client.get("/health").json()
    assert body["vocab_lists"]["cefrj"] > 7000


def test_catalog_has_cefr_levels(client):
    rows = client.get("/catalog").json()["constructions"]
    assert all("cefr_level" in r for r in rows)


def test_reference_shows_levels(client):
    page = client.get("/reference").text
    assert 'data-level="' in page and 'id="level"' in page


def test_verify_vocab(client):
    body = client.get("/verify/vocab", params={"q": "abandon"}).json()
    assert body["entries"][0]["level"] == "B1"


def test_verify_analyze_stream(client):
    import json
    with client.stream("POST", "/verify/analyze",
                       json={"text": "It was built in 1900.\nGo away!",
                             "use_llm": False}) as r:
        assert r.status_code == 200
        lines = [json.loads(l) for l in r.iter_lines() if l.strip()]
    assert lines[0]["type"] == "meta" and lines[-1]["type"] == "done"
    assert [l["type"] for l in lines[1:-1]] == ["sentence", "sentence"]
    assert lines[1]["vocab"]["tokens"] and lines[1]["max_vocab_level"]


def test_verify_analyze_json_and_empty(client):
    body = client.post("/verify/analyze/json",
                       json={"text": "She has lived here for years.",
                             "use_llm": False}).json()
    assert body["summary"]["sentences"] == 1
    assert client.post("/verify/analyze", json={"text": " "}).status_code == 400
