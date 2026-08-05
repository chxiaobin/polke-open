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
