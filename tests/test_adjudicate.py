"""Adjudication server: page serving + verdict persistence round-trip."""
import json
import threading
import urllib.request

import pytest

from polke.adjudicate import make_server
from polke.viewer import _JS


@pytest.fixture()
def server(tmp_path):
    vf = tmp_path / "verdicts.json"
    srv = make_server("<html>PAGE</html>", vf, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", vf
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read().decode("utf-8")


def _put(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 method="PUT",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.status


def test_serves_page_and_empty_store(server):
    base, _vf = server
    assert _get(base + "/") == (200, "<html>PAGE</html>")
    status, body = _get(base + "/verdicts.json")
    assert status == 200 and json.loads(body) == {"verdicts": {}}


def test_put_roundtrip_and_atomic_file(server):
    base, vf = server
    store = {"sys|a|PAS-02|11|21": {"verdict": "tp", "corrected_id": None}}
    assert _put(base + "/verdicts.json", {"verdicts": store}) == 200
    on_disk = json.loads(vf.read_text(encoding="utf-8"))
    assert on_disk["verdicts"] == store
    status, body = _get(base + "/verdicts.json")
    assert json.loads(body)["verdicts"] == store
    # the stored file is what `polke score` expects
    from polke.score import load_verdicts
    assert load_verdicts(vf)["sys|a|PAS-02|11|21"]["verdict"] == "tp"


def test_put_rejects_bad_body(server):
    base, vf = server
    req = urllib.request.Request(base + "/verdicts.json", data=b"not json",
                                 method="PUT")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 400
    assert not vf.exists()


def test_unknown_path_404(server):
    base, _vf = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/etc/passwd")
    assert exc.value.code == 404


def test_viewer_js_has_sync():
    assert "pushStore" in _JS and "verdicts.json" in _JS
    assert "PUT" in _JS   # the sync writes back to the server
