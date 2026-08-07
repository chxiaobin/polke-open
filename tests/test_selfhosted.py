"""Self-hosted model support: models.list fallback + no-think switch."""
import pytest

from polke.llm import model_available


class _M:
    def __init__(self, id_):
        self.id = id_


class _Models:
    def __init__(self, served, retrieve_ok=False):
        self._served, self._retrieve_ok = served, retrieve_ok

    def retrieve(self, model):
        if not self._retrieve_ok:
            raise RuntimeError("404 Not Found")

    def list(self):
        if self._served is None:
            raise RuntimeError("connection refused")
        return [_M(s) for s in self._served]


class _Client:
    def __init__(self, served, retrieve_ok=False):
        self.models = _Models(served, retrieve_ok)


def test_model_available_via_retrieve():
    assert model_available(_Client([], retrieve_ok=True), "m") is None


def test_model_available_via_list_fallback():
    assert model_available(_Client(["a/b", "m"]), "m") is None


def test_model_not_served():
    reason = model_available(_Client(["a/b"]), "m")
    assert "not served" in reason and "a/b" in reason


def test_server_unreachable():
    assert "not reachable" in model_available(_Client(None), "m")


class _Completions:
    def __init__(self):
        self.last = None

    def create(self, **kw):
        self.last = kw

        class _Msg:
            content = '{"construct_id": "NONE", "confidence": 0}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _ChatClient:
    def __init__(self):
        self.chat = type("C", (), {"completions": _Completions()})()


def _classifier():
    from polke.detectors.openai_client import OpenAIClassifier
    c = OpenAIClassifier.__new__(OpenAIClassifier)
    c._client, c.model, c._cache = _ChatClient(), "m", {}
    import threading
    c._lock = threading.Lock()
    return c


def test_classifier_no_think_off_by_default(monkeypatch):
    monkeypatch.delenv("POLKE_LLM_NO_THINK", raising=False)
    c = _classifier()
    c.classify("s", "u", ["X"])
    assert "extra_body" not in c._client.chat.completions.last


def test_classifier_no_think_enabled(monkeypatch):
    monkeypatch.setenv("POLKE_LLM_NO_THINK", "1")
    c = _classifier()
    c.classify("s", "u", ["X"])
    extra = c._client.chat.completions.last["extra_body"]
    assert extra == {"chat_template_kwargs": {"enable_thinking": False}}


def test_prober_no_think_enabled(monkeypatch):
    monkeypatch.setenv("POLKE_LLM_NO_THINK", "1")
    from polke.probe import Prober
    p = Prober.__new__(Prober)
    p.model, p._client = "m", _ChatClient()
    p._client.chat.completions.last = None

    class _Msg:
        content = '{"present": []}'
    p.sentence("PAS", "block", "text")
    kw = p._client.chat.completions.last
    assert kw["extra_body"] == {"chat_template_kwargs":
                                {"enable_thinking": False}}
