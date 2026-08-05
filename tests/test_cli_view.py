"""CLI: `annotate --view` builds the viewer alongside the annotations."""
from polke.cli import main


def test_annotate_view_folder(tmp_path, capsys):
    (tmp_path / "a.txt").write_text("The window was broken. She lives here.",
                                    encoding="utf-8")
    rc = main(["annotate", str(tmp_path), "--no-llm", "--view"])
    assert rc == 0
    view = tmp_path / "annotations" / "view.html"
    assert view.is_file()
    page = view.read_text(encoding="utf-8")
    assert "PAS-02" in page
    # detection-mechanism metadata for the hover tooltip (registry was built)
    assert '"how"' in page and "deterministic routing" in page
    assert "viewer written:" in capsys.readouterr().out


def test_annotate_view_jsonl(tmp_path):
    (tmp_path / "a.txt").write_text("The window was broken.", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    rc = main(["annotate", str(tmp_path), "--no-llm",
               "--jsonl", str(out), "--view"])
    assert rc == 0
    assert (tmp_path / "out.view.html").is_file()


def test_annotate_open_implies_view(tmp_path, monkeypatch):
    opened = []
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    (tmp_path / "a.txt").write_text("The window was broken.", encoding="utf-8")
    rc = main(["annotate", str(tmp_path), "--no-llm", "--open"])
    assert rc == 0
    assert (tmp_path / "annotations" / "view.html").is_file()
    assert opened and opened[0].startswith("file://")
