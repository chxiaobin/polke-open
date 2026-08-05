"""Self-contained HTML viewer for annotation output (polke view).

Loads the records written by `polke annotate` — a single `*.annotations.json`
file, a `*.jsonl` file, or a folder scanned recursively for both — aligns each
annotation to the sentence its span falls in, and writes one static HTML page
with client-side filters (record, search, category, tier) so a human can check
the annotations against the text. No external assets, no server needed.

Records written by current polke embed the source text (`"text"`); for older
records the text is re-read from the recorded `"file"` path (tried as-is, then
relative to the annotation file). Sentence boundaries come from the spaCy
pipeline when one is passed in, else from a punctuation/newline heuristic.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__
from .reference import TIER_LABELS
from .llm import LLM_TYPES
from .registry import constructs


def _read_records(path: Path) -> List[Tuple[Path, dict]]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(set(path.rglob("*.annotations.json"))
                       | set(path.rglob("*.jsonl")))
    out: List[Tuple[Path, dict]] = []
    for f in files:
        try:
            if f.suffix == ".jsonl":
                recs = [json.loads(line) for line
                        in f.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
            else:
                recs = [json.loads(f.read_text(encoding="utf-8"))]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {f}: {exc}") from exc
        out.extend((f, r) for r in recs
                   if isinstance(r, dict) and "annotations" in r)
    if not out:
        raise ValueError(
            f"no annotation records under {path} (expected the "
            "*.annotations.json / *.jsonl output of `polke annotate`)")
    return out


def _resolve_text(src: Path, rec: dict) -> Optional[str]:
    """The annotated text: embedded in current records, else re-read from the
    recorded source path (best effort — annotate ran from an unknown cwd)."""
    if isinstance(rec.get("text"), str):
        return rec["text"]
    recorded = rec.get("file")
    if not recorded:
        return None
    p = Path(recorded)
    for cand in (p, src.parent / p, src.parent / p.name,
                 src.parent.parent / p.name):
        try:
            if cand.is_file():
                return cand.read_text(encoding="utf-8")
        except OSError:
            pass
    return None


def _heuristic_sents(text: str) -> List[Tuple[int, int]]:
    """Approximate sentence spans when no spaCy pipeline is available:
    break after end punctuation (plus closing quotes/brackets) or newlines,
    then trim whitespace from each span."""
    breaks = [0]
    for m in re.finditer(r'[.!?…]+["\'\)\]]*\s+|\n+', text):
        breaks.append(m.end())
    breaks.append(len(text))
    spans = []
    for lo, hi in zip(breaks, breaks[1:]):
        chunk = text[lo:hi]
        start = lo + len(chunk) - len(chunk.lstrip())
        end = lo + len(chunk.rstrip())
        if end > start:
            spans.append((start, end))
    return spans


def sentence_spans(text: str, nlp=None) -> List[Tuple[int, int]]:
    if nlp is None:
        return _heuristic_sents(text)
    return [(s.start_char, s.end_char) for s in nlp(text).sents]


def _construct_meta(ids) -> dict:
    from .detectors.base import registry
    from .explain import mechanism
    inv = constructs()
    reg = registry()   # populated only when detectors were built this process
    meta = {}
    for cid in ids:
        c = inv.get(cid)
        if c is None:
            meta[cid] = {"name": "(unknown construct)", "cat": cid.split("-")[0],
                         "catName": "", "tier": "", "example": "",
                         "part": "", "family": "", "note": ""}
            continue
        meta[cid] = {
            "name": c.get("name", ""),
            "cat": c["category"],
            "catName": c.get("category_name", c["category"]),
            "part": c.get("part", ""),
            "family": c.get("family", ""),
            "note": (c.get("use_notes") or "").strip(),
            "tier": TIER_LABELS.get(c.get("detector_type", ""),
                                    c.get("detector_type", "")),
            "needsLlm": c.get("detector_type", "") in LLM_TYPES,
            "example": c.get("example", ""),
        }
        det = reg.get(cid)
        if det is not None:
            meta[cid]["det"] = {
                "how": mechanism(det),
                "cls": type(det).__name__,
                "version": det.version,
                "group": (list(det.construct_ids)
                          if len(det.construct_ids) > 1 else []),
            }
    return meta


def build_payload(path: Path, nlp=None) -> dict:
    records = []
    ids = set()
    for src, rec in _read_records(path):
        text = _resolve_text(src, rec)
        anns = rec.get("annotations") or []
        ids |= {a.get("construct_id", "?") for a in anns}
        records.append({
            "label": rec.get("text_id") or src.stem,
            "source": str(src),
            "file": rec.get("file", ""),
            "spacy_model": rec.get("spacy_model", ""),
            "llm": rec.get("llm"),
            "text": text,
            "sentences": (sentence_spans(text, nlp=nlp)
                          if text is not None else []),
            "annotations": anns,
        })
    records.sort(key=lambda r: r["label"])
    return {"version": __version__, "records": records,
            "constructs": _construct_meta(sorted(ids))}


_CSS = """
:root { color-scheme: light dark;
  --fg: #1a1a1a; --muted: #6b6b6b; --bg: #ffffff; --line: #dcdcdc;
  --chip: #f0f0f0; --accent: #0b6bcb; --mark: #ffe08a; --markfg: #1a1a1a; }
@media (prefers-color-scheme: dark) { :root {
  --fg: #e6e6e6; --muted: #9a9a9a; --bg: #131313; --line: #333;
  --chip: #232323; --accent: #6cb2ff; --mark: #8a6d1a; --markfg: #fff3d0; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 62rem; margin: 0 auto; padding: 1.5rem 1rem 4rem; }
h1 { font-size: 1.35rem; margin: 0 0 .25rem; }
.meta { color: var(--muted); font-size: .85rem; margin: 0 0 1rem; }
.controls { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center;
  position: sticky; top: 0; background: var(--bg); padding: .75rem 0;
  border-bottom: 1px solid var(--line); z-index: 2; }
.controls input[type=search], .controls select { font: inherit; color: inherit;
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  padding: .35rem .5rem; max-width: 100%; }
.controls input[type=search] { flex: 1 1 12rem; min-width: 9rem; }
.controls label { display: flex; gap: .35rem; align-items: center;
  font-size: .85rem; color: var(--muted); white-space: nowrap; }
#count { font-size: .85rem; color: var(--muted); margin-left: auto; }
.recmeta { color: var(--muted); font-size: .8rem; margin: .75rem 0 0; }
.notext { border: 1px solid var(--line); border-radius: 8px; padding: .75rem;
  color: var(--muted); margin-top: 1rem; }
.sent { border: 1px solid var(--line); border-radius: 8px;
  margin-top: 1rem; }
.sent-head { display: flex; gap: .75rem; align-items: baseline;
  padding: .55rem .8rem; }
.sent-no { font: .75rem/1.4 ui-monospace, monospace; color: var(--muted);
  white-space: nowrap; }
.sent-text { flex: 1; }
.sent-n { font-size: .75rem; color: var(--muted); white-space: nowrap; }
ul.anns { list-style: none; margin: 0; padding: 0; }
li.ann { display: flex; gap: .9rem; align-items: baseline;
  padding: .5rem .8rem .5rem 2.2rem; border-top: 1px solid var(--line); }
.cid { font: .8rem/1.4 ui-monospace, monospace; color: var(--accent);
  white-space: nowrap; min-width: 4.6rem; position: relative; cursor: help;
  border-bottom: 1px dotted var(--muted); }
.cid:hover::after { content: attr(data-tip); position: absolute; left: 0;
  top: 1.6em; z-index: 5; background: var(--bg); color: var(--fg);
  border: 1px solid var(--line); border-radius: 6px; padding: .5rem .65rem;
  white-space: pre-line; width: max-content; max-width: 26rem;
  font: .78rem/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  box-shadow: 0 2px 10px rgba(0,0,0,.25); }
.body { flex: 1; }
.name { margin: 0; font-weight: 600; font-size: .88rem; }
.frag { margin: .15rem 0 0; font-size: .9rem; }
mark { background: var(--mark); color: var(--markfg); border-radius: 3px;
  padding: 0 .1rem; }
.sub { margin: .15rem 0 0; color: var(--muted); font-size: .78rem; }
.tier { font-size: .72rem; padding: .1rem .5rem; border-radius: 999px;
  background: var(--chip); color: var(--muted); white-space: nowrap; }
.tier.needs-llm { color: var(--accent); }
.hidden { display: none !important; }
"""

_JS = """
(function () {
  var data = JSON.parse(document.getElementById('data').textContent),
      recSel = document.getElementById('rec'),
      q = document.getElementById('q'),
      catSel = document.getElementById('cat'),
      tierSel = document.getElementById('tier'),
      hideEmpty = document.getElementById('hide-empty'),
      count = document.getElementById('count'),
      main = document.getElementById('main');

  function h(tag, cls, text) {
    var el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  data.records.forEach(function (r, i) {
    var o = document.createElement('option');
    o.value = i; o.textContent = r.label + ' — ' + r.source;
    recSel.appendChild(o);
  });
  if (data.records.length < 2) recSel.classList.add('hidden');

  function meta(cid) {
    return data.constructs[cid] ||
           {name: '', cat: '', catName: '', tier: '', example: ''};
  }

  function rebuildFilters(rec) {
    var cats = {}, tiers = {};
    rec.annotations.forEach(function (a) {
      var m = meta(a.construct_id);
      if (m.cat) cats[m.cat] = m.catName;
      tiers[m.tier || a.detector_type] = 1;
    });
    catSel.innerHTML = '<option value="">All categories</option>';
    Object.keys(cats).sort().forEach(function (c) {
      var o = document.createElement('option');
      o.value = c; o.textContent = c + ' — ' + cats[c]; catSel.appendChild(o);
    });
    tierSel.innerHTML = '<option value="">All tiers</option>';
    Object.keys(tiers).sort().forEach(function (t) {
      var o = document.createElement('option');
      o.value = t; o.textContent = t; tierSel.appendChild(o);
    });
  }

  function fragment(text, s0, s1, a0, a1) {
    var p = h('p', 'frag'), lo = Math.max(a0, s0), hi = Math.min(a1, s1);
    if (hi <= lo) { p.textContent = text.slice(s0, s1); return p; }
    p.appendChild(document.createTextNode(text.slice(s0, lo)));
    p.appendChild(h('mark', null, text.slice(lo, hi)));
    p.appendChild(document.createTextNode(text.slice(hi, s1)));
    return p;
  }

  function annRow(rec, a, s0, s1) {
    var m = meta(a.construct_id),
        li = h('li', 'ann'),
        cid = h('span', 'cid', a.construct_id),
        body = h('div', 'body'),
        name = h('p', 'name', m.name || a.construct_id);
    var tip = [a.construct_id + ' — ' + (m.name || '?')];
    if (m.cat) tip.push('Category: ' + m.cat + ' — ' + m.catName +
                        (m.part ? '  (' + m.part + ')' : ''));
    if (m.family) tip.push('Family: ' + m.family);
    if (m.example) tip.push('e.g. ' + m.example);
    if (m.note) tip.push('Note: ' + m.note);
    if (m.det) {
      tip.push('Detection: ' + (m.tier ? m.tier + ' — ' : '') + m.det.how +
               ' (' + m.det.cls + ' @ ' + m.det.version + ')');
      var g = m.det.group || [];
      if (g.length) tip.push('Detector group: ' + g[0] + ' … ' +
                             g[g.length - 1] + ' (' + g.length +
                             ' siblings, one detector)');
    } else if (m.tier) tip.push('Detector: ' + m.tier);
    cid.setAttribute('data-tip', tip.join('\\n'));
    body.appendChild(name);
    if (rec.text !== null) {
      body.appendChild(fragment(rec.text, s0, s1,
                                a.span.start_char, a.span.end_char));
    } else if (a.evidence && a.evidence.matched) {
      var p = h('p', 'frag'); p.appendChild(h('mark', null, a.evidence.matched));
      body.appendChild(p);
    }
    var sub = [];
    if (a.confidence < 1) sub.push('confidence ' + a.confidence.toFixed(2));
    if (a.model) sub.push(a.model);
    if (a.evidence && a.evidence.rationale) sub.push(a.evidence.rationale);
    if (a.span.end_char > s1 || a.span.start_char < s0)
      sub.push('span continues beyond this sentence');
    if (sub.length) body.appendChild(h('p', 'sub', sub.join(' · ')));
    li.appendChild(cid); li.appendChild(body);
    var tier = h('span', 'tier' + (m.needsLlm ? ' needs-llm' : ''),
                 m.tier || a.detector_type);
    li.appendChild(tier);
    return li;
  }

  function annVisible(a) {
    var m = meta(a.construct_id),
        needle = q.value.trim().toLowerCase(),
        hay = (a.construct_id + ' ' + m.name + ' ' +
               ((a.evidence && a.evidence.matched) || '') + ' ' +
               ((a.evidence && a.evidence.rationale) || '')).toLowerCase();
    return (!catSel.value || m.cat === catSel.value) &&
           (!tierSel.value || (m.tier || a.detector_type) === tierSel.value) &&
           (!needle || hay.indexOf(needle) >= 0);
  }

  function render() {
    var rec = data.records[+recSel.value || 0], shown = 0;
    main.innerHTML = '';
    var rm = h('p', 'recmeta',
               (rec.file ? rec.file + ' · ' : '') +
               rec.annotations.length + ' annotations · spaCy ' +
               (rec.spacy_model || '?') +
               (rec.llm && rec.llm.ready ? ' · LLM ' + rec.llm.model
                                         : ' · offline tiers only'));
    main.appendChild(rm);

    var anns = rec.annotations.filter(annVisible);

    if (rec.text === null) {
      var note = h('div', 'notext',
        'Source text unavailable (older annotation file without an embedded ' +
        '"text" field, and "' + rec.file + '" was not found). Showing ' +
        'matched fragments only.');
      main.appendChild(note);
      var ul = h('ul', 'anns');
      ul.style.marginTop = '1rem';
      anns.forEach(function (a) { ul.appendChild(annRow(rec, a, 0, 0)); });
      shown = anns.length;
      var box = h('div', 'sent'); box.appendChild(ul); main.appendChild(box);
    } else {
      var assigned = new Set();
      rec.sentences.forEach(function (sp, i) {
        var s0 = sp[0], s1 = sp[1];
        var here = anns.filter(function (a, j) {
          if (assigned.has(j)) return false;
          var hit = a.span.start_char < s1 && a.span.end_char > s0;
          if (hit) assigned.add(j);
          return hit;
        });
        if (!here.length && hideEmpty.checked) return;
        var box = h('div', 'sent'),
            head = h('div', 'sent-head');
        head.appendChild(h('span', 'sent-no', 'S' + (i + 1)));
        head.appendChild(h('div', 'sent-text', rec.text.slice(s0, s1)));
        head.appendChild(h('span', 'sent-n', here.length + ' ann.'));
        box.appendChild(head);
        if (here.length) {
          var ul = h('ul', 'anns');
          here.forEach(function (a) { ul.appendChild(annRow(rec, a, s0, s1)); });
          box.appendChild(ul);
        }
        shown += here.length;
        main.appendChild(box);
      });
    }
    count.textContent = shown + ' / ' + rec.annotations.length + ' annotations';
  }

  recSel.addEventListener('change', function () {
    rebuildFilters(data.records[+recSel.value || 0]); render();
  });
  [q, catSel, tierSel, hideEmpty].forEach(function (el) {
    el.addEventListener(el === q ? 'input' : 'change', render);
  });
  rebuildFilters(data.records[0]); render();
})();
"""


def viewer_html(payload: dict) -> str:
    e = html.escape
    n_rec = len(payload["records"])
    n_ann = sum(len(r["annotations"]) for r in payload["records"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>POLKE annotations</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>POLKE — annotation viewer</h1>
<p class="meta">{n_rec} text(s) · {n_ann} annotations · polke {e(payload["version"])}</p>
<div class="controls">
<select id="rec" aria-label="Text"></select>
<input id="q" type="search" placeholder="Search id, name, matched text…"
 aria-label="Search annotations">
<select id="cat" aria-label="Category"></select>
<select id="tier" aria-label="Detector tier"></select>
<label><input type="checkbox" id="hide-empty"> Hide unannotated sentences</label>
<span id="count"></span>
</div>
<div id="main"></div>
<script id="data" type="application/json">{json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")}</script>
<script>{_JS}</script>
</div></body></html>"""


def default_out(src: Path) -> Path:
    if src.is_dir():
        return src / "view.html"
    stem = src.stem
    if stem.endswith(".annotations"):
        stem = stem[: -len(".annotations")]
    return src.with_name(stem + ".view.html")


def write_viewer(src: Path, out: Optional[Path] = None, nlp=None) -> Path:
    payload = build_payload(src, nlp=nlp)
    out = Path(out) if out else default_out(src)
    out.write_text(viewer_html(payload), encoding="utf-8")
    return out
