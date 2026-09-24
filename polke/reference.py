"""Self-contained HTML reference for the construct inventory (GET /reference).

One static page, rendered once per process from data/constructs.json: every
construction grouped Part -> Category, with client-side search, part/category
filters, an LLM-tier toggle, and a stable anchor per construction so
`/reference#PAS-01` is a shareable link. No external assets; works with JS
disabled (filters simply inactive).
"""
from __future__ import annotations

import functools
import html
import json

from . import __version__
from .llm import LLM_TYPES
from .registry import CEFR_LEVELS, cefr_level, constructs

TIER_LABELS = {
    "rule": "rule",
    "lexicon": "lexicon",
    "hybrid_rule_llm": "rule+LLM",
    "hybrid_lexicon_llm": "lexicon+LLM",
    "llm": "LLM",
}

_CSS = """
:root { color-scheme: light dark;
  --fg: #1a1a1a; --muted: #6b6b6b; --bg: #ffffff; --line: #dcdcdc;
  --chip: #f0f0f0; --accent: #0b6bcb; --ok: #1a7f37; }
@media (prefers-color-scheme: dark) { :root {
  --fg: #e6e6e6; --muted: #9a9a9a; --bg: #131313; --line: #333;
  --chip: #232323; --accent: #6cb2ff; --ok: #4ac26b; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 60rem; margin: 0 auto; padding: 1.5rem 1rem 4rem; }
h1 { font-size: 1.35rem; margin: 0 0 .25rem; }
.meta { color: var(--muted); font-size: .85rem; margin: 0 0 1.25rem; }
.meta a { color: var(--accent); }
.controls { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center;
  position: sticky; top: 0; background: var(--bg); padding: .75rem 0;
  border-bottom: 1px solid var(--line); z-index: 2; }
.controls input[type=search], .controls select { font: inherit; color: inherit;
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  padding: .35rem .5rem; }
.controls input[type=search] { flex: 1 1 14rem; min-width: 10rem; }
.controls label { display: flex; gap: .35rem; align-items: center;
  font-size: .85rem; color: var(--muted); white-space: nowrap; }
#count { font-size: .85rem; color: var(--muted); margin-left: auto; }
h2 { font-size: 1.05rem; margin: 2rem 0 .5rem; }
h3 { font-size: .9rem; color: var(--muted); font-weight: 600;
  margin: 1.25rem 0 .4rem; }
ul.rows { list-style: none; margin: 0; padding: 0;
  border: 1px solid var(--line); border-radius: 8px; }
li.row { display: flex; gap: .9rem; align-items: baseline;
  padding: .55rem .8rem; border-top: 1px solid var(--line); }
li.row:first-child { border-top: 0; }
li.row:target { outline: 2px solid var(--accent); outline-offset: -2px;
  border-radius: 6px; }
a.cid { font: .8rem/1.4 ui-monospace, monospace; color: var(--accent);
  text-decoration: none; white-space: nowrap; min-width: 4.6rem; }
a.cid:hover { text-decoration: underline; }
.body { flex: 1; }
.name { margin: 0; font-weight: 600; font-size: .92rem; }
.ex { margin: .1rem 0 0; font-style: italic; color: var(--muted);
  font-size: .85rem; }
.sub { margin: .1rem 0 0; color: var(--muted); font-size: .78rem; }
.tier { font-size: .72rem; padding: .1rem .5rem; border-radius: 999px;
  background: var(--chip); color: var(--muted); white-space: nowrap; }
.tier.needs-llm { color: var(--accent); }
.cefr { font-size: .72rem; font-weight: 600; padding: .1rem .45rem; border-radius: 4px;
  color: #fff; background: #9ca3af; white-space: nowrap; min-width: 1.6rem; text-align: center; }
.cefr.A1 { background: #16a34a; } .cefr.A2 { background: #0d9488; } .cefr.B1 { background: #2563eb; }
.cefr.B2 { background: #7c3aed; } .cefr.C1 { background: #ea580c; } .cefr.C2 { background: #dc2626; }
.hidden { display: none !important; }
"""

_JS = """
(function () {
  var q = document.getElementById('q'),
      partSel = document.getElementById('part'),
      catSel = document.getElementById('cat'),
      llm = document.getElementById('llm'),
      lvlSel = document.getElementById('level'),
      count = document.getElementById('count'),
      rows = Array.prototype.slice.call(document.querySelectorAll('li.row')),
      catsByPart = JSON.parse(document.getElementById('cats-by-part').textContent);

  function rebuildCats() {
    var part = partSel.value, keep = catSel.value;
    catSel.innerHTML = '<option value="">All categories</option>';
    var cats = part ? (catsByPart[part] || []) : [].concat.apply([], Object.keys(catsByPart).map(function (p) { return catsByPart[p]; }));
    cats.forEach(function (c) {
      var o = document.createElement('option');
      o.value = c; o.textContent = c; catSel.appendChild(o);
    });
    catSel.value = cats.indexOf(keep) >= 0 ? keep : '';
  }

  function apply() {
    var needle = q.value.trim().toLowerCase(),
        part = partSel.value, cat = catSel.value, llmOnly = llm.checked,
        level = lvlSel.value, shown = 0;
    rows.forEach(function (r) {
      var ok = (!part || r.dataset.part === part) &&
               (!cat || r.dataset.cat === cat) &&
               (!level || r.dataset.level === level) &&
               (!llmOnly || r.dataset.llm === '1') &&
               (!needle || r.dataset.search.indexOf(needle) >= 0);
      r.classList.toggle('hidden', !ok);
      if (ok) shown++;
    });
    document.querySelectorAll('.catgroup').forEach(function (g) {
      g.classList.toggle('hidden', !g.querySelector('li.row:not(.hidden)'));
    });
    document.querySelectorAll('.partgroup').forEach(function (g) {
      g.classList.toggle('hidden', !g.querySelector('li.row:not(.hidden)'));
    });
    count.textContent = shown + ' / ' + rows.length + ' constructions';
  }

  q.addEventListener('input', apply);
  partSel.addEventListener('change', function () { rebuildCats(); apply(); });
  catSel.addEventListener('change', apply);
  llm.addEventListener('change', apply);
  lvlSel.addEventListener('change', apply);
  rebuildCats(); apply();
})();
"""


def _grouped():
    """Part -> [(category, category_name, [construct,...])], inventory order."""
    parts: dict = {}
    for c in constructs().values():
        cats = parts.setdefault(c.get("part", ""), {})
        cats.setdefault((c["category"], c.get("category_name", c["category"])), []).append(c)
    return parts


def _row(c: dict) -> str:
    e = html.escape
    needs_llm = c.get("detector_type", "") in LLM_TYPES
    use_notes = (c.get("use_notes") or "").strip()
    sub = f"Family: {e(c.get('family') or '—')}"
    if use_notes:
        sub += f" · {e(use_notes)}"
    search = " ".join([c["id"], c.get("name", ""), c.get("family", ""),
                       c.get("category_name", ""), c.get("example", ""),
                       use_notes]).lower()
    example = (f'<p class="ex">{e(c.get("example", ""))}</p>'
               if c.get("example") else "")
    tier = TIER_LABELS.get(c.get("detector_type", ""), c.get("detector_type", ""))
    level = cefr_level(c["id"])
    badge = (f'<span class="cefr {level}" title="CEFR level (data/cefr_levels.csv)">'
             f'{level if level in CEFR_LEVELS else "—"}</span>')
    return (
        f'<li class="row" id="{e(c["id"])}" data-part="{e(c.get("part", ""))}" '
        f'data-cat="{e(c.get("category_name", ""))}" data-level="{e(level)}" '
        f'data-llm="{"1" if needs_llm else "0"}" data-search="{e(search)}">'
        f'<a class="cid" href="#{e(c["id"])}">{e(c["id"])}</a>{badge}'
        f'<div class="body"><p class="name">{e(c.get("name", ""))}</p>'
        f'{example}<p class="sub">{sub}</p></div>'
        f'<span class="tier{" needs-llm" if needs_llm else ""}">{e(tier)}</span>'
        f'</li>')


@functools.lru_cache(maxsize=1)
def reference_html() -> str:
    e = html.escape
    parts = _grouped()
    n = len(constructs())

    cats_by_part = {part: [name for (_, name) in cats]
                    for part, cats in parts.items()}

    body = []
    for part, cats in parts.items():
        body.append(f'<section class="partgroup"><h2>{e(part)}</h2>')
        for (cat, cat_name), items in cats.items():
            body.append(f'<div class="catgroup"><h3>{e(cat)} — {e(cat_name)}</h3>'
                        '<ul class="rows">')
            body.extend(_row(c) for c in items)
            body.append('</ul></div>')
        body.append('</section>')

    part_options = "".join(f'<option value="{e(p)}">{e(p)}</option>'
                           for p in parts)
    level_options = "".join(f'<option value="{lv}">{lv}</option>'
                            for lv in CEFR_LEVELS) + \
        '<option value="unrated">unrated</option>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>POLKE construction reference</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>POLKE — construction inventory</h1>
<p class="meta">{n} grammatical constructions · polke {e(__version__)} ·
<a href="https://github.com/chxiaobin/polke-open">source &amp; docs</a> ·
<a href="/verify">level verifier</a></p>
<div class="controls">
<input id="q" type="search" placeholder="Search id, name, example…"
 aria-label="Search constructions">
<select id="part" aria-label="Part"><option value="">All parts</option>{part_options}</select>
<select id="cat" aria-label="Category"><option value="">All categories</option></select>
<select id="level" aria-label="CEFR level"><option value="">All CEFR levels</option>{level_options}</select>
<label><input type="checkbox" id="llm"> LLM tier only</label>
<span id="count"></span>
</div>
{"".join(body)}
<script id="cats-by-part" type="application/json">{json.dumps(cats_by_part)}</script>
<script>{_JS}</script>
</div></body></html>"""
