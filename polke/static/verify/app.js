/* CEFR Verifier UI: streams /api/analyze (NDJSON) and renders per-sentence
   vocabulary levels + grammar constructions with client-side filters. */
(() => {
  const LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"];
  const COLORS = { A1: "#16a34a", A2: "#0d9488", B1: "#2563eb", B2: "#7c3aed", C1: "#ea580c", C2: "#dc2626", none: "#9ca3af", unrated: "#9ca3af" };
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "dataset") Object.assign(n.dataset, v);
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) n.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k !== null && k !== undefined) n.append(k.nodeType ? k : document.createTextNode(k));
    return n;
  };

  const state = {
    catalog: null, categories: [], llm: null,
    records: [], meta: null, summary: null, abort: null,
    filters: { vocab: new Set([...LEVELS, "none"]), gram: new Set([...LEVELS, "unrated"]),
               tiers: new Set(["rule", "lexicon", "llm"]), conf: 0, cats: new Set() },
    pinned: new Set(),
  };

  // ---------- filter chips ----------
  function chip(group, value, label) {
    const input = el("input", { type: "checkbox", value, checked: "" });
    const lab = el("label", { class: "chip on", style: `--c:${COLORS[value] || "#374151"}` }, input, label);
    input.addEventListener("change", () => {
      lab.classList.toggle("on", input.checked);
      if (input.checked) state.filters[group].add(value); else state.filters[group].delete(value);
      render();
    });
    return lab;
  }
  function buildChips() {
    const v = $("#vocab-levels"), g = $("#gram-levels");
    for (const lv of LEVELS) { v.append(chip("vocab", lv, lv)); g.append(chip("gram", lv, lv)); }
    v.append(chip("vocab", "none", "not in list"));
    g.append(chip("gram", "unrated", "unrated"));
    $("#tiers").querySelectorAll("input").forEach(inp => inp.addEventListener("change", () => {
      inp.parentElement.classList.toggle("on", inp.checked);
      if (inp.checked) state.filters.tiers.add(inp.value); else state.filters.tiers.delete(inp.value);
      render();
    }));
    $("#conf").addEventListener("input", e => {
      state.filters.conf = parseFloat(e.target.value); $("#conf-val").textContent = state.filters.conf.toFixed(2); render();
    });
    document.querySelectorAll("[data-all],[data-none]").forEach(b => b.addEventListener("click", () => {
      const group = b.dataset.all || b.dataset.none, on = "all" in b.dataset;
      const box = { vocab: "#vocab-levels", gram: "#gram-levels", cat: "#categories" }[group];
      document.querySelectorAll(`${box} input`).forEach(inp => { if (inp.checked !== on) { inp.checked = on; inp.dispatchEvent(new Event("change")); } });
    }));
    ["#color-words", "#show-phrases", "#only-matching", "#show-rationale"].forEach(s => $(s).addEventListener("change", render));
    $("#reset").addEventListener("click", () => {
      document.querySelectorAll("#vocab-levels input, #gram-levels input, #tiers input, #categories input").forEach(inp => {
        if (!inp.checked) { inp.checked = true; inp.dispatchEvent(new Event("change")); } });
      $("#conf").value = 0; $("#conf").dispatchEvent(new Event("input"));
    });
  }
  function buildCategories(rows) {
    const byPart = new Map();
    for (const r of rows) {
      if (!byPart.has(r.part)) byPart.set(r.part, new Map());
      const m = byPart.get(r.part);
      if (!m.has(r.category)) m.set(r.category, { name: r.category_name, n: 0 });
      m.get(r.category).n++;
    }
    const box = $("#categories");
    for (const [part, cats] of byPart) {
      box.append(el("div", { class: "part" }, part || "Other"));
      for (const [cat, info] of cats) {
        state.filters.cats.add(cat);
        const inp = el("input", { type: "checkbox", value: cat, checked: "" });
        inp.addEventListener("change", () => { if (inp.checked) state.filters.cats.add(cat); else state.filters.cats.delete(cat); updateCatCount(); render(); });
        box.append(el("label", {}, inp, el("code", {}, cat), ` ${info.name} `, el("span", { class: "hint", style: "display:inline" }, `(${info.n})`)));
      }
    }
    updateCatCount();
  }
  function updateCatCount() {
    const total = document.querySelectorAll("#categories input").length;
    $("#cat-count").textContent = state.filters.cats.size === total ? "(all)" : `(${state.filters.cats.size}/${total})`;
  }

  // ---------- predicates ----------
  const tierKey = t => (t === "rule" || t === "lexicon") ? t : "llm";
  const consVisible = c => state.filters.gram.has(c.level) && state.filters.tiers.has(tierKey(c.tier))
    && (c.confidence ?? 1) >= state.filters.conf && state.filters.cats.has(c.category);
  const tokLevelKey = t => t.level || "none";
  const tokVisible = t => t.kind !== "skip" && state.filters.vocab.has(tokLevelKey(t));

  // ---------- rendering ----------
  function render() {
    const box = $("#sentences");
    box.replaceChildren();
    const onlyMatching = $("#only-matching").checked;
    let shown = 0;
    for (const r of state.records) {
      const cons = r.constructions.filter(consVisible);
      const toks = r.vocab.tokens.filter(tokVisible);
      const matches = cons.length > 0 || toks.length > 0;
      if (onlyMatching && !matches) continue;
      shown++;
      box.append(renderSentence(r, cons, toks.length));
    }
    $("#empty").hidden = state.records.length > 0;
    if (state.records.length && !shown) box.append(el("div", { class: "empty" }, "No sentence matches the current filters."));
    renderSummary();
  }

  function renderSentence(r, cons, nTok) {
    const card = el("div", { class: "sentence" + (cons.length + nTok ? "" : " dim"), dataset: { i: r.index } });
    const badges = el("div", { class: "badges" },
      el("span", { class: "hint", style: "display:inline" }, "vocab"), levelBadge(r.max_vocab_level || "none", r.max_vocab_level ? "" : "–"),
      el("span", { class: "hint", style: "display:inline" }, "grammar"), levelBadge(r.max_grammar_level || "unrated", r.max_grammar_level ? "" : "–"));
    card.append(el("div", { class: "shead" }, el("span", { class: "num" }, `#${r.index + 1}`),
      el("span", {}, `${cons.length}/${r.constructions.length} constructions`), badges));
    if (r.error) card.append(el("div", { class: "serr" }, "Error: " + r.error));

    // sentence text
    const stext = el("div", { class: "stext" + ($("#color-words").checked ? " colour" : "") + ($("#show-phrases").checked ? " phrases" : "") });
    const tokEls = [];
    for (const t of r.vocab.tokens) {
      const cls = ["tok"];
      if (t.kind !== "skip") {
        cls.push("w", tokLevelKey(t));
        if (!state.filters.vocab.has(tokLevelKey(t))) cls.push("filtered");
        if (t.phrase !== null && t.phrase !== undefined) cls.push("ph");
        if (!t.pos_match) cls.push("posmis");
      }
      const span = el("span", { class: cls.join(" "), dataset: { i: t.i } }, t.text);
      if (t.kind !== "skip") {
        span.addEventListener("mouseenter", e => showTip(e, tokTip(t, r)));
        span.addEventListener("mouseleave", hideTip);
      }
      tokEls.push(span);
      stext.append(span, t.ws);
    }
    card.append(stext);

    // constructions
    if (cons.length) {
      const showRat = $("#show-rationale").checked;
      const tbl = el("table", { class: "ctable" },
        el("thead", {}, el("tr", {}, el("th", {}, "Level"), el("th", {}, "ID"), el("th", {}, "Construction"), el("th", {}, "Category"), el("th", {}, "Span"), el("th", {}, "Tier"))));
      const tb = el("tbody");
      for (const c of cons) {
        const key = `${r.index}:${c.id}:${c.start}`;
        const tr = el("tr", { class: "c" + (state.pinned.has(key) ? " pin" : "") },
          el("td", {}, levelBadge(c.level)),
          el("td", { class: "id" }, c.id),
          el("td", {}, c.name, c.family ? el("div", { class: "hint" }, c.family) : null,
            showRat && c.rationale ? el("div", { class: "rat" }, c.rationale) : null),
          el("td", {}, el("span", { class: "hint", style: "display:inline" }, c.category_name || c.category)),
          el("td", { class: "m" }, c.matched || ""),
          el("td", { class: "tier" }, tierLabel(c)));
        const hl = on => { for (let i = c.token_start; i <= c.token_end && i < tokEls.length; i++) tokEls[i].classList.toggle("hl", on || state.pinned.has(key)); };
        tr.addEventListener("mouseenter", () => hl(true));
        tr.addEventListener("mouseleave", () => hl(false));
        tr.addEventListener("click", () => { if (state.pinned.has(key)) state.pinned.delete(key); else state.pinned.add(key); tr.classList.toggle("pin"); hl(false); });
        if (state.pinned.has(key)) hl(true);
        tb.append(tr);
      }
      tbl.append(tb);
      card.append(tbl);
    } else if (!r.error) {
      card.append(el("div", { class: "nocons" }, r.constructions.length ? "All constructions hidden by filters." : "No constructions detected."));
    }
    return card;
  }

  const tierLabel = c => {
    const t = { rule: "rule", lexicon: "lexicon", llm: "LLM", hybrid_rule_llm: "rule+LLM", hybrid_lexicon_llm: "lexicon+LLM" }[c.tier] || c.tier;
    return c.tier === "rule" || c.tier === "lexicon" ? t : `${t} · ${(c.confidence ?? 1).toFixed(2)}`;
  };
  const levelBadge = (lv, text) => el("span", { class: `lv lv-${lv}` }, text || (lv === "none" ? "n/a" : lv));

  function tokTip(t, r) {
    const parts = [];
    const lv = t.level ? levelBadge(t.level) : levelBadge("none", t.kind === "proper" ? "proper noun" : "not in list");
    parts.push(el("div", {}, lv, el("b", {}, t.headword || t.lemma)));
    if (t.phrase !== null && t.phrase !== undefined) parts.push(el("div", {}, "multi-word entry: ", el("b", {}, r.vocab.phrases[t.phrase].headword)));
    parts.push(el("div", {}, `lemma: ${t.lemma} · POS: ${t.pos} (${t.tag})`));
    if (t.list_pos) parts.push(el("div", {}, `list POS: ${t.list_pos}${t.pos_match ? "" : " (POS mismatch – lowest level taken)"} · ${t.source === "octanove" ? "Octanove C1/C2" : "CEFR-J 1.5"}`));
    return parts;
  }
  const tip = $("#tooltip");
  function showTip(e, nodes) {
    tip.replaceChildren(...nodes); tip.hidden = false;
    const rect = e.target.getBoundingClientRect();
    let x = rect.left, y = rect.bottom + 6;
    tip.style.left = Math.min(x, window.innerWidth - 300) + "px"; tip.style.top = y + "px";
  }
  function hideTip() { tip.hidden = true; }

  function renderSummary() {
    const box = $("#summary");
    if (!state.records.length) { box.hidden = true; return; }
    box.hidden = false; box.replaceChildren();
    // recompute over records with current filters (vocab counts respect level filter for display only)
    const vc = {}, gc = {}; let words = 0, consN = 0;
    for (const r of state.records) {
      const seen = new Set();
      for (const t of r.vocab.tokens) {
        if (t.kind === "skip") continue;
        if (t.phrase !== null && t.phrase !== undefined) { if (seen.has(t.phrase)) continue; seen.add(t.phrase); }
        words++; const k = tokLevelKey(t); vc[k] = (vc[k] || 0) + 1;
      }
      for (const c of r.constructions) { if (!consVisible(c)) continue; consN++; gc[c.level] = (gc[c.level] || 0) + 1; }
    }
    box.append(dist("Vocabulary (word-list entries by level)", vc, words, [...LEVELS, "none"], "word"),
               dist("Grammar constructions (after filters)", gc, consN, [...LEVELS, "unrated"], "construction"));
    const m = state.meta || {};
    box.append(el("div", {}, el("h3", {}, "Run"),
      el("div", { class: "kv" }, el("b", {}, state.records.length), ` sentence${state.records.length === 1 ? "" : "s"} · `, el("b", {}, words), " words"),
      el("div", { class: "kv" }, "LLM tiers: ", el("b", {}, m.llm ? (m.llm.used ? `on (${m.llm.model})` : `off — ${m.llm.reason}`) : "–")),
      state.summary && state.summary.errors ? el("div", { class: "serr" }, `${state.summary.errors} sentence(s) failed`) : null));
  }
  function dist(title, counts, total, keys, noun) {
    const bar = el("div", { class: "dist" }), leg = el("div", { class: "dist-legend" });
    for (const k of keys) {
      const n = counts[k] || 0; if (!n) continue;
      bar.append(el("span", { style: `width:${(100 * n / Math.max(total, 1)).toFixed(2)}%;background:${COLORS[k]}`, title: `${k}: ${n}` }));
      leg.append(el("span", {}, el("i", { style: `background:${COLORS[k]}` }), `${k === "none" ? "not in list" : k} ${n} (${total ? Math.round(100 * n / total) : 0}%)`));
    }
    return el("div", {}, el("h3", {}, title), bar, leg, el("div", { class: "kv" }, `${total} ${noun}${total === 1 ? "" : "s"}`));
  }

  // ---------- analysis ----------
  async function analyze() {
    const text = $("#text").value;
    if (!text.trim()) return;
    if (state.abort) state.abort.abort();
    const ctrl = new AbortController(); state.abort = ctrl;
    state.records = []; state.meta = null; state.summary = null; state.pinned.clear();
    render();
    $("#error").hidden = true; $("#analyze").disabled = true; $("#cancel").hidden = false; $("#export").disabled = true;
    const prog = $("#progress"); prog.hidden = false; setProgress(0, 0, "splitting sentences…");
    try {
      const resp = await fetch("/verify/analyze", { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ text, use_llm: $("#use-llm").checked }), signal: ctrl.signal });
      if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
      const reader = resp.body.getReader(), dec = new TextDecoder();
      let buf = "", total = 0;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
          if (!line) continue;
          const rec = JSON.parse(line);
          if (rec.type === "meta") { state.meta = rec; total = rec.sentence_count; setProgress(0, total); }
          else if (rec.type === "sentence") { state.records.push(rec); setProgress(state.records.length, total); $("#empty").hidden = true; $("#sentences").append(renderSentence(rec, rec.constructions.filter(consVisible), rec.vocab.tokens.filter(tokVisible).length)); renderSummary(); }
          else if (rec.type === "done") { state.summary = rec.summary; }
        }
      }
      render();
    } catch (e) {
      if (e.name !== "AbortError") { $("#error").textContent = "Analysis failed: " + e.message; $("#error").hidden = false; }
    } finally {
      $("#analyze").disabled = false; $("#cancel").hidden = true; prog.hidden = true; state.abort = null;
      $("#export").disabled = !state.records.length;
    }
  }
  function setProgress(done, total, text) {
    const p = $("#progress"); p.querySelector(".bar").style.setProperty("--p", total ? `${100 * done / total}%` : "0%");
    p.querySelector(".ptext").textContent = text || `${done} / ${total} sentences`;
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify({ meta: state.meta, sentences: state.records, summary: state.summary }, null, 1)], { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: "cefr-analysis.json" }); document.body.append(a); a.click(); a.remove();
  }

  const SAMPLE = `I have lived in this town for ten years.
If I had known about the meeting, I would have come earlier.
The bridge, which was built in 1900, is said to be the oldest in the region.
Could you pass the salt, please?
Never have I seen such a reluctant audience.
What I need now is a decent cup of coffee.
She kept talking although nobody was listening.
The committee's decision was unanimously endorsed by the shareholders.`;

  // ---------- init ----------
  async function init() {
    buildChips();
    $("#analyze").addEventListener("click", analyze);
    $("#cancel").addEventListener("click", () => state.abort && state.abort.abort());
    $("#sample").addEventListener("click", () => { $("#text").value = SAMPLE; });
    $("#export").addEventListener("click", exportJson);
    $("#text").addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze(); });
    try {
      const cat = await (await fetch("/catalog")).json();
      state.catalog = cat.constructions; buildCategories(cat.constructions);
    } catch (e) { console.error(e); }
    await pollHealth();
  }
  async function pollHealth() {
    const s = $("#status");
    try {
      const h = await (await fetch("/health")).json();
      if (h.status !== "ok") { s.className = "status"; s.replaceChildren(el("span", { class: "dot" }), "loading models…"); setTimeout(pollHealth, 2000); return; }
      state.llm = h.llm;
      if (h.llm && h.llm.ready) { s.className = "status ok"; s.replaceChildren(el("span", { class: "dot" }), `ready · LLM tiers: ${h.llm.model}`); }
      else { s.className = "status off"; s.replaceChildren(el("span", { class: "dot" }), `ready · LLM tiers off (${h.llm ? h.llm.reason : "?"})`); $("#use-llm").checked = false; $("#use-llm").disabled = true; }
      if (!state.catalog) { const cat = await (await fetch("/catalog")).json(); state.catalog = cat.constructions; buildCategories(cat.constructions); }
    } catch (e) { s.className = "status err"; s.replaceChildren(el("span", { class: "dot" }), "server unreachable"); setTimeout(pollHealth, 3000); }
  }
  init();
})();
