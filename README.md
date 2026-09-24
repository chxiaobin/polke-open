# POLKE — pedagogical grammar construction annotator

POLKE detects **670 grammatical constructions** of a pedagogical inventory in
English text and emits span-level annotations. The inventory covers the verb
phrase, noun phrase, clause & sentence grammar, clause connection, discourse &
information structure, and spoken/interactional grammar; the Part VI spoken
stratum was derived from a gap analysis against the *Grammar of Spoken and
Written English* (Biber et al. 2021) and the *Cambridge Grammar of English*
(Carter & McCarthy 2006) — see `docs/gap_analysis_report.md`. The full
inventory with IDs, examples, and detector tiers is listed in
[`docs/constructions.md`](docs/constructions.md).

POLKE also ships a **CEFR level verifier** (`polke level`, and the `/verify`
web UI of `polke serve`): it levels every word of a text with the CEFR-J /
Octanove vocabulary profiles and every detected construction with a CEFR
level from `polke/data/cefr_levels.csv` — see
[Level verifier](#level-verifier-vocabulary--grammar-cefr-levels).

Detectors come in four tiers:

| tier | mechanism | needs LLM | constructs |
|---|---|---|---|
| `rule` | spaCy `DependencyMatcher` patterns | no | 181 |
| `lexicon` | dependency pattern gated by curated lexicons | no | 209 |
| `hybrid_rule_llm` | rule proposes a span, LLM picks the reading | yes | 209 |
| `hybrid_lexicon_llm` | lexicon proposes, LLM picks the reading | yes | 6 |
| `llm` | LLM judges presence over sentence spans | yes | 65 |

Without an API key the tool still runs: the two offline tiers annotate
normally and the LLM tiers are skipped (with a startup warning).

**Use-mode caveat:** input is assumed well-formed (run grammatical error
correction upstream if you feed learner text), and the detectors have not yet
been validated against a human-annotated gold sample — treat output as
pre-annotation, not ground truth.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
cp .env.example .env        # add OPENAI_API_KEY for the LLM tiers (optional)
polke check                 # diagnostics: spaCy model + model API readiness
```

## Command line: annotate a corpus folder

```bash
# All 670 constructs, every .txt under corpus/, one JSON per file:
polke annotate path/to/corpus/

# Only passives and relative clauses, combined JSONL output, offline tiers only:
polke annotate corpus/ -c PAS,REL --no-llm --jsonl out.jsonl

# Browse the inventory (also listed in docs/constructions.md):
polke catalog | less
polke catalog --full -c PAS,VTA-03   # with family, example, use notes, tier

# How is a construct actually detected? Mechanism, dependency patterns,
# lexicons, LLM prompts, and the contract test sentences:
polke explain PAS-01,VTA-29          # or a whole category: polke explain REL
```

Each output record:

```json
{
  "file": "corpus/essay1.txt",
  "text_id": "essay1",
  "spacy_model": "en_core_web_sm",
  "llm": {"ready": true, "model": "gpt-4o-mini", "reason": "ok"},
  "text": "the annotated text, embedded so the record is self-contained",
  "annotations": [
    {"construct_id": "PAS-01",
     "span": {"start_char": 4, "end_char": 14, "token_start": 1, "token_end": 3},
     "detector_type": "rule", "confidence": 1.0, "...": "..."}
  ]
}
```

## Checking annotations: the viewer

```bash
polke view corpus/annotations/          # or a single .annotations.json / .jsonl
polke view out.jsonl -o report.html --open

# or build it directly as part of annotation (--open also launches a browser):
polke annotate corpus/ --view
```

Builds one self-contained HTML page (default: `view.html` next to the input)
showing every annotation aligned under the sentence it falls in, with the
annotated span highlighted — plus per-record selection, text search,
category/tier filters, and LLM confidence + rationale where present. Nothing
to install or serve; open the file in any browser. Older annotation files
without the embedded `text` field still work as long as the recorded source
`.txt` path is resolvable (otherwise only the matched fragments are shown).

### Verification workflow: probe → adjudicate → score

To validate annotation quality on a real corpus (per-construct precision,
recall, F1):

```bash
polke annotate corpus/                      # 1. annotate
polke probe corpus/annotations/ -c PAS,VTA  # 2. LLM sweep for MISS candidates
polke view corpus/annotations/ --open       # 3. adjudicate in the browser
polke score corpus/annotations/ verdicts.json --min-support 5   # 4. stats
```

For **remote adjudication** (records on a server, judging from your own
machine), serve the viewer instead of opening the file:

```bash
polke adjudicate corpus/annotations/ --by-line          # http://127.0.0.1:8123
ssh -L 8123:127.0.0.1:8123 you@server                   # then browse localhost:8123
```

Every verdict is persisted server-side to `verdicts.json` next to the
records as you judge (the browser's localStorage is only a cache), so
`polke score` reads it directly — no export/import dance — and you can
switch machines mid-adjudication.

`probe` asks an LLM — per sentence, per category, with no rule gating — which
constructions are present, and stores whatever the system did *not* annotate
as probe candidates inside the records (one call per sentence × category;
restrict with `-c`, and prefer a *different* model via `--model` /
`POLKE_PROBE_MODEL` so probe errors don't correlate with the annotator's LLM
tiers). The probe supports two providers, chosen by the model id: `claude-*`
models run on the Anthropic API (`pip install "polke[anthropic]"` and set
`ANTHROPIC_API_KEY`) — e.g. `--model claude-opus-5`, or `claude-haiku-4-5`
for large corpora on a budget — everything else runs on the OpenAI API.
Since the annotator's LLM tiers are OpenAI-based, an Anthropic probe gives
you a genuinely independent second annotator. In the viewer every row then carries verdict buttons — system
annotations: **TP** / **FP** / **ID?** (right span, wrong construct — you
supply the correction) / **span** (right construction, wrong extent); probe
candidates: **miss** (confirmed false negative) / **no** (probe noise).
Verdicts persist in the browser (localStorage) and export as `verdicts.json`;
`score` turns records + verdicts into per-construct TP/FP/FN, P/R/F1 and
support, with pending items excluded and reported. A "wrong id" verdict counts
as an FP for the marked construct and an FN for the corrected one. Recall/F1
are relative to the pooled candidates (system + probe) — misses that neither
surfaced stay invisible, so read them as upper bounds.

### Spoken corpora: utterance segmentation + the Spoken BNC2014

Transcribed speech has no sentence punctuation, so the parser's sentence
segmentation is unreliable there. For files with **one utterance per line**,
pass `--by-line` (or set `POLKE_SEGMENT=line`) to make each line the
annotation unit — supported by `annotate`, `view`, and `probe` (use it
consistently across all three):

```bash
polke annotate corpus/ --by-line
```

`scripts/spoken_bnc_prepare.py` turns the downloadable XML edition of the
[Spoken BNC2014](http://corpora.lancs.ac.uk/bnc2014/) into that format —
cleaning transcription markup (vocalisations, pauses, truncations, foreign
material dropped; `<unclear>` best-guesses kept; anonymisation tags replaced
by fixed placeholders) and writing a `.utt.json` sidecar per text that maps
each line to its original utterance number and speaker id (speaker
demographics are in the download's metadata TSVs):

```bash
python scripts/spoken_bnc_prepare.py convert \
    --src path/to/bnc2014-download --out spokenBNC-corpus
python scripts/spoken_bnc_prepare.py sample \
    --corpus spokenBNC-corpus --out validation-chunks \
    --chunks 50 --utterances 40 --seed 42
polke annotate validation-chunks/ --by-line
```

The corpus is free for research (licence signature required) but **must not
be redistributed** — keep the download and everything derived from it (the
converted texts, and annotation records, which embed the source text) out of
version control.

## Level verifier: vocabulary & grammar CEFR levels

`polke level` reports, sentence by sentence, the CEFR level of the
vocabulary and of the grammar used:

```bash
polke level essay.txt              # table: vocab level, grammar level, sentence
polke level essay.txt -v           # + every construction and the words above A2
polke level essay.txt --json       # the stream records, one JSON object per line
polke level - --no-llm < essay.txt # stdin; offline tiers only
```

Every line break is a sentence boundary; longer lines are split by the
parser. The same analysis is exposed by the server as a web UI at
`http://localhost:8100/verify` (the root URL redirects there): paste text,
results stream in per sentence, words are colour-coded by level with the
matching list entry on hover, each sentence lists its constructions with
level, tier, confidence and span (hover to highlight, click to pin), and
filters narrow the view by vocabulary level, grammar level, detector tier,
minimum LLM confidence and construction category. Results can be exported as
JSON.

**Vocabulary levels** come from the Open Language Profiles word lists in
`polke/data/` — the CEFR-J Vocabulary Profile 1.5 (A1–B2, 7 799 entries)
and the Octanove Vocabulary Profile 1.0 (C1–C2, 2 136 entries). Each token
is looked up by spaCy lemma under the list POS labels compatible with its
tag; an entry with another POS is used as a fallback and flagged
(`pos_match: false`). Multi-word entries (*bus stop*, *according to*) are
matched greedily over token sequences first. Proper nouns, numbers and
punctuation are not levelled; other words absent from both lists are
reported as *not in list*. A sentence's vocabulary level is the highest
level found in it.

**Grammar levels**: the inventory itself has no CEFR levels, so
`polke/data/cefr_levels.csv` assigns one to each of the 670 constructions,
calibrated against the Cambridge English Grammar Profile, the CEFR-J Grammar
Profile (`polke/data/cefrj-grammar-profile-20180315.csv`), the Core
Inventory and the GSE. These are **informed estimates, not validated data**
— edit the CSV to recalibrate (loaded at startup; also shown in `/reference`,
`/catalog` and `docs/constructions.md`). The vernacular (`VER`) and
dysfluency (`DYS`) strata are not level-bearing and appear as *unrated*. A
sentence's grammar level is the highest level among its detected
constructions, so a single false positive can raise it — the UI and `-v`
show which construction is responsible.

Python:

```python
from polke.cefr import LevelAnalyzer

an = LevelAnalyzer()                                  # or LevelAnalyzer(annotator=existing)
for rec in an.analyze_iter("I have lived here for years.\nIt was built in 1900."):
    if rec["type"] == "sentence":
        print(rec["max_vocab_level"], rec["max_grammar_level"], rec["text"])
```

Word-list licences: the CEFR-J profiles are © Tono Laboratory, TUFS, free
for research and commercial use with citation; the Octanove profile is
CC BY-SA 4.0 — see `polke/data/README-openlanguageprofiles.md`.

## Docker

```bash
docker build -t polke .
docker run --rm -p 8100:8100 --env-file .env polke

# or pull the published image (tags track releases):
docker run --rm -p 8100:8100 --env-file .env ghcr.io/chxiaobin/polke:latest

# CLI inside the container, corpus mounted from the host:
docker run --rm -v ./corpus:/corpus --env-file .env polke \
  polke annotate /corpus --jsonl /corpus/annotations.jsonl
```

The image ships with `en_core_web_sm` and `en_core_web_md`; select via
`POLKE_SPACY_MODEL`. From another Compose service, build straight from the
repo — no vendored code:

```yaml
polke:
  build:
    context: https://github.com/chxiaobin/polke-open.git#v0.1.0
  environment:
    OPENAI_API_KEY: ${OPENAI_API_KEY:-}
```

## HTTP API

```bash
polke serve --port 8100     # or: uvicorn polke.server:app
```

- `GET /health` — service status + LLM readiness + word-list sizes
- `GET /catalog` — the 670-construct inventory (incl. definitions/use notes
  and `cefr_level`)
- `GET /reference` — browsable HTML reference: search, part/category/CEFR
  level/LLM-tier filters, and a stable anchor per construction (`/reference#PAS-01`)
- `POST /annotate` — `{"text": "...", "constructions": ["PAS", "REL-01"], "context": ["optional preceding utterances"]}`
- `GET /verify` — the level verifier UI (`GET /` redirects here)
- `POST /verify/analyze` — `{"text": "...", "use_llm": true, "constructions": ["PAS"]}`
  → NDJSON stream: one `meta` record, one `sentence` record per sentence (text,
  `vocab.tokens` with level/headword/list source, `vocab.phrases`,
  `constructions` with level/tier/confidence/span/rationale, `max_vocab_level`,
  `max_grammar_level`), one `done` record with totals
- `POST /verify/analyze/json` — the same as one JSON document
- `GET /verify/vocab?q=word` — word-list entries for one word

```bash
curl -s localhost:8100/annotate -H 'content-type: application/json' \
  -d '{"text": "The window was broken by the storm.", "constructions": ["PAS"]}'
```

## Python library

```python
from polke.annotate import Annotator

ann = Annotator()                        # loads spaCy + all detectors once
spans = ann.annotate("She has lived here for years.", selection=["VTA"])
```

## Configuration (.env)

| variable | purpose | default |
|---|---|---|
| `OPENAI_API_KEY` | key for the LLM tiers | unset → LLM tiers off |
| `POLKE_LLM_MODEL` | chat model for reading classifiers | `gpt-4o-mini` |
| `POLKE_SPACY_MODEL` | spaCy pipeline | `en_core_web_sm` |
| `POLKE_LLM_CONCURRENCY` | parallel LLM calls per text | `8` |
| `POLKE_SEGMENT` | `parser`, or `line` = each input line is one sentence (utterance-per-line transcripts) | `parser` |
| `OPENAI_BASE_URL` | point the OpenAI client at a self-hosted OpenAI-compatible server (vLLM, TGI, …) | OpenAI API |
| `POLKE_LLM_NO_THINK` | `1` = disable reasoning mode on self-hosted reasoning models (they otherwise spend the whole token budget thinking); leave unset for the OpenAI API | unset |
| `POLKE_SENTENCE_CONCURRENCY` | level verifier: sentences analysed in parallel when the LLM tiers are on | `3` |
| `POLKE_MAX_SENTENCES` / `POLKE_MAX_CHARS` | level verifier: caps per request | `500` / `100000` |

### Self-hosted / open-weights models

The LLM tiers and the probe speak the OpenAI chat API, so any
OpenAI-compatible server works:

```bash
OPENAI_BASE_URL=https://your-cluster/v1
OPENAI_API_KEY=<cluster token>
POLKE_LLM_MODEL='google/gemma-4-31B-it-qat-w4a16-ct'    # annotator tiers
POLKE_PROBE_MODEL='Qwen/Qwen3.6-35B-A3B'                # independent probe
POLKE_LLM_NO_THINK=1     # if the served models have reasoning mode enabled
```

Readiness checks fall back to the server's model *list* when the
single-model retrieve endpoint is not implemented. Validate a candidate
model against the construct contract with `POLKE_LLM=1 pytest -q
tests/test_contract.py`.

At startup (CLI `annotate`/`check` and server alike) the model API is probed;
if it is unreachable a warning names the reason and the affected tier sizes,
and annotation proceeds with the offline tiers.

## Tests

```bash
pytest -q                   # offline: harness, registry, rule/lexicon contract
POLKE_LLM=1 pytest -q       # + LLM tiers (needs OPENAI_API_KEY)
```

The per-construct contract (positive/negative example sentences in
`polke/data/detectors.json`) is the specification; `tests/test_contract.py`
runs one test per construct.

## Layout

```
polke/
  schema.py registry.py pipeline.py build.py    core: dataclasses, data, build
  detectors/          rule/lexicon/LLM detector machinery + per-category catalog
  data/               constructs.json (inventory), detectors.json (contract),
                      lexicons.json (curated word lists), cefr_levels.csv
                      (CEFR level per construct), CEFR-J / Octanove
                      vocabulary profiles
  annotate.py         high-level Annotator API
  vocab.py cefr.py    word-list vocabulary levelling + the per-sentence
                      level verifier (polke level, /verify)
  static/verify/      the level verifier web UI
  viewer.py           self-contained HTML annotation viewer (polke view)
  cli.py server.py    command line and FastAPI service
docs/                 constructions.md (full inventory reference; regenerate with
                      scripts/generate_constructions_doc.py), inventory provenance
                      (gap analysis, detector notes)
```

## License

MIT — see `LICENSE`.
