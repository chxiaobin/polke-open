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

- `GET /health` — service status + LLM readiness
- `GET /catalog` — the 670-construct inventory (incl. definitions/use notes)
- `GET /reference` — browsable HTML reference: search, part/category/LLM-tier
  filters, and a stable anchor per construction (`/reference#PAS-01`)
- `POST /annotate` — `{"text": "...", "constructions": ["PAS", "REL-01"], "context": ["optional preceding utterances"]}`

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
                      lexicons.json (curated word lists)
  annotate.py         high-level Annotator API
  viewer.py           self-contained HTML annotation viewer (polke view)
  cli.py server.py    command line and FastAPI service
docs/                 constructions.md (full inventory reference; regenerate with
                      scripts/generate_constructions_doc.py), inventory provenance
                      (gap analysis, detector notes)
```

## License

MIT — see `LICENSE`.
