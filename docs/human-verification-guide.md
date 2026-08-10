# Human verification guide (test phase)

You are judging a capped sample from the **test corpus** (25 Spoken BNC2014
chunks from recordings disjoint from the development set, seed 2026). The
detectors were frozen before this corpus was drawn; nothing you judge here
feeds back into tuning — it measures the final precision/recall.

## Setup

```bash
source .venv/bin/activate
polke adjudicate examples/test-corpus-bnc/annotations-human --by-line --port 8123
# open http://127.0.0.1:8123/
```

Verdicts save automatically to
`examples/test-corpus-bnc/annotations-human/verdicts.json` on every click.
You can stop and resume at any time — nothing is lost.

## What you will see

- **System annotations** (~3 per construct): spans POLKE claims realise the
  construct. Judge each:
  - **tp** — the span does realise the construct (be lenient about a span
    being a word or two off if the construction is clearly there; use
    **span** if the extent is badly wrong but the construct is present).
  - **fp** — the construct is not there.
  - **wrong** — a real construction, but a *different* construct id
    (pick the correct one if offered).
- **Probe candidates** (~2 per construct, marked as probe/LLM suggestions):
  places a second/third model thinks POLKE **missed** the construct. Judge:
  - **fn** — yes, the construct is there and POLKE missed it.
  - **reject** — the probe is wrong; nothing was missed.

## Conventions (match the dev-phase judging)

- Utterance-per-line: each line is one speaker turn; sentence boundaries
  inside a line are parser guesses.
- Judge the construct **definition**, not the detector's rationale.
- Dysfluent repeats ("that scheme that scheme he was talking about") still
  count if the construction is genuinely produced.
- Backchannels (bare *yeah/mm/right*) are inserts, not ellipsis/answers.
- Verb **particles** (*tidy up, turn it off*) are not prepositions;
  possession *have got* is not a passive; *you know / I mean* fillers are
  discourse markers, not their literal constructions.
- When genuinely torn, prefer the stricter reading (fp/reject) and move on —
  target pace is 3–4 seconds per item; the whole package is ≤ ~1,800
  judgments (~2.5–3.5 h; two sittings recommended).

## After you finish

```bash
polke score examples/test-corpus-bnc/annotations-human \
    examples/test-corpus-bnc/annotations-human/verdicts.json
```

reports per-construct precision (from tp/fp) and probe-based recall
(fn/reject). A separate, blind model adjudication of the same items exists
for agreement analysis — do not look at it before judging.
