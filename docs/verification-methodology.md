# Human verification of POLKE on the Spoken BNC2014: methodology

This document describes, end to end, how the system's annotation quality was
established by human adjudication (the "test phase", completed 2026-09-23).
It is written to be lifted into the paper's methods section. Final numbers:
precision 0.824, recall 0.676 (upper bound, see §6), F1 0.743 over 1,800
human judgments; per-tier and per-construct tables in
`examples/test-corpus-bnc/score-final-2026-09-23.json`.

## 1. Design principles

1. **Frozen system.** All detectors, prompts, lexicons, and the annotator
   LLM (Gemma-4-31B, reasoning disabled) were frozen before the test corpus
   was drawn. The earlier development cycle (a 50-chunk dev corpus,
   model-assisted adjudication, and two detector fix packages) is a separate
   tuning phase; nothing judged in the test phase fed back into the system.
2. **Held-out data.** The test corpus was drawn exclusively from recordings
   with no overlap with the development sample (`sample --exclude`
   guarantees disjoint source recordings; seed 2026).
3. **Blind judging.** The human judge saw annotations and probe candidates
   without model confidences, probability scores, or any model's
   adjudication. A blind model adjudication of the identical items was
   produced in advance and kept sealed (outside the judging tool's reach)
   until human judging was complete, for a later human-model agreement
   analysis.
4. **Bounded, stratified workload.** Judgments were capped per construct so
   one person could adjudicate the whole inventory breadth-first rather
   than exhaustively judging frequent constructions.

## 2. Test corpus

25 chunks of consecutive utterances (625 utterances in total, ~25 per
chunk) sampled from the converted Spoken BNC2014 (utterance-per-line, the
corpus's own utterance segmentation, markup cleaned per the corpus DTD,
anonymisation placeholders substituted). Chunks come from 25 distinct
recordings, all disjoint from the development recordings.

## 3. System annotation and recall-probe layers

- **System layer:** the full POLKE pipeline (rule, lexicon, hybrid, and LLM
  tiers; utterance = sentence unit) annotated the 25 chunks.
- **Probe layer (false-negative candidates):** because a system cannot
  reveal its own misses, two further LLMs from different model families
  than the annotator (DeepSeek-V4-Flash and Qwen3.6-35B) were each asked,
  for every utterance x construction category with no rule gating, which
  constructions are present. Claims the system had not annotated became
  *probe candidates*; a candidate proposed independently by both models is
  marked as two-model agreement. Probe candidates carry utterance-level
  spans and the probe's quoted evidence.

## 4. The judging package

From these layers a stratified package was drawn (seed 2026): per
construct, up to **3 system annotations** and up to **2 probe candidates**,
selected at random within the construct (probe candidates preferring
two-model agreement), within a total budget of 1,800 items. The final
package: **1,800 items = 937 system annotations + 863 probe candidates,
covering 606 constructs** (the remaining constructs produced neither a
system annotation nor a probe candidate in the test corpus). A manifest
records the sampled items and parameters.

## 5. Adjudication protocol

**Judge.** One expert judge (the author), working in multiple sittings over
twelve days at a target pace of 3-4 seconds per item (~3 hours total).

**Tool.** `polke adjudicate` serves the annotation viewer over HTTP with
every verdict persisted server-side atomically on each click
(`verdicts.json`); judging is resumable across sittings and machines, and
the browser's local storage is only a cache. Each item is shown as the
full utterance with the annotated span highlighted, in document context,
with the construct's definition available on hover. Items are not ordered
by any model score.

**Verdict categories.**
- System annotations: **tp** (the span realises the construct; spans a word
  or two off still count), **span** (construct clearly present but the
  extent is badly wrong - counted as a detection hit, tallied separately
  as a localisation error), **fp** (construct not present), **wrong** (a
  real construction but a different construct id, with correction - counts
  as fp for the marked id and fn for the corrected one; available but not
  used in the final test set).
- Probe candidates: **fn** (the construct is there; a confirmed system
  miss) or **reject** (probe noise).

**Conventions** (fixed in a written guide before judging, matching the
dev-phase conventions): judge the construct definition, never the
detector's rationale; dysfluent repeats count when the construction is
genuinely produced; bare backchannels are inserts, not ellipsis or
response structures; verb particles are not prepositions; possession
*have got* is not a passive; *you know / I mean* fillers are discourse
markers; when genuinely torn, prefer the stricter reading (fp/reject).

**Outcome.** 1,800/1,800 items judged; distribution: tp 643, span 129,
fp 165 (system side), fn 370, reject 493 (probe side); zero pending, zero
stale.

## 6. Scoring

Per construct and aggregated: precision = (tp + span) / (tp + span + fp)
over system annotations - the *lenient* reading in which a span verdict is
a detection hit; the strict variant excluding span verdicts is also
reported (overall P 0.824 lenient vs 0.686 strict; 129/772 = 17% of hits
had imperfect extents, concentrated in determiner/article spans). Recall =
(tp + span) / (tp + span + fn), where fn are the human-confirmed probe
misses. Because misses that neither the system nor either probe surfaced
remain invisible, **recall and F1 are upper bounds relative to the pooled
candidate set**; precision is exact for the judged sample. With ~3 system
items per construct, per-construct figures are indicative only; tier- and
category-level aggregates are the supported granularity, and per-construct
precision/recall accompany downstream frequency tables as qualifiers, not
guarantees.

## 7. Known limitations

1. Single judge: no inter-annotator reliability for the human gold itself.
   The sealed model adjudication of the same 1,800 items is reserved for a
   human-model agreement analysis to partially address this.
2. Pool-based recall (§6): true recall may be lower than 0.676.
3. The per-construct item cap trades depth for breadth: 413 of the 606
   covered constructs have fewer than 4 judged items.
4. The leniency convention for near-miss spans makes the headline numbers
   detection-oriented; localisation quality is reported separately.
5. The probe pool shares a weakness with any LLM-based miss detection:
   constructions that all three model families systematically miss (e.g.
   heavily disfluent realisations) are under-represented among the
   candidates and therefore in the recall denominator.

## 8. Artifacts

| artifact | location |
|---|---|
| judging guide | `docs/human-verification-guide.md` |
| package builder | `scripts/build_human_package.py` |
| package + manifest + verdicts | `examples/test-corpus-bnc/annotations-human/` (local; licensed corpus text) |
| final verdicts backup | `examples/test-corpus-bnc/verdicts-final-2026-09-23.json` |
| full per-construct scoring | `examples/test-corpus-bnc/score-final-2026-09-23.json` |
| scorer | `polke/score.py` (`polke score`) |
| adjudication server | `polke/adjudicate.py` (`polke adjudicate`) |
| sealed model verdicts (unopened) | `examples/bakeoff-logs/test-model-verdicts.json` |
