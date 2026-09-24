# Jev vs POLKE pipeline — utterance-level presence on the human-judged test items

606 constructs, 1779 (utterance, construct) pairs from the 1,800-verdict human gold. Jev positive from probability 0.3.

| system | P | R | F1 |
|---|---|---|---|
| POLKE | 0.826 | 0.675 | 0.743 |
| JEV | 0.779 | 0.778 | 0.779 |

## By detector tier of the construct

| tier | n pairs | POLKE P/R/F1 | Jev P/R/F1 |
|---|---|---|---|
| rule | 503 | 0.794/0.664/0.723 | 0.765/0.803/0.783 |
| lexicon | 507 | 0.843/0.587/0.692 | 0.791/0.794/0.792 |
| hybrid_rule_llm | 578 | 0.825/0.735/0.777 | 0.777/0.779/0.778 |
| hybrid_lexicon_llm | 17 | 0.929/1.000/0.963 | 0.900/0.692/0.783 |
| llm | 174 | 0.857/0.737/0.792 | 0.778/0.675/0.723 |

## Jev threshold sweep (micro)

| threshold | P | R | F1 |
|---|---|---|---|
| 0.3 | 0.779 | 0.778 | 0.779 |
| 0.4 | 0.818 | 0.688 | 0.747 |
| 0.5 | 0.847 | 0.609 | 0.708 |
| 0.6 | 0.879 | 0.534 | 0.664 |
| 0.7 | 0.895 | 0.455 | 0.604 |

## Speed and cost (Jev)

Model jev-1.13.0, 0.430 Mtok input per pass ≈ $0.018 (at jev-1.13 pricing).

| pass | requests | questions | wall s | questions/s | latency mean/median/max s |
|---|---|---|---|---|---|
| sequential | 606 | 1779 | 151.99 | 11.7 | 0.251/0.241/0.8502 |
| parallel | 606 | 1779 | 9.55 | 186.4 | 0.248/0.233/0.852 |

Parallel = 16 concurrent requests. Max answer difference between passes: 0.18.

## Speed comparison with the POLKE pipeline

Measured on the same test-corpus texts (full 670-construct annotation,
Gemma-4-31B cluster backend):

| system | workload | wall clock | per utterance |
|---|---|---|---|
| POLKE pipeline (conc 8, measured today) | 3 chunks, 75 utterances, 1,107 annotations | 1,345 s | **17.9 s** |
| POLKE pipeline (conc 16, production runs) | 50 chunks, 2,000 utterances | ~4 h 45 m | **~8.6 s** |
| Jev, experiment workload (judged pairs only) | 1,779 presence questions | 9.6 s (parallel, conc 16) | — |
| Jev, full-inventory extrapolation | 670 presence questions/utterance at measured 242 tok/question, 186 q/s (conc 16) | — | **~3.6 s** (≈$0.0068) |
| Jev at the earlier study's measured throughput (2,864 q/s, written corpus) | same extrapolation | — | **~0.23 s** |

Reading: at matched concurrency Jev is ~2.5–5x faster per utterance than the
pipeline for presence-level annotation, and its throughput scales with
concurrency far beyond the cluster's (the bottleneck is requests, not GPU
time on our side). The extrapolations assume presence-only annotation
(no span localization; spans would add a second, smaller wave of
enumerate-and-select Choice requests for utterances with hits).

## Caveats

- Gold is the human-judged pool (system + probe items): both systems are
  scored only on judged (utterance, construct) pairs; absolute recall
  remains pool-relative for both.
- The Jev threshold sweep is evaluated on the same gold it would be tuned
  on; 0.3 as an operating point needs confirmation on a held-out split.
- POLKE's LLM tiers run on a free university cluster (cost ~$0); Jev cost
  is real but small ($0.018 for the whole experiment; ~$0.007/utterance
  extrapolated full-inventory).
- Jev received only the inventory entry per construct (name, area,
  definition, one catalog example) — no few-shot examples from the corpus,
  no tuning. The pipeline behind the POLKE numbers has been through two
  fix rounds against dev-set evidence.
