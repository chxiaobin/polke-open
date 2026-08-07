# Known detector gaps

Constructs whose **rule stage proposes no candidate span** on at least one
contract-positive sentence — the LLM judgment stage is never consulted, so
these fail the contract suite with *any* model and are silent recall holes in
production annotation. Identified during the 2026-08 contract cleanup
(triage: run each failing contract case and count LLM tasks; zero tasks =
rule gap).

| construct | example the rule misses | note |
|---|---|---|
| ASC-06 | When he comes, we'll eat. | time-clause present-for-future; same gap family as FUT-13/VTA-10 |
| ASC-29 | If necessary, call. | verbless reduced clause |
| CLS-16 | She seems nice. | copular adjective-vs-adverb |
| FOC-06 | All I want is the truth | all/reason cleft |
| FUT-13 | I'll call when I get there. | see ASC-06 |
| IMP-04 | You sit here | imperative with overt subject |
| IMP-07 | Let me help | let-imperative |
| MOD-22 | You needn't have paid. | needn't tokenisation |
| MOD-28 | Why not ask her? | why (don't you) / why not advice |
| NEG-03 | I never saw anyone there. | never as licensor (only not/n't handled) |
| NFV-15 | Swimming is good for you. | gerund as subject |
| NFV-19 | Seeing is believing. | gerund as complement |
| NFV-31 | I love travelling. | -ing branch of would-like contrast |
| PAS-18 | I had my visa renewed. | causative have/get + object + participle |
| VCP-07 | The two lines intersect. | reciprocal intransitive |
| VCP-35 | I find it hard to concentrate | anticipatory it object |
| VSP-09 | If only I had asked. | if only inversion family |
| VTA-10 | When he arrives, we'll start. | see ASC-06 |

Additional weakness (not a rule bug): with `en_core_web_sm`, **contracted
auxiliaries** ("She's just left", "He's broken his leg") are often mistagged,
so present-perfect detectors (VTA-31/32 among others) can miss contracted
forms — highly relevant for spoken corpora, where 's = has is the norm. The
contract uses uncontracted forms for these two; the corpus validation
(probe → adjudicate → score) is the instrument that quantifies the real-world
impact. A larger spaCy model (`POLKE_SPACY_MODEL=en_core_web_trf`) may reduce
it.

Constructs in the table should either get rule fixes or be flagged when
reporting corpus frequencies.
