# Implementation notes — ambiguous / wrong / hard contract cases

Log of construct IDs where the contract test case is wrong, fragmentary, or the
construct has no clean formal trigger, plus the judgment calls made. Flagging a
bad case here is a correct outcome (per the task brief), not a failure.

## Reference / harness observations

- **PAS-01 reference mismatch.** `detectors/reference.py` labels PAS-01 as
  "present/past simple passive" and matches any `VBN + auxpass`, so it fires on
  PAS-02's past passive ("The bridge was built in 1890.") and fails its own
  contract negative. PAS-01 is *present* simple passive only; PAS-02 is past.
  Resolved by a deterministic `RuleRoutingDetector` (PAS-01..10) that routes on
  the aux-chain tense/aspect and is registered after the reference (overrides
  it). `reference.py` is left untouched so its own dedicated tests stay green.

## Design decisions (judgment calls)

- **Deterministic sibling routers.** Where a group of siblings shares one form
  and the choice among them is decidable by FORM (not meaning) — e.g. the
  passive paradigm PAS-01..10 differing only by aux-chain tense/aspect/modality
  — one `RuleRoutingDetector` serves the whole group and emits at most one id
  per span. This mirrors the LLM hybrid shape but needs no model, and keeps
  precision high (siblings can't cross-fire). See `detectors/routing.py`.

## Flagged constructs

### PREP (prepositions)
- **PREP-01 vs PREP-04** share the prepositions {in, on, at} (place vs time).
  Separated by a DATE/TIME/number-object heuristic on the pobj. Robust on the
  contract fragments but not perfect on free text ("in the morning" has no
  DATE entity); a proper split is a reading task (should arguably be hybrid).
- **PREP-08** ("from") — its positive list mixes `from nine to five`,
  `two years ago`, and `in an hour`. The last overlaps PREP-04's {in} set;
  handled by matching a few fixed `in a/an <period>` phrases. The row conflates
  three time relations under one label.
- **PREP-24** (preposition omission) detects a *proxy* — bare temporal NP
  adjuncts and a lexicon of transitive-no-prep verbs (discuss/enter/reach/...) —
  not true preposition absence, which is not observable in USE mode. Best-effort.
- Many PREP positives are illustrative fragments (bare preposition lists), not
  sentences; handled by surface `PhraseLexiconDetector` rather than dependency
  patterns.

### COH (cohesion & linking adverbials)
- **COH-14 (punctuation of connectors)** is a degenerate contract row: the
  positive is `however, ...` and the negatives are comma-separated *lists* of
  connectors (`moreover, in addition, furthermore, ...`) — which also contain
  connector+comma. There is no form-level signal separating "a connector punctuated
  in running text" from "an item in a metalinguistic list". Best-effort: fire on
  a connector+comma only when what follows the comma is not itself another
  connector (i.e. not an enumeration). Flagged; low validity.
- COH-01..11,13 are served precisely by one routing `PhraseLexiconDetector`
  (each linker belongs to exactly one semantic group).

### ADV / ASC
- **ADV-04** (two adverb forms, meaning difference) — **no positive test case**;
  passes vacuously. Implemented on the distinct `-ly` members; cannot validate.
- **ASC-04 (since=time) vs ASC-08 (since=reason)**, **ASC-02 (as=time) vs
  ASC-08 (as=reason)** — genuinely ambiguous; both fire on the shared connective.
  Not deterministically separable → belongs on the LLM tier. Isolation tests pass.
- **ASC-14** — only the `with the result that` arm has a clean trigger; the
  "so that (no modal)" reading overlaps ASC-10 and is left to it.
- ADV-02/03 (flat adverb citation vs in-use) split on `advmod`-of-VERB; spaCy
  tags the bare citation list as JJ/ADJ, which the gate exploits.

### ADJ / COM / CLS
- **ADJ-15, COM-04** — **no positive test cases**; implemented as real
  JJR/JJS-morphology / irregular-comparative detection; pass only vacuously.
- **CLS-03 vs CLS-07** — structurally identical SVO; separated only by subject
  POS (full lexical NP → CLS-07). Arbitrary but contract-safe.
- **CLS-15** — degenerate: positive is a bare copular-verb list; its negatives
  (`It looks good`=CLS-16, `He seems to know`=CLS-17) are the real realisations.
  Implemented as copular/perception verb minus realised acomp-adj / to-inf, so
  the prototypical copula+adj defers to CLS-16.
- **ADJ-16 vs ADJ-17** (tough-adjectives) split structurally (adj+to-inf vs
  raised-subject acomp); full "The book is hard to read" defers to the LLM tier.

### VCP
- **VCP-27** (`name it X`), **VCP-17/23** — placeholder `X` / slash-lists garble
  the parse; handled with surface `Matcher` fallbacks + `CompositeDetector`.
- **VCP-01/02/05/06/07** — ergative/middle/reciprocal are form-identical to plain
  intransitive; the truly ambiguous ones are correctly on the LLM tier, and
  VCP-01 uses an ergative-verb exclusion lexicon for precision.

### VTA / FUT / MOD
- **VTA-11** (typed `rule`) — "action in progress now"; its 4 negatives are all
  *other* present-continuous readings, so it is inseparable by form. Implemented
  as the default continuous reading with heuristic exclusions; genuinely belongs
  in the present-continuous LLM group. Mistyped in the contract.
- **VTA-17/18/20**, **VTA-23/26** — past-simple / past-continuous families routed
  as a group; the residual "single completed event" / "in progress at past point"
  defaults are partly readings.
- **FUT-20** (`be due to`) over-fires on causal `due to + NOUN` (parser mis-tags
  the infinitive). **FUT-14 vs FUT-15** split only by `?`.

### NEG / IMP / FOC / CON / REP / COORD / ELS
- **NEG-02** (contractions) and **REP-08** (say-vs-tell) are metalinguistic rows:
  positives are form lists/labels, negatives are real clauses. Split on surface
  base form; low validity.
- **FOC-10 vs FOC-13** (so/neither echo) split only by subject type (pronoun vs
  NP). **CON-13** inverted-conditional excluded from inverted questions by a
  trailing-`?` heuristic.

### NOU / ART / DET / PRO
- **NOU-05** — `news` conflated with the generic uncountable list; triggered on
  the `-ics` nouns only.
- **NOU-13 vs NOU-18/19/20** — `'s` genitive split deterministically on possessor
  tag / clitic surface / dep / TIME_MEASURE lexicon / preceding-`of` guard.
- **ART-05** (a/an by sound) — fired only on the salient divergence cases
  (`a university`, `an hour`, `an MP`); firing on all a/an would hit every negative.
- **DET-01/02** (this book vs this week) split by a TIME_NOUNS lexicon; **DET-03
  vs DET-04** resolve `its` to DET-04 gated on a following noun.
- The inventory's zero article is a literal `Ø` token (tagged NNP); ART detectors
  use an optional-`ORTH:"Ø"` Matcher so one pattern fires on both marked examples
  and natural text.

### NFV / NCL / VSP
- **NFV-21/22/23** — bare slash-lists of verbs; implemented as LEMMA phrase
  lexicons over disjoint verb sets, with cross-exclusions forced by a shared
  negative fragment. Fire on any listed verb in real text (coarse, intended).
- **NCL-03/04** — positives mix a real clause with ellipsis fragments; each a
  `CompositeDetector` (real acl/ccomp+that arm OR noun/adj-attached fragment arm).
- **NCL-06 / VSP-05** (mandative subjunctive) overlap by design; gated by a local
  mandative verb/adjective lexicon + base-form check.
- **NFV-03** left as a pure syntactic rule (lemma gates unreliable:
  `hopes`→`hop`, `hated`→`hat`).

## Lexicon consolidation (follow-up)
Agents kept category word-lists as in-module Python constants (to stay
conflict-free during parallel work). Candidates to promote/unify into shared
`data/lexicons.json`: reporting verbs (already shared; VTA/PAS duplicate a local
copy), reflexive/personal/possessive pronoun sets, copular verb sets (current vs
resulting), prepositional/control/gerund/mandative verb tables, article
name/place/zero-idiom sets, linking-adverb sets (COH/ADV overlap), quantifier and
non-assertive sets. Functionally equivalent today; consolidation is a
maintainability cleanup, not a correctness fix.

## New reusable primitives added (beyond the three references)
- `detectors/routing.py :: RuleRoutingDetector` — deterministic sibling router.
- `detectors/routing.py :: LexiconRuleDetector` — lexicon-gated router.
- `detectors/routing.py :: CompositeDetector` — OR several arms for one id.
- `detectors/lexical.py :: PhraseLexiconDetector` — surface/lemma multiword
  matcher (longest-match, non-overlapping); for closed-class categories and the
  fragment-style examples.
