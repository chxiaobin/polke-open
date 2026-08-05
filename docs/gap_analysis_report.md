# Acteams Grammar Inventory — Gap Analysis against GSWE (2021) and Carter & McCarthy (2006)

**Inputs.** Acteams pedagogical inventory v0.5 (595 constructs, 29 categories, 5 parts, from `constructs.json`); *Grammar of Spoken and Written English* (Biber, Johansson, Leech, Conrad & Finegan, 2021 Benjamins edition, 14 chapters); *Cambridge Grammar of English* (Carter & McCarthy, 2006, 541 numbered sections).

**Method.** Both books were reduced to their full heading scaffolds (GSWE: complete multi-level table of contents, ~950 headings, chapters 2–14; C&M: chapter mini-contents plus in-body heading recovery, ~1,100 section headings, sections 1–538), supplemented by targeted body reads where headings were insufficient (GSWE §2.9.2, §3.14, §9.4.11, ch. 13–14; C&M §94–122, §103, §501, speech-representation chapter). Every book topic was checked against the 595 construct definitions (`name`, `example`, `definition_raw`), not just construct names, to avoid false gaps — e.g. *would you mind + -ing* is already inside MOD-09's definition, and imperative *will you* tags are inside QUE-12, so neither is listed as missing.

**Headline result.** The Acteams inventory covers the shared "common grammar" of both books remarkably well: tense/aspect/modality/voice, complementation, the entire NP apparatus, clause types, subordination, conditionals, relatives, information packaging (clefts, fronting, inversion, dislocation), and cohesion are essentially complete at the books' grain or finer. The systematic gap is exactly where theory predicted it: **the conversation-specific stratum** — GSWE Chapter 14 (inserts, C-units, situational ellipsis subtypes, quotatives, vernacular grammar) and C&M's spoken-language chapters §82–122 (headers/tails aside, which FOC-17 covers: vague language, hedges, vocatives, response tokens, discourse-marker grain, interactional question types). A second, smaller gap set consists of scattered "common-grammar" constructions the pedagogical tradition tends to skip (for…to complements, NP apposition, exclamative clauses, *dare*, independent double negation).

**Deliverable.** 75 proposed constructs in `new_constructs.json` (schema-compatible with `constructs.json`): 64 in a proposed new **Part VI. Spoken & Interactional Grammar** (13 categories) and 11 additions to existing categories. IDs continue existing numbering (no collisions). Detector tiers follow the kit's scheme (rule / lexicon / hybrid_rule_llm / llm).

---

## 1. New Part VI: Spoken & Interactional Grammar (64 constructs)

### INS — Inserts (9)
GSWE treats inserts as a third major word class alongside lexical and function words (§2.2.3.3, §2.5, §14.3.3); the Acteams inventory has no equivalent. These are the highest-priority additions for the BLC agenda: they are pervasive across speakers in conversation regardless of education.

| ID | Construct | Example | Source |
|---|---|---|---|
| INS-01 | Interjections | *Oh! Wow! Ouch!* | GSWE 14.3.3.1; C&M 113 |
| INS-02 | Greetings & farewells | *Hi. See you later.* | GSWE 14.3.3.2; C&M 115 |
| INS-03 | Attention signals | *Hey! Listen. Excuse me!* | GSWE 14.3.3.4 |
| INS-04 | Response elicitors (freestanding) | *Eh? Right? You know?* | GSWE 14.3.3.5 |
| INS-05 | Response forms / backchannels | *Yeah. Mm. Uh huh. Absolutely.* | GSWE 14.3.3.6; C&M 110, 248 |
| INS-06 | Hesitators / filled pauses | *er, erm, um* | GSWE 14.2.1, 14.3.3.7 |
| INS-07 | Polite speech-act formulae | *Please. Thanks. Sorry. Pardon?* | GSWE 14.3.3.8; C&M 423e |
| INS-08 | Expletives & taboo interjections | *Damn! Bloody hell!* | GSWE 14.3.3.9; C&M 114a-b |
| INS-09 | Taboo intensifiers | *bloody freezing* | C&M 114c (FOC-15 covers only wh-intensifiers) |

Boundary note: INS-05 (stand-alone response tokens) is distinct from QUE-03 (elliptical short answers *Yes, I do*) and ELS-08 (*Me too; So do I*), which the inventory already has.

### VAG — Vague language & approximation (4)
C&M §103; GSWE calls the extender type "coordination tags" (§2.9.2) and treats approximate numerals in §2.7.7.3–4. Nothing in Acteams covers these.

VAG-01 general extenders (*and stuff (like that), or something, and so on*); VAG-02 vague category nouns & placeholders (*thing, stuff, thingy, whatsit*); VAG-03 hedging *sort of / kind of* + VP/AdjP (ART-19 covers only the classifying *kind of + noun*); VAG-04 numeral approximators (*about fifty, twenty or so, fiftyish, five or six*).

### CMT — Comment clauses & parentheticals (4)
GSWE §3.4.2, §3.11.6 (comment clauses as a finite dependent clause type); C&M §112, §531f. CMT-01 first-person epistemic parentheticals (*I think / I guess / I reckon*, medial/final or initial without *that*); CMT-02 interactive parentheticals in medial/final position (*you know, you see, mind you*); CMT-03 as-comment clauses (*as you know, as I said*; C&M 314j); CMT-04 medial/final reporting clauses including inversion (*'Fine,' said John* — GSWE 11.2.3.6, currently absent from REP, which only covers initial-frame reporting).

### VOC — Vocatives (3)
GSWE §3.4.6, §14.4.1; C&M §116–118 give vocatives a full treatment (types, positions, discourse functions); Acteams has nothing. Split by form type: VOC-01 names/titles; VOC-02 kinship/endearment/familiarizers (*Mum, love, mate*); VOC-03 impersonal/plural/honorifics (*you guys, folks, sir*). C&M's positional and functional distinctions (§118: summons, turn management, etc.) are left as annotation features rather than separate constructs.

### DMG — Discourse-marker grain (9)
Acteams has COH-11 as a single catch-all ("well, anyway, by the way, you know, I mean, right, actually"). Both books individuate markers by function (GSWE 14.3.3.3; C&M 106–110 distinguish organising, monitoring, reformulating, responding). For per-speaker dispersion measurement a single construct is too coarse — *well* and *I mean* have different social distributions. Proposed splits: DMG-01 *well*; DMG-02 *oh*; DMG-03 *you know*; DMG-04 *I mean*; DMG-05 discourse *like* (llm tier — hardest disambiguation in the set); DMG-06 launcher *so*; DMG-07 *anyway/anyhow*; DMG-08 *right/okay*; DMG-09 turn-initial *and/but* (GSWE 2.4.7.4). COH-11 can be retained as the residual category (*actually, by the way, look/listen…*) or retired to a lexicon.

### QIN — Interactional questions & tags (6)
QIN-01 declarative questions (*You're leaving?* — C&M 430; no Acteams equivalent); QIN-02 statement/copy tags (*He's mad, he is* — C&M 302; distinct from the NP tail in FOC-17 and from reversed-polarity QUE-11); QIN-03 exclamation tags (*Wasn't it brilliant!* — C&M 303); QIN-04 follow-up/two-step questions (C&M 100–101; discourse-level, llm tier); QIN-05 preface questions (*(You) know what? Guess what?* — C&M 102); QIN-06 invariant tags (*innit, yeah?, eh?* — C&M 98, 119).

Verified as already covered: reversed-polarity tags (QUE-11), same-polarity and imperative/*let's* tags and *aren't I* (QUE-12), tags after negative/indefinite hosts (QUE-13), echo/alternative/rhetorical questions (QUE-17), negative questions (QUE-14).

### EXC — Exclamatives (3)
The clause type is missing entirely: Acteams has only ART-04 (*a/an* in exclamations) and DET-23 (*what (a)* as predeterminer). EXC-01 what-exclamatives; EXC-02 how-exclamatives; EXC-03 verbless/phrasal exclamatives (*Nice one! The cheek of it!* — GSWE 7.6.4, 14.3.4.2; C&M 538).

### QUO — Conversational reporting & quotatives (5)
REP covers canonical direct/indirect reporting thoroughly, but the entire conversational reporting repertoire of GSWE §14.4.4 is absent: QUO-01 quotative *go*; QUO-02 quotative *be like / be all*; QUO-03 conversational historic-present reporting (*so he says…*; extends REP-11, whose current definition is news/summary register only); QUO-04 past-progressive reporting frame (*she was saying…*); QUO-05 free direct and free indirect speech/thought (C&M speech-representation chapter — the one written-narrative item in this set).

### SIT — Situational-ellipsis subtypes (4)
ELS-11 exists as one construct; GSWE §14.3.5 and C&M §94 both subclassify, and the subtypes plausibly differ in distribution. SIT-01 initial subject ellipsis (*Doesn't matter*); SIT-02 subject+operator ellipsis (*Want a coffee?*); SIT-03 operator-only medial ellipsis (*You seen John?*); SIT-04 other situational fragments — copula/determiner/preposition drop (*Ready? Good film, that.* — C&M 94e–m). ELS-11 becomes the parent (or is retired with an xref).

### ISB — Insubordination (3)
GSWE §3.14 "Unembedded dependent clauses": ISB-01 stand-alone if-clauses as polite directives (*If you could just sign here.*); ISB-02 stand-alone because/cos answers; ISB-03 stand-alone *which*-comments (*…Which was nice.*). CON-14 (implied conditionals) is about elliptical apodoses, not freestanding protases, so this is a genuine gap.

### PSC — Pseudo-coordination & phrasal intensification (3)
PSC-01 *try and* + V (GSWE 9.4.11 gives it its own section); PSC-02 *go/come and* + V (C&M 532c); PSC-03 *nice and / good and* + Adj (GSWE 7.10.2). All three are formally coordination but functionally single predicates/intensification — exactly the kind of item a written-norm inventory misses.

### VER — Vernacular grammar, flagged stratum (7)
GSWE §14.4.5 and §3.8–3.9; C&M §119 and the NAmE appendix. For your BLC purposes this stratum is **not optional decoration**: education-stratified spoken data will contain these forms, and detectors that don't recognize them will either crash or — worse — mis-score low-education speakers' fully systematic grammar as absence of the standard construct. Each carries a `vernacular` flag: VER-01 negative concord; VER-02 *ain't*; VER-03 non-standard concord (*we was, he don't*); VER-04 demonstrative *them*; VER-05 leveled verb forms (*he seen it, she done it*); VER-06 reduced semi-modals *gonna/wanna/gotta* as transcribed; VER-07 double comparatives (*more better*, GSWE 7.7.5).

### DYS — Performance phenomena, optional annotation layer (4)
GSWE §14.2 (repeats, retrace-and-repair, incomplete utterances, syntactic blends). These are performance phenomena, not competence constructs, and the kit's use-mode assumes GEC-corrected text — so they don't belong in the 595-style inventory proper. They are included as a clearly flagged optional layer because your planned spoken-corpus pipeline will need them: dysfluency regions must be delimited before construct detectors run on speech transcripts, and dysfluency rates are themselves of interest in the BLC/aging literature. Adopt or drop as a block.

---

## 2. Additions to existing categories (11)

| ID | Construct | Example | Source | Why it's a gap |
|---|---|---|---|---|
| VCP-41 | V + *for* + NP + to-infinitive | *We arranged for him to travel.* | GSWE 9.4.2.5 | VCP has object-control (VCP-29) but no for…to frame |
| MOD-38 | *be bound to / be sure to* | *It's bound to rain.* | C&M 404c | Epistemic-certainty periphrastics absent |
| MOD-39 | *be likely to / be unlikely to* | *They're likely to win.* | GSWE 9.4.9.4; C&M 404e | Only appears as an ADJ-16 example, not a modal construct |
| MOD-40 | *dare* (modal & blend) | *I daren't ask. How dare you!* | C&M 396; GSWE 3.8.2.3 | Absent entirely |
| NOU-25 | Appositive noun phrase | *my brother, a doctor* | GSWE 8.10; C&M 173, 318 | Only appositive *that*-clauses (NCL-03) exist |
| NOU-26 | Plural noun as premodifier | *drugs policy, arms race* | GSWE 8.3.2 | NOU-22 asserts singular-only premodification |
| ADJ-21 | Adj + *for* + NP + to-inf | *It's important for us to leave.* | GSWE 9.4.5 | Same for…to gap on the adjective side |
| NEG-14 | Independent double negation | *You can't not go.* | GSWE 3.8.7.2 | NEG-12 (litotes) is the nearest but distinct |
| NCL-13 | Noun + wh-complement | *the question whether to stay* | GSWE 8.14.4 | NCL covers noun + that only |
| NCL-14 | Noun + *of* + -ing complement | *the possibility of going* | GSWE 8.14.3 | Currently only implicit in PREP-20/21; xref'd |
| FOC-18 | Thing-is focus formula | *The thing is, we're broke.* | C&M 475e; GSWE 14.3.2.1 | Related to but distinct from wh-clefts (FOC-04) |

---

## 3. Checked and NOT gaps (so you don't re-litigate them)

*would you mind + -ing* (inside MOD-09); imperative *will you* / *let's… shall we* / *aren't I* tags (QUE-12); headers/left-dislocation and NP tails/right-dislocation (FOC-17); get-passive, causatives, *need/want + -ing* passives (PAS-15–20); middle voice (VCP-06, = C&M 475i pseudo-intransitives); *I was wondering* politeness (VTA-28); echo questions (QUE-17); *how come / what if / how about* (QUE-16); *why don't you* (MOD-28); situational ellipsis as a phenomenon (ELS-11 — subtyped above, not missing); linking-adjunct semantics (COH-01–10 map cleanly onto C&M §136b–j); comparative correlatives and incremental comparatives (COM-13/14); *used to / be used to* (VSP-01–03); concord complications (CLS-08–12); *that*-omission (NCL-05); preposition stranding/pied-piping (PREP-22/23, REL-11/12); reversed pseudo-clefts (FOC-05); existential expansions (CLS-13/14).

## 4. Deliberately excluded (documented scope decisions)

1. **Word formation** (C&M §258–268; GSWE §4.8, §5.2.7, §7.9.2): derivation, conversion, clipping, blending. Lexical morphology, not syntactic constructions; adding it would change the inventory's construct definition. Compounding is already covered where syntactic (NOU-21, ADJ-13).
2. **Lexical bundles / word clusters** (GSWE ch. 13; C&M §503–505): frequency-defined recurrent n-grams, not rule-detectable constructions. Highly relevant to your BLC agenda but methodologically a separate, corpus-driven stratum — recommend a parallel n-gram pipeline rather than construct rows. (The one bundle-adjacent items promoted to constructs: general extenders VAG-01, thing-is FOC-18, binomial intensifier PSC-03.)
3. **Appendices**: punctuation, spelling, numbers/time/measurement conventions, bibliographic style (C&M §506–528; GSWE contractions appendix) — orthographic/encyclopedic, not grammatical.
4. **Prosody-dependent distinctions** (question intonation, tag intonation, emphatic stress): flagged inside affected constructs (QIN-01) rather than added as constructs, since text transcripts carry only punctuation proxies.
5. **Internet discourse** (C&M §122) and register-specific style notes (academic titles etc.): register description, not constructions.
6. **North American variant forms** (*gotten*, AmE *shall* avoidance, C&M §530–538): dialect distribution facts about constructs already in the inventory; better handled as a dialect attribute than as rows. Exception: items promoted into VER where they pattern with vernacular grammar generally.

## 5. Integration notes for the kit

- **ID scheme**: all new IDs continue existing category numbering; the 13 new categories use fresh prefixes (INS, VAG, CMT, VOC, DMG, QIN, EXC, QUO, SIT, ISB, PSC, VER, DYS) under proposed Part VI.
- **Refactors rather than pure additions**: COH-11 → parent of DMG-01…09 (keep as residual or retire); ELS-11 → parent of SIT-01…04; REP-11's definition should widen or xref QUO-03; FOC-15's definition should xref INS-09.
- **Detector tiers**: the new stratum is lexicon-heavy (INS, VOC-02/03, DMG, VAG-01/02) — extend `lexicons.json` with insert/DM/vocative/extender lists rather than writing rules. The genuinely hard (llm-tier) items: discourse *like* (DMG-05), declarative questions (QIN-01), follow-up questions (QIN-04), free indirect speech (QUO-05), syntactic blends (DYS-04).
- **Pipeline assumption to revisit**: the kit's use-mode assumes GEC-corrected, well-formed input. For your planned education-stratified spoken corpora, GEC upstream would *destroy the signal* — it would "correct" precisely the vernacular (VER) and dysfluency (DYS) phenomena you need to observe, and normalize situational ellipsis. Recommend a speech-mode pipeline variant: no GEC; DYS delimitation first; VER detectors run alongside their standard counterparts with variant linking (e.g. VER-03 hit suppresses a CLS-08 miss).
- **Contract tests**: each new construct in the JSON ships with one positive example; the detectors.json contract seeding (~1.5 pos / ~3.8 neg) should be extended the same way, with negatives drawn from the xref'd sibling constructs (e.g. INS-05 negatives from QUE-03; DMG-05 negatives from COM-16 and QUO-02).

## 6. Resulting inventory size

595 existing + 64 (Part VI) + 11 (additions) = **670 constructs** (666 competence constructs + 4 optional performance-layer items).
