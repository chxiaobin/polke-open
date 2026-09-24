# Construction inventory

POLKE annotates **670 grammatical constructions** (inventory schema 0.2). This reference is generated from `polke/data/constructs.json` by `scripts/generate_constructions_doc.py` — do not edit it by hand.

The **CEFR** column is the level assigned to each construction in `polke/data/cefr_levels.csv` (EGP / CEFR-J-informed estimates used by the level verifier, `polke level` and `/verify`; edit the CSV to recalibrate). The vernacular and dysfluency strata are not level-bearing (—).

Each construction has a stable ID (`CATEGORY-NN`) usable in the CLI (`polke annotate -c PAS,REL-01`), the HTTP API (`constructions` field), and the Python API (`selection=`). Passing a bare category code selects every construction in that category.

## Detector tiers

| tier | mechanism | needs LLM | constructions |
|---|---|---|---|
| `rule` | spaCy `DependencyMatcher` patterns | no | 181 |
| `lexicon` | dependency pattern gated by curated lexicons | no | 209 |
| `rule+LLM` | rule proposes a span, LLM picks the reading | yes | 209 |
| `lexicon+LLM` | lexicon proposes, LLM picks the reading | yes | 6 |
| `LLM` | LLM judges presence over sentence spans | yes | 65 |

## Contents

- **I. Verb Phrase**
  - [VTA — Tense and aspect](#vta--tense-and-aspect) (49)
  - [FUT — Future reference](#fut--future-reference) (21)
  - [MOD — Modality](#mod--modality) (40)
  - [PAS — Voice (passives)](#pas--voice-passives) (20)
  - [NFV — Non-finite verb forms (infinitives & -ing)](#nfv--non-finite-verb-forms-infinitives---ing) (31)
  - [VCP — Verb-complementation patterns (full)](#vcp--verb-complementation-patterns-full) (41)
  - [PHV — Phrasal & prepositional verbs](#phv--phrasal--prepositional-verbs) (7)
  - [VSP — Special verb-meaning areas](#vsp--special-verb-meaning-areas) (13)
- **II. Noun Phrase**
  - [NOU — Nouns](#nou--nouns) (26)
  - [ART — Articles](#art--articles) (24)
  - [DET — Determiners & quantifiers](#det--determiners--quantifiers) (28)
  - [PRO — Pronouns](#pro--pronouns) (15)
  - [ADJ — Adjectives](#adj--adjectives) (21)
  - [ADV — Adverbs & adverbials](#adv--adverbs--adverbials) (29)
  - [COM — Comparison](#com--comparison) (18)
  - [PREP — Prepositions](#prep--prepositions) (25)
- **III. Clause & Sentence**
  - [CLS — Basic clause patterns](#cls--basic-clause-patterns) (17)
  - [QUE — Questions](#que--questions) (17)
  - [NEG — Negation](#neg--negation) (14)
  - [IMP — Imperatives & directives](#imp--imperatives--directives) (9)
- **IV. Connecting Clauses**
  - [COORD — Coordination](#coord--coordination) (9)
  - [ASC — Adverbial subordinate clauses](#asc--adverbial-subordinate-clauses) (29)
  - [CON — Conditionals](#con--conditionals) (17)
  - [REL — Relative clauses](#rel--relative-clauses) (18)
  - [NCL — Noun / complement clauses](#ncl--noun--complement-clauses) (14)
  - [REP — Reported speech](#rep--reported-speech) (11)
- **V. Discourse & Information Structure**
  - [FOC — Focus, fronting, inversion, emphasis](#foc--focus-fronting-inversion-emphasis) (18)
  - [ELS — Ellipsis & substitution](#els--ellipsis--substitution) (11)
  - [COH — Cohesion & linking adverbials](#coh--cohesion--linking-adverbials) (14)
- **VI. Spoken & Interactional Grammar**
  - [INS — Inserts (freestanding interactional forms)](#ins--inserts-freestanding-interactional-forms) (9)
  - [VAG — Vague language & approximation](#vag--vague-language--approximation) (4)
  - [CMT — Comment clauses & parentheticals](#cmt--comment-clauses--parentheticals) (4)
  - [VOC — Vocatives](#voc--vocatives) (3)
  - [DMG — Discourse markers (item grain)](#dmg--discourse-markers-item-grain) (9)
  - [QIN — Interactional questions & tags](#qin--interactional-questions--tags) (6)
  - [EXC — Exclamatives](#exc--exclamatives) (3)
  - [QUO — Conversational reporting & quotatives](#quo--conversational-reporting--quotatives) (5)
  - [SIT — Situational ellipsis (subtypes)](#sit--situational-ellipsis-subtypes) (4)
  - [ISB — Insubordination (freestanding subordinate clauses)](#isb--insubordination-freestanding-subordinate-clauses) (3)
  - [PSC — Pseudo-coordination & phrasal intensification](#psc--pseudo-coordination--phrasal-intensification) (3)
  - [VER — Vernacular grammar (flagged stratum)](#ver--vernacular-grammar-flagged-stratum) (7)
  - [DYS — Performance phenomena (optional annotation layer)](#dys--performance-phenomena-optional-annotation-layer) (4)

## I. Verb Phrase

### VTA — Tense and aspect

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| VTA-01 | A1 | Present simple | General/timeless truth | Water boils at 100°C. | rule+LLM |
| VTA-02 | A1 | Present simple | Permanent state | She comes from Spain. I live in Berlin. | rule+LLM |
| VTA-03 | A1 | Present simple | Habit / routine — often with frequency adverbs. | I get up at seven. | rule+LLM |
| VTA-04 | A2 | Present simple | Instructions / directions | You take the second left. | rule+LLM |
| VTA-05 | B2 | Present simple | Commentary / demonstration | He passes, he shoots, he scores. | LLM |
| VTA-06 | B2 | Present simple | Narrative / historic present | So I walk in and he just stares at me. | LLM |
| VTA-07 | B1 | Present simple | Performative | I promise I'll pay. I apologise. | lexicon |
| VTA-08 | B2 | Present simple | Headlines / captions | Minister resigns. | LLM |
| VTA-09 | A2 | Present simple | Scheduled future | The train leaves at six. | LLM |
| VTA-10 | A2 | Present simple | Present in time/conditional clauses (future ref) | When he arrives, we'll start. | rule+LLM |
| VTA-11 | A1 | Present continuous | Action in progress now | I'm reading your draft. | rule |
| VTA-12 | A2 | Present continuous | Temporary situation | She's staying with us this week. | rule+LLM |
| VTA-13 | B1 | Present continuous | Changing / developing situation | The climate is getting warmer. | rule+LLM |
| VTA-14 | B2 | Present continuous | Repeated action + annoyance — with *always/forever/constantly*. | You're always interrupting! | rule+LLM |
| VTA-15 | A2 | Present continuous | Fixed future arrangement | We're meeting at six. | LLM |
| VTA-16 | B1 | Present continuous | Background in present narrative | It's raining and people are running for cover. | LLM |
| VTA-17 | A1 | Past simple | Completed single past event | We arrived late. | rule |
| VTA-18 | A2 | Past simple | Sequence of events (narrative) | She opened the door, looked in and left. | rule |
| VTA-19 | A2 | Past simple | Past habit / repeated action | I walked to school every day. | rule+LLM |
| VTA-20 | A1 | Past simple | Past state | He had long hair then. I knew her well. | rule |
| VTA-21 | B2 | Past simple | Remote / polite (tentative) | I wondered if you could help. Did you want anything? | LLM |
| VTA-22 | B1 | Past simple | Hypothetical/unreal past tense form | If I knew...; I wish I had...; It's time we left. | rule+LLM |
| VTA-23 | A2 | Past continuous | Action in progress at a past point | At nine I was working. | rule |
| VTA-24 | B1 | Past continuous | Interrupted action | I was cooking when the phone rang. | rule+LLM |
| VTA-25 | B1 | Past continuous | Background to past events | The sun was shining when we set off. | rule+LLM |
| VTA-26 | B1 | Past continuous | Parallel past actions | While I cooked, she was setting the table. | rule |
| VTA-27 | B2 | Past continuous | Temporary past situation / repeated annoyance | He was always losing his keys. | rule+LLM |
| VTA-28 | B2 | Past continuous | Polite / tentative | I was wondering whether... | LLM |
| VTA-29 | A2 | Present perfect simple | Experience (indefinite past) — with *ever/never/before*. | I've been to Japan. Have you ever tried sushi? | rule+LLM |
| VTA-30 | B1 | Present perfect simple | Unfinished time-span up to now | I've lived here since 2010 / for ten years. | rule+LLM |
| VTA-31 | A2 | Present perfect simple | Recent event with present result — with *just/already/yet*. | I've lost my keys. She's just left. | rule+LLM |
| VTA-32 | B1 | Present perfect simple | Resultative state change | He's broken his leg | LLM |
| VTA-33 | B1 | Present perfect simple | With superlative / first/last/only | It's the best film I've ever seen. This is the first time I've flown. | rule+LLM |
| VTA-34 | B1 | Present perfect simple | News reporting / announcement | Scientists have discovered a new species. | LLM |
| VTA-35 | B1 | Present perfect continuous | Duration of activity up to now — with *for/since/how long*. | I've been waiting for an hour. | rule+LLM |
| VTA-36 | B2 | Present perfect continuous | Recent activity with present evidence. Your eyes are red | Your eyes are red — have you been crying? | LLM |
| VTA-37 | B2 | Present perfect continuous | Temporary / repeated recent activity | I've been going to the gym lately. | rule+LLM |
| VTA-38 | B2 | Present perfect continuous | Activity-focus vs result-focus (contrast w/ simple) | I've been painting (activity) / I've painted the door (result). | LLM |
| VTA-39 | B1 | Past perfect simple | Earlier past (anterior to a past point) | The train had left before we arrived. | rule |
| VTA-40 | B1 | Past perfect simple | With by the time / before / after / when | By the time we got there, it had closed. | rule |
| VTA-41 | B1 | Past perfect simple | Backshift in reported speech | She said she had finished. | rule |
| VTA-42 | B2 | Past perfect simple | In third/mixed conditionals | If you had asked... | rule |
| VTA-43 | C1 | Past perfect simple | Unfulfilled hopes/intentions | I had hoped to see you. We had intended to leave earlier. | rule+LLM |
| VTA-44 | B2 | Past perfect continuous | Duration up to a past point | She'd been working there for years before she quit. | rule |
| VTA-45 | B2 | Past perfect continuous | Cause of a past state/result | His eyes were red; he'd been crying. | LLM |
| VTA-46 | A2 | Aspect & lexical-aspect constraints | Stative verbs resisting the progressive — cognition/perception/possession/emotion. | I know / | LLM |
| VTA-47 | B1 | Aspect & lexical-aspect constraints | Perception verbs: see/hear vs look/listen/watch | I see it / I'm looking at it. | LLM |
| VTA-48 | C1 | Aspect & lexical-aspect constraints | Coercion of statives into progressive | I'm loving it. You're being silly. | LLM |
| VTA-49 | B1 | Aspect & lexical-aspect constraints | Sequence of tenses in subordinate clauses | He said (that) he was tired. | rule+LLM |

### FUT — Future reference

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| FUT-01 | A2 | *will* | Prediction / future fact | It'll rain tomorrow. | rule+LLM |
| FUT-02 | A2 | *will* | Spontaneous decision (at moment of speaking) | I'll get it! | LLM |
| FUT-03 | A2 | *will* | Offer | I'll carry that for you. | LLM |
| FUT-04 | A2 | *will* | Promise / threat | I'll always love you. You'll regret this. | LLM |
| FUT-05 | B1 | *will* | Willingness / refusal (won't) | She won't answer the phone. | rule+LLM |
| FUT-06 | B1 | *will* | Request (Will you | Will you close the door? | rule+LLM |
| FUT-07 | B2 | *will* | Habitual/characteristic will | He'll sit there for hours. | LLM |
| FUT-08 | A2 | *be going to* | Prior intention/plan | I'm going to apply for the job. | rule+LLM |
| FUT-09 | A2 | *be going to* | Prediction from present evidence. Look out | Look out — it's going to fall! | rule+LLM |
| FUT-10 | B1 | *be going to* | was/were going to (unfulfilled intention) | I was going to call but forgot. | rule+LLM |
| FUT-11 | A2 | Present forms for future | Present continuous for arrangement | We're flying on Monday. | LLM |
| FUT-12 | A2 | Present forms for future | Present simple for timetable | The show starts at eight. | LLM |
| FUT-13 | A2 | Present forms for future | Present simple after time/conditional conjunctions | I'll call when I get there. | rule+LLM |
| FUT-14 | B2 | Future progressive & perfect | Future continuous (in progress at future point) | This time tomorrow I'll be flying. | rule |
| FUT-15 | C1 | Future progressive & perfect | Future continuous (future as a matter of course / polite) | Will you be using the car tonight? | LLM |
| FUT-16 | B2 | Future progressive & perfect | Future perfect simple | By June I'll have finished. | rule |
| FUT-17 | C1 | Future progressive & perfect | Future perfect continuous | By 5 I'll have been driving for ten hours. | rule |
| FUT-18 | C1 | Other future frames | be to (formal arrangement/instruction) | The President is to visit Berlin. | lexicon |
| FUT-19 | B1 | Other future frames | be about to (imminence) | We're about to start. | lexicon |
| FUT-20 | B2 | Other future frames | be due to / be set to / be on the point of / be on the verge of | The flight is due to land at noon. | lexicon |
| FUT-21 | B2 | Other future frames | Future in the past (would / was to / was about to) | He didn't know he would never return. | rule+LLM |

### MOD — Modality

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| MOD-01 | A1 | Ability | can | I can swim. | rule+LLM |
| MOD-02 | A2 | Ability | could | I could read at four. | rule+LLM |
| MOD-03 | B1 | Ability | was/were able to / managed to | We were able to escape. | lexicon |
| MOD-04 | B1 | Ability | be able to | I haven't been able to reach her. | lexicon |
| MOD-05 | B2 | Ability | could have | I could have gone, but I didn't. | rule+LLM |
| MOD-06 | A1 | Permission | can | You can use my phone. | rule+LLM |
| MOD-07 | A2 | Permission | could / may / might | May I leave early? | rule+LLM |
| MOD-08 | B1 | Permission | be allowed to / be permitted to | We weren't allowed to enter. | lexicon |
| MOD-09 | A2 | Requests / offers / suggestions | Requests: can/could/will/would you | Could you help me? | rule+LLM |
| MOD-10 | A2 | Requests / offers / suggestions | Offers/suggestions: shall I/we; why don't we; how about; let's | Shall I open the window? | rule+LLM |
| MOD-11 | B1 | Possibility (epistemic) | may / might / could | It may snow. They might be late. | rule+LLM |
| MOD-12 | B2 | Possibility (epistemic) | may / might / could have | She may have missed the train. | rule+LLM |
| MOD-13 | B1 | Possibility (epistemic) | can | It can get very cold here. | rule+LLM |
| MOD-14 | B1 | Possibility (epistemic) | could / might | You could try restarting it. | LLM |
| MOD-15 | A2 | Obligation & necessity | must | I must finish this today. | rule+LLM |
| MOD-16 | A2 | Obligation & necessity | have to / have got to | I have to renew my passport. | lexicon |
| MOD-17 | A2 | Obligation & necessity | need to | You need to sign here. | lexicon |
| MOD-18 | B2 | Obligation & necessity | be supposed to / be meant to | You're supposed to wear a tie. | lexicon |
| MOD-19 | C1 | Obligation & necessity | be required to / be obliged to | Staff are required to log in. | lexicon |
| MOD-20 | C2 | Obligation & necessity | shall | Tenants shall maintain the property. | rule+LLM |
| MOD-21 | A2 | Absence of necessity & prohibition | don't have to / needn't / don't need to | You don't have to come. | lexicon |
| MOD-22 | C1 | Absence of necessity & prohibition | needn't have + pp | You needn't have paid. | rule+LLM |
| MOD-23 | B2 | Absence of necessity & prohibition | didn't need to | I didn't need to wait. | LLM |
| MOD-24 | A2 | Absence of necessity & prohibition | mustn't / can't / not allowed to / may not | You mustn't smoke here. | rule+LLM |
| MOD-25 | A2 | Advice / recommendation | should / ought to | You should rest. | rule+LLM |
| MOD-26 | B1 | Advice / recommendation | had better | You'd better hurry. | lexicon |
| MOD-27 | B2 | Advice / recommendation | should/ought to/could/might have + pp | You should have told me. | rule+LLM |
| MOD-28 | A2 | Advice / recommendation | why don't you / why not / it might be a good idea to | Why not ask her? | rule+LLM |
| MOD-29 | B1 | Deduction / certainty (epistemic) | must | That must be the postman. | rule+LLM |
| MOD-30 | B1 | Deduction / certainty (epistemic) | can't / couldn't | He can't be serious. | rule+LLM |
| MOD-31 | B2 | Deduction / certainty (epistemic) | must have / can't have + pp | They must have left already. | rule+LLM |
| MOD-32 | B2 | Deduction / certainty (epistemic) | should / ought to | The parcel should arrive today. | rule+LLM |
| MOD-33 | C1 | Deduction / certainty (epistemic) | will / would | That'll be the courier. He'd be about fifty now. | LLM |
| MOD-34 | B2 | Volition, habit, preference | will / won't | The door won't open. | rule+LLM |
| MOD-35 | B1 | Volition, habit, preference | would | On Sundays we would visit grandma. | rule+LLM |
| MOD-36 | C1 | Volition, habit, preference | will | Oil will float on water. | LLM |
| MOD-37 | B1 | Volition, habit, preference | would rather / would sooner / would prefer | would rather / would sooner / would prefer | lexicon |
| MOD-38 | B2 | Deduction / certainty (epistemic) | be bound to / be sure to | It's bound to rain; she's sure to notice. | lexicon |
| MOD-39 | B1 | Possibility (epistemic) | be likely to / be unlikely to (+ extraposed it is likely that) | They're likely to win; it's unlikely that he knew. | lexicon |
| MOD-40 | C1 | Obligation & necessity | dare (modal & blend uses) | I daren't ask. How dare you! Dare I say it. | lexicon |

### PAS — Voice (passives)

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| PAS-01 | A2 | Passive across the paradigm | Present simple passive | The form is signed here. | rule |
| PAS-02 | A2 | Passive across the paradigm | Past simple passive | The bridge was built in 1890. | rule |
| PAS-03 | B1 | Passive across the paradigm | Present continuous passive | The road is being repaired. | rule |
| PAS-04 | B2 | Passive across the paradigm | Past continuous passive | The house was being painted. | rule |
| PAS-05 | B1 | Passive across the paradigm | Present perfect passive | The report has been approved. | rule |
| PAS-06 | B2 | Passive across the paradigm | Past perfect passive | It had been agreed beforehand. | rule |
| PAS-07 | B1 | Passive across the paradigm | Future / going to passive | The results will be published soon. | rule |
| PAS-08 | B1 | Passive across the paradigm | Modal passive | It must be submitted by Friday. | rule |
| PAS-09 | B2 | Passive across the paradigm | Perfect modal passive | It should have been checked. | rule |
| PAS-10 | B2 | Passive across the paradigm | Infinitive/-ing passive | It needs to be done. I hate being interrupted. | rule |
| PAS-11 | B1 | Agents, recipients, special passives | by-agent phrase | The novel was written by a teenager. | rule |
| PAS-12 | B1 | Agents, recipients, special passives | with-instrument / of-material | It was filled with water. | rule+LLM |
| PAS-13 | B2 | Agents, recipients, special passives | Ditransitive passive (two patterns) | I was given a prize. / A prize was given to me. | rule |
| PAS-14 | B2 | Agents, recipients, special passives | Passive of phrasal/prepositional verbs | The meeting was called off. She was looked after. | lexicon |
| PAS-15 | B2 | Agents, recipients, special passives | Get-passive (dynamic/adversative/fortuitous) | He got promoted. The vase got broken. | rule+LLM |
| PAS-16 | B2 | Agents, recipients, special passives | Passive of reporting verbs (It is said that | It is believed that he fled. | lexicon |
| PAS-17 | C1 | Agents, recipients, special passives | Subject-raised reporting passive (S is said to | The suspect is thought to have fled. | lexicon |
| PAS-18 | B2 | Agents, recipients, special passives | have/get something done (causative) | I had my visa renewed. | rule+LLM |
| PAS-19 | C1 | Agents, recipients, special passives | Adversative causative | He had his car stolen. | LLM |
| PAS-20 | B2 | Agents, recipients, special passives | need / want / require + -ing (passive meaning) | The plants need watering. | lexicon |

### NFV — Non-finite verb forms (infinitives & -ing)

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| NFV-01 | A2 | To-infinitive | Purpose (to / in order to / so as to) | I came to help. | rule+LLM |
| NFV-02 | B2 | To-infinitive | Negative purpose (in order not to / so as not to) | He left early so as not to miss it. | rule |
| NFV-03 | A1 | To-infinitive | After certain verbs (subject-control) | I decided to leave. She hopes to win. | lexicon |
| NFV-04 | A2 | To-infinitive | After adjectives | easy to please; glad to help; likely to rain. | lexicon |
| NFV-05 | B1 | To-infinitive | After nouns | a decision to resign; the need to act. | lexicon |
| NFV-06 | B1 | To-infinitive | After too / enough | too tired to walk; old enough to vote. | rule |
| NFV-07 | B1 | To-infinitive | After wh-words | I don't know what to do / where to go. | rule |
| NFV-08 | C1 | To-infinitive | Infinitive of result/outcome | He awoke to find the house empty. only to fail. | LLM |
| NFV-09 | A1 | Bare infinitive | After modals | You must go. | rule |
| NFV-10 | A2 | Bare infinitive | After let / make (+ object) | Let it go. They made him wait. | lexicon |
| NFV-11 | B1 | Bare infinitive | After perception verbs (completed action) | I saw her leave. | rule+LLM |
| NFV-12 | B1 | Bare infinitive | After had better / would rather / why (not) | You'd better stop. Why wait? | lexicon |
| NFV-13 | C1 | Bare infinitive | After rather than / sooner than / but / except | Rather than wait, she left. | lexicon |
| NFV-14 | B1 | Bare infinitive | help (+ to) + infinitive | This will help (to) reduce costs. | lexicon |
| NFV-15 | A2 | Gerund (-ing as noun) | Gerund as subject | Swimming is good for you. | rule+LLM |
| NFV-16 | A1 | Gerund (-ing as noun) | Gerund as object of verb | I enjoy reading. | lexicon |
| NFV-17 | A2 | Gerund (-ing as noun) | Gerund after preposition | good at cooking; before leaving; instead of waiting. | lexicon |
| NFV-18 | C1 | Gerund (-ing as noun) | Gerund after possessive/genitive | I appreciate your helping / his leaving. | rule+LLM |
| NFV-19 | B2 | Gerund (-ing as noun) | Gerund as complement of be | Seeing is believing. | rule+LLM |
| NFV-20 | B2 | Gerund (-ing as noun) | Gerund in fixed expressions | It's no use crying; it's worth trying; there's no point (in) waiting; have difficulty (in) sleeping; spend/waste time reading; be busy working. | lexicon |
| NFV-21 | A2 | Infinitive vs -ing | Verb selects infinitive (no choice) | want/decide/hope/manage/refuse + to. | lexicon |
| NFV-22 | B1 | Infinitive vs -ing | Verb selects -ing (no choice) | avoid/admit/consider/deny/finish/suggest/risk + -ing. | lexicon |
| NFV-23 | B1 | Infinitive vs -ing | Both, little change | begin/start/continue/like/love/hate/prefer + to/-ing. | lexicon |
| NFV-24 | B2 | Infinitive vs -ing | Both, meaning change — *stop to smoke ≠ stop smoking.* | stop/remember/forget/try/regret/go on/mean/need + to vs -ing. | lexicon+LLM |
| NFV-25 | B2 | Infinitive vs -ing | like + inf vs -ing nuance | I like to check (choose to) / I like checking (enjoy). | LLM |
| NFV-26 | B2 | Perfect, passive & participial non-finites | Perfect infinitive/gerund | to have finished; having spoken. | rule |
| NFV-27 | B2 | Perfect, passive & participial non-finites | Passive infinitive/gerund | to be told; being watched. | rule |
| NFV-28 | B2 | Perfect, passive & participial non-finites | Present participle clause (simultaneous/active) | Smiling, she opened the door. | rule+LLM |
| NFV-29 | B2 | Perfect, passive & participial non-finites | Past participle clause (passive) | Asked to leave, he refused. | rule+LLM |
| NFV-30 | C1 | Perfect, passive & participial non-finites | Perfect participle clause (anterior) | Having finished, she left. | rule |
| NFV-31 | A2 | Perfect, passive & participial non-finites | would like/love/hate/prefer + to-infinitive (specific occasion / polite) vs -ing (general) | I'd love to come tonight; I love travelling. | rule+LLM |

### VCP — Verb-complementation patterns (full)

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| VCP-01 | A1 | Intransitive frames | V (pure intransitive) | She smiled. The baby slept. | rule |
| VCP-02 | A1 | Intransitive frames | V + obligatory adverbial (place/direction/measure/duration) | He lives abroad; it lasted an hour; she headed home. | rule |
| VCP-03 | A2 | Intransitive frames | V + subject predicative | She seems tired. | lexicon |
| VCP-04 | A2 | Intransitive frames | V + subject predicative | He became a doctor; it went sour. | lexicon |
| VCP-05 | B1 | Intransitive frames | Ergative / inchoative alternation (same verb intr/tr) | The door opened / She opened the door; the glass broke; the price increased. | rule+LLM |
| VCP-06 | C1 | Intransitive frames | Middle voice (patient-subject, generic) | This shirt washes easily; the book sells well; the door won't lock. | rule+LLM |
| VCP-07 | B1 | Intransitive frames | Reciprocal intransitive (plural/conjoined subject) | They met; we argued; the two lines intersect. | rule+LLM |
| VCP-08 | A1 | Monotransitive frames | V + NP (object) | She read the report. | rule |
| VCP-09 | A2 | Monotransitive frames | V + reflexive pronoun | He hurt himself; behave yourself; help yourself. | lexicon |
| VCP-10 | B1 | Monotransitive frames | V + reciprocal object | They met each other; we know one another. | rule |
| VCP-11 | A2 | Monotransitive frames | V + that-clause | I think (that) it works. | lexicon |
| VCP-12 | C1 | Monotransitive frames | V + (that) + mandative subjunctive | I insist that he be present. | lexicon |
| VCP-13 | B1 | Monotransitive frames | V + wh-clause | I wonder what happened. | rule |
| VCP-14 | B1 | Monotransitive frames | V + wh + to-infinitive | I don't know what to do. | rule |
| VCP-15 | A2 | Monotransitive frames | V + to-infinitive (subject control) | She agreed to help. | lexicon |
| VCP-16 | A2 | Monotransitive frames | V + -ing | She admitted taking it. | lexicon |
| VCP-17 | B1 | Monotransitive frames | V + to-inf OR -ing, ~no change | It started to rain / raining. | lexicon |
| VCP-18 | B2 | Monotransitive frames | V + to-inf vs -ing, meaning change | stop to rest ≠ stop resting. | lexicon+LLM |
| VCP-19 | A2 | Monotransitive frames | V + prep + NP (prepositional verb) | It depends on the weather. | lexicon |
| VCP-20 | B1 | Monotransitive frames | V + prep + -ing | She succeeded in passing. | lexicon |
| VCP-21 | A2 | Ditransitive frames | V + IO + DO (double object) | She gave him a book. | rule |
| VCP-22 | A2 | Ditransitive frames | V + DO + to-NP (dative shift) | give a book to him; explain it to me | lexicon |
| VCP-23 | B1 | Ditransitive frames | V + DO + for-NP (benefactive) | buy/make/cook/find/get/save a gift for her. | lexicon |
| VCP-24 | B1 | Ditransitive frames | V + IO + that-clause | He assured me that it was safe. | lexicon |
| VCP-25 | B1 | Ditransitive frames | V + IO + wh-clause | Tell me what you need. | rule |
| VCP-26 | B1 | Complex-transitive frames | V + O + adjective (object predicative) | paint it red; keep it clean; find it useful; drive sb crazy; cut it short. | rule |
| VCP-27 | B2 | Complex-transitive frames | V + O + NP (object predicative) | call him a genius; elect her chair; name it X; consider it a mistake; appoint sb director. | rule |
| VCP-28 | B2 | Complex-transitive frames | V + O + as-phrase | They regard her as an expert. | lexicon |
| VCP-29 | B1 | Complex-transitive frames | V + O + to-infinitive (object control / ECM) | I want you to stay. | lexicon |
| VCP-30 | B1 | Complex-transitive frames | V + O + bare infinitive (causative) | They made him apologise; let it go. | lexicon |
| VCP-31 | B1 | Complex-transitive frames | V + O + bare infinitive (perception, completed) | I saw her leave. | rule+LLM |
| VCP-32 | B2 | Complex-transitive frames | V + O + -ing (perception/continuation, ongoing) | I caught him cheating; she kept me waiting. | rule+LLM |
| VCP-33 | B2 | Complex-transitive frames | V + O + past participle | I had it repaired; I want it finished. | rule+LLM |
| VCP-34 | B2 | Complex-transitive frames | V + O + prep + NP | They accused him of fraud. | lexicon |
| VCP-35 | B2 | Complex-transitive frames | V + anticipatory it + complement | I find it hard to concentrate; they made it clear that... | rule+LLM |
| VCP-36 | C1 | Other verb constructions | Existential/presentational verbs (there + V) | There remains much to do. | rule |
| VCP-37 | A2 | Other verb constructions | Light/delexical verb + NP | have a look/rest/shower; take a look/break/photo; make a decision/mistake; do the shopping; give a smile/sigh. | lexicon |
| VCP-38 | B1 | Other verb constructions | Verb idiom / fixed VP | take place, make sense, pay attention, take part, keep an eye on, bear in mind. | lexicon |
| VCP-39 | C2 | Other verb constructions | Cognate-object construction | live a good life; die a hero's death; smile a faint smile. | rule+LLM |
| VCP-40 | C1 | Other verb constructions | Catenative chains (stacked non-finites) | She seems to want to start trying to help. | rule |
| VCP-41 | B2 | Monotransitive frames | V + for + NP + to-infinitive | We arranged for him to travel; I'd hate for you to miss it. | rule |

### PHV — Phrasal & prepositional verbs

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| PHV-01 | A2 |  | Intransitive phrasal verb | The plane took off. Sit down. | lexicon |
| PHV-02 | A2 |  | Transitive separable phrasal verb | Turn off the light / turn the light off. | lexicon |
| PHV-03 | B1 |  | Obligatory separation with pronoun object | Turn it off | lexicon |
| PHV-04 | A2 |  | Prepositional verb (inseparable, transitive) | She looks after the kids. | lexicon |
| PHV-05 | B1 |  | Phrasal-prepositional verb (three-part) | I look forward to it. We've run out of milk. | lexicon |
| PHV-06 | B2 |  | Literal vs idiomatic meaning | take off (remove) vs take off (succeed/depart). | lexicon+LLM |
| PHV-07 | C1 |  | Aspectual particles (up completive, on/away continuative, out thoroughness) | drink up; carry on; work out. | lexicon |

### VSP — Special verb-meaning areas

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| VSP-01 | B1 |  | used to + base (discontinued habit/state) | I used to smoke. | lexicon |
| VSP-02 | B1 |  | used to negatives/questions | I didn't use to like it. Did you use to live here? | lexicon |
| VSP-03 | B2 |  | be/get used to + -ing/NP (familiarity) | I'm used to the noise; she's getting used to driving. | lexicon |
| VSP-04 | B2 |  | would rather/sooner/prefer (preference + tense rules) | I'd rather you didn't. I'd prefer to wait. | lexicon |
| VSP-05 | C1 |  | Mandative subjunctive | It's essential that she attend. I demand that he be present. | lexicon |
| VSP-06 | C2 |  | Formulaic subjunctive | God save the Queen; be that as it may; come what may; if need be. | lexicon |
| VSP-07 | B1 |  | were-subjunctive | If I were you; as it were; if I were to... | lexicon |
| VSP-08 | B1 |  | wish/if only + past (present regret) | I wish I knew. | rule+LLM |
| VSP-09 | B2 |  | wish/if only + past perfect (past regret) | If only I had asked. | rule+LLM |
| VSP-10 | B2 |  | wish + would (annoyance / desired change) | I wish you'd stop. | rule+LLM |
| VSP-11 | B1 |  | wish + could | I wish I could help. | rule+LLM |
| VSP-12 | B2 |  | It's (high) time + past | It's time we left. | rule+LLM |
| VSP-13 | A1 |  | have / have got (possession, states, relationships) — *have got* is informal and present-only; plain *have* takes *do*-support. | I've got a car / I have a car; she's got two sisters. | lexicon |

## II. Noun Phrase

### NOU — Nouns

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| NOU-01 | A1 | Countability | Countable nouns (sg/pl) | a chair / two chairs. | rule |
| NOU-02 | A2 | Countability | Uncountable nouns | advice, information, news, furniture, luggage, money, research. | LLM |
| NOU-03 | B2 | Countability | Nouns both count & uncount (meaning shift) | a coffee/coffee; experience(s); a paper/paper; a glass/glass. | LLM |
| NOU-04 | A2 | Countability | Pluralia tantum (only plural) | scissors, trousers, jeans, glasses, clothes, goods. | lexicon |
| NOU-05 | B1 | Countability | Uncount nouns looking plural / singular agreement | news, mathematics, physics, economics is... | lexicon |
| NOU-06 | B1 | Countability | Partitives for uncount | a piece of advice; a slice of bread; an item of news; a bit of. | lexicon |
| NOU-07 | B1 | Countability | Quantifying coercion | two coffees; three beers | LLM |
| NOU-08 | A1 | Number | Regular plurals (-s/-es/-y→-ies/-f→-ves) | books, boxes, cities, leaves. | rule |
| NOU-09 | A1 | Number | Irregular plurals | man/men, child/children, foot/feet, tooth, mouse, goose. | lexicon |
| NOU-10 | A2 | Number | Zero/invariant plurals | sheep, fish, deer, aircraft, series, species, means. | lexicon |
| NOU-11 | C1 | Number | Foreign/Latinate plurals | criterion/criteria, phenomenon/phenomena, datum/data, analysis/analyses, crisis/crises, index/indices. | lexicon |
| NOU-12 | C1 | Number | Compound-noun plurals | passers-by, mothers-in-law, runners-up. | lexicon |
| NOU-13 | A1 | Possession | Possessive 's (singular) | the dog's tail. | rule |
| NOU-14 | A2 | Possession | Possessive s' (regular plural) | the students' results. | rule |
| NOU-15 | A2 | Possession | Irregular-plural possessive ('s) | the children's books. | rule |
| NOU-16 | A2 | Possession | Of-genitive | the roof of the house; the end of the film. | rule |
| NOU-17 | B1 | Possession | Double genitive | a friend of mine / of John's. | rule+LLM |
| NOU-18 | B2 | Possession | Group genitive | the King of Spain's visit; someone else's idea. | rule+LLM |
| NOU-19 | B2 | Possession | Time/measure genitive | a day's work; two weeks' notice; a pound's worth. | lexicon |
| NOU-20 | B2 | Possession | Local/elliptical genitive | at the baker's; at my aunt's. | lexicon |
| NOU-21 | A2 | Modification & other | Noun + noun (compound) | a bus stop; coffee table; government policy. | rule+LLM |
| NOU-22 | B1 | Modification & other | Singular modifying noun | a three-hour meeting; a ten-pound note; a shoe shop. | rule+LLM |
| NOU-23 | B2 | Modification & other | Collective nouns & agreement — grammatical vs notional. | The team is/are winning. | LLM |
| NOU-24 | B1 | Modification & other | Nationality/group nouns | the French, the Dutch, the British. | lexicon |
| NOU-25 | B2 | Modification & other | Appositive noun phrase | my brother, a doctor; the novelist John le Carré; we teachers | rule |
| NOU-26 | C1 | Modification & other | Plural noun as premodifier | sports car; drugs policy; arms race | rule |

### ART — Articles

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| ART-01 | A1 | Indefinite *a/an* | First mention / non-specific | I saw a dog. | rule+LLM |
| ART-02 | A1 | Indefinite *a/an* | Classifying (occupation/category) | She's a teacher. It's a tool. | rule+LLM |
| ART-03 | A2 | Indefinite *a/an* | Per / rates | twice a day; 60 km an hour. | lexicon |
| ART-04 | A2 | Indefinite *a/an* | In exclamations | What a day! Such a mess! | rule |
| ART-05 | A1 | Indefinite *a/an* | a/an by sound | a university, an hour, an MP. | rule |
| ART-06 | A1 | Definite *the* | Anaphoric (second mention) | I bought a book. The book was great. | rule+LLM |
| ART-07 | A2 | Definite *the* | Cataphoric (post-modified) | the book on the table; the man who called. | rule+LLM |
| ART-08 | A2 | Definite *the* | Situational / shared knowledge | Could you shut the door? | LLM |
| ART-09 | A2 | Definite *the* | Unique referents | the sun, the moon, the government, the internet. | lexicon |
| ART-10 | A2 | Definite *the* | With superlative / ordinal / only/same/very | the best, the first, the only one. | rule |
| ART-11 | B2 | Definite *the* | Generic the + singular (class) | The tiger is endangered. The piano was invented... | LLM |
| ART-12 | A1 | Definite *the* | the + musical instrument / invention | play the piano. | lexicon |
| ART-13 | B1 | Definite *the* | the + adjective/nationality (group) | the rich, the unemployed, the Chinese. | rule+LLM |
| ART-14 | B1 | Definite *the* | the + decade/period | the 1990s, the Renaissance. | lexicon |
| ART-15 | A1 | Zero article | Generic plural/uncount | Ø Dogs bark. I like Ø music. | LLM |
| ART-16 | A1 | Zero article | Meals, transport, times | have Ø lunch; by Ø car; at Ø night; on Ø Monday. | lexicon |
| ART-17 | B1 | Zero article | Institutions (purpose sense) | the. | lexicon+LLM |
| ART-18 | A1 | Zero article | Languages/subjects/sports/games | study Ø biology; speak Ø French; play Ø tennis. | lexicon |
| ART-19 | B1 | Zero article | After kind/sort/type of | what kind of Ø person. | lexicon |
| ART-20 | A2 | Articles with names & places | Countries: Ø vs the (unions/plurals) | Ø France / the USA, the UK, the Netherlands, the Philippines. | lexicon |
| ART-21 | B1 | Articles with names & places | Geography: the (rivers/seas/oceans/ranges/deserts) vs Ø (single mountains/lakes) | the Thames; Ø Lake Geneva. | lexicon |
| ART-22 | B2 | Articles with names & places | Institutions/buildings/media | the BBC, the Times; Ø Time magazine; the Hilton; Ø Oxford University. | lexicon |
| ART-23 | B2 | Articles with names & places | Fixed/idiomatic contrasts | in Ø hospital / in the hospital; at Ø sea / at the seaside. | lexicon |
| ART-24 | B1 | Articles with names & places | most contrasts | most people / the most expensive / most of the people. | rule+LLM |

### DET — Determiners & quantifiers

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| DET-01 | A1 | Demonstratives & possessives | Demonstratives (proximal/distal) | this/that/these/those book(s). | rule |
| DET-02 | B1 | Demonstratives & possessives | Demonstratives (temporal/textual/emotive) | this week; that idea; this guy comes up to me. | LLM |
| DET-03 | A1 | Demonstratives & possessives | Possessive determiners | my, your, his, her, its, our, their. | rule |
| DET-04 | A2 | Demonstratives & possessives | its vs it's (orthographic) | The dog wagged its tail. | lexicon |
| DET-05 | A1 | some / any | some (assertive, indefinite quantity) | I bought some apples. | rule+LLM |
| DET-06 | A2 | some / any | some in offers/requests | Would you like some tea? | LLM |
| DET-07 | B1 | some / any | some = certain/particular | Some people just don't listen. | LLM |
| DET-08 | A1 | some / any | any (non-assertive: neg/question/conditional) | Have you any questions? not...any. | rule+LLM |
| DET-09 | B1 | some / any | any = no-matter-which / every | Take any seat. Any child can do it. | LLM |
| DET-10 | B2 | some / any | any + comparative | Is it any better? | rule+LLM |
| DET-11 | A1 | Amount quantifiers | much / many | much time; many people. | rule+LLM |
| DET-12 | A1 | Amount quantifiers | a lot of / lots of / plenty of | a lot of work; plenty of time. | rule+LLM |
| DET-13 | B2 | Amount quantifiers | a great deal of / a large number of / a good many | a great deal of effort. | lexicon |
| DET-14 | A2 | Amount quantifiers | (a) few / (a) little (count/uncount; +/− nuance) | a few friends; few options; little hope. | rule+LLM |
| DET-15 | B1 | Amount quantifiers | fewer / less / least / fewest (+ prescriptive fewer/less) | fewer cars; less traffic. | rule+LLM |
| DET-16 | A2 | Amount quantifiers | too much/many; so much/many; as much/many | too many emails. | rule |
| DET-17 | A2 | Universal & distributive | all / whole | all (of) the students; the whole class. | rule+LLM |
| DET-18 | A2 | Universal & distributive | both (of) | both options; both of them. | rule |
| DET-19 | A2 | Universal & distributive | each / every (+ singular agreement; each vs every) | every student; each child. | rule+LLM |
| DET-20 | B1 | Universal & distributive | either / neither (of) | either day; neither answer. | rule+LLM |
| DET-21 | A2 | Universal & distributive | half (of) | half (of) the cake. | rule |
| DET-22 | A2 | Negative & predeterminer & misc | no (+ sg/pl/uncount) | no money; no cars. | rule |
| DET-23 | B1 | Negative & predeterminer & misc | Predeterminers (all/both/half + det; such (a); what (a); rather/quite a; many a) | half the time; such a pity. | rule+LLM |
| DET-24 | B1 | Negative & predeterminer & misc | enough; several; various; certain; numerous | enough chairs; several issues. | rule+LLM |
| DET-25 | A2 | Negative & predeterminer & misc | Numbers, fractions, multipliers | three books; two-thirds of; twice the size. | rule+LLM |
| DET-26 | B1 | Negative & predeterminer & misc | Distributive per / apiece / each | €5 per person; two each. | lexicon |
| DET-27 | A2 | Negative & predeterminer & misc | another / other(s) / the other(s) | another cup; the other one; some...others. | rule+LLM |
| DET-28 | B1 | Negative & predeterminer & misc | Possessive + own (emphatic ownership) | my own room; her own car; a place of my own. | rule+LLM |

### PRO — Pronouns

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| PRO-01 | A1 |  | Personal subject/object | I/me, he/him, they/them. | rule |
| PRO-02 | B1 |  | Generic/impersonal you / they / one / we | You never know. They say it'll rain. | LLM |
| PRO-03 | A1 |  | Dummy it (weather/time/distance) | It's raining. It's five o'clock. It's far. | rule+LLM |
| PRO-04 | B1 |  | Anticipatory it (extraposition) | It's hard to say. It seems that... | rule+LLM |
| PRO-05 | A2 |  | Possessive pronouns | mine, yours, hers, ours, theirs. | rule |
| PRO-06 | A2 |  | Reflexive (true reflexive) | She blamed herself. Help yourself. | rule |
| PRO-07 | B1 |  | Emphatic reflexive / by oneself | I'll do it myself. He lives by himself. | rule+LLM |
| PRO-08 | B1 |  | Reciprocal | each other, one another. | rule |
| PRO-09 | A1 |  | Indefinite compounds (some-/any-/no-/every- + -one/-body/-thing/-where) | Nobody knew. Is anyone there? | rule+LLM |
| PRO-10 | B1 |  | Indefinite + adjective / to-inf | something new; nothing to do. | rule |
| PRO-11 | B2 |  | Singular they | Someone left their bag. | rule+LLM |
| PRO-12 | A2 |  | one / ones substitution | the red one; the ones on the left. | rule+LLM |
| PRO-13 | C1 |  | Generic/formal one | One should be careful. | LLM |
| PRO-14 | A1 |  | Demonstrative pronouns (standalone) | This is mine. Those are old. | rule |
| PRO-15 | A2 |  | Quantifying pronouns (of them) | all of them; none of us; some of these. | rule |

### ADJ — Adjectives

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| ADJ-01 | A1 | Position | Attributive position | a tall man. | rule |
| ADJ-02 | A1 | Position | Predicative position | The man is tall. | rule |
| ADJ-03 | B2 | Position | Attributive-only adjectives | the main reason; the former president; sheer luck. | lexicon |
| ADJ-04 | B1 | Position | Predicative-only (a-adjectives, health) | asleep, afraid, alive, alone, aware; She's well/ill. | lexicon |
| ADJ-05 | B2 | Position | Postpositive adjectives | something nice; the people present; attorney general. | lexicon |
| ADJ-06 | B1 | Order, gradability, form | Adjective order (opinion–size–age–shape–colour–origin–material–purpose) | a large round wooden table. | rule+LLM |
| ADJ-07 | B2 | Order, gradability, form | Coordinate vs cumulative adjectives (commas) | a long, hot summer. | rule+LLM |
| ADJ-08 | A1 | Order, gradability, form | Gradable adjectives + very/quite/rather | very tired. | lexicon+LLM |
| ADJ-09 | B2 | Order, gradability, form | Non-gradable/absolute + absolutely/completely | absolutely freezing; | lexicon+LLM |
| ADJ-10 | B1 | Order, gradability, form | Classifying adjectives (non-gradable) | nuclear power; a daily routine. | LLM |
| ADJ-11 | A2 | Order, gradability, form | -ing participial adjective (source) | an interesting book; the news is boring. | rule+LLM |
| ADJ-12 | A2 | Order, gradability, form | -ed participial adjective (experiencer) | I'm interested/bored. | rule+LLM |
| ADJ-13 | B1 | Order, gradability, form | Compound adjectives | well-known; a three-year-old child; world-famous; good-looking. | lexicon |
| ADJ-14 | B1 | Order, gradability, form | Adjectives as nouns | the poor; the impossible; the unknown. | rule+LLM |
| ADJ-15 | A2 | Comparison (see COM) | Comparative/superlative of adjectives |  | rule |
| ADJ-16 | A2 | Complementation | Adjective + to-infinitive | easy to please; ready to go; likely to win. | lexicon |
| ADJ-17 | B2 | Complementation | Tough-movement | The book is hard to read. | rule+LLM |
| ADJ-18 | B1 | Complementation | Adjective + that-clause | I'm glad (that) you came; sure that... | lexicon |
| ADJ-19 | A2 | Complementation | Adjective + preposition (+ -ing/NP) | good at; interested in; afraid of; keen on. | lexicon |
| ADJ-20 | B1 | Complementation | too/enough/so | too cold to swim; warm enough; so tired that... | rule |
| ADJ-21 | B1 | Complementation | Adjective + for + NP + to-infinitive (incl. extraposed) | It's important for us to leave; I'm eager for it to end. | rule |

### ADV — Adverbs & adverbials

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| ADV-01 | A1 | Formation | Regular adverb (adj + -ly) | quickly, carefully. | rule |
| ADV-02 | A2 | Formation | Irregular adverb | good→well; fast, hard, late, early, daily. | lexicon |
| ADV-03 | A2 | Formation | Same form as adjective | drive fast; work hard; arrive early/late. | lexicon |
| ADV-04 | B2 | Formation | Two adverb forms with meaning difference | hard/hardly; late/lately; near/nearly; high/highly; deep/deeply. | lexicon |
| ADV-05 | A2 | Types | Adverbs of manner | She spoke quietly. | rule |
| ADV-06 | A1 | Types | Adverbs of place/direction | here, there, outside, abroad, upstairs. | rule |
| ADV-07 | A1 | Types | Adverbs of definite time | yesterday, now, then, today, soon. | rule |
| ADV-08 | A1 | Types | Adverbs of indefinite frequency | always, usually, often, sometimes, rarely, never. | rule |
| ADV-09 | A2 | Types | Adverbs of definite frequency | daily, weekly, twice a week, every day. | lexicon |
| ADV-10 | A2 | Types | Degree adverbs | very, too, quite, rather, extremely, fairly, pretty. | rule+LLM |
| ADV-11 | A2 | Types | Focusing adverbs | only, even, just, also, too, as well, either. | rule+LLM |
| ADV-12 | C1 | Types | Viewpoint/domain adverbs | financially, technically, politically. | lexicon |
| ADV-13 | B1 | Types | Comment/stance adverbs (disjuncts) | frankly, obviously, fortunately, apparently. | lexicon |
| ADV-14 | B1 | Types | Linking adverbs (conjuncts) | however, therefore, moreover. | lexicon |
| ADV-15 | B1 | Position & scope | Front/initial position | Suddenly, the lights went out. | rule |
| ADV-16 | A2 | Position & scope | Mid-position (after operator / before main verb) | She has always known. I rarely eat out. | rule |
| ADV-17 | A1 | Position & scope | End position | She sang beautifully. | rule |
| ADV-18 | B1 | Position & scope | Order of manner–place–time | She worked quietly at home all day. | rule+LLM |
| ADV-19 | C1 | Position & scope | Scope of focusing adverbs (only/even) | Only I saw it / I only saw it. | LLM |
| ADV-20 | A2 | Position & scope | Frequency adverbs with imperative | Always check your mirrors. | rule |
| ADV-21 | A2 | Position & scope | ever in questions/negatives | Have you ever...? Nobody ever calls. | rule+LLM |
| ADV-22 | B1 | Degree details | Modifying comparatives (much/far/a lot/a bit/slightly/even/no/any + comp) | far better. | rule |
| ADV-23 | B1 | Degree details | Modifying verbs (completely/totally/strongly agree) | I totally agree. | lexicon |
| ADV-24 | A1 | Degree details | very/really/so + adjective/adverb | so quickly; really good. | rule |
| ADV-25 | B1 | Degree details | quite/rather/fairly/pretty (downtoners; quite ambiguity) | quite good (= fairly/completely). | LLM |
| ADV-26 | B1 | Degree details | too / enough (+ for + to-inf) | too hot (for me) to drink; fast enough to win. | rule |
| ADV-27 | B1 | Comparison of adverbs | Comparative/superlative adverbs | more quickly; (the) fastest; the most carefully. | rule |
| ADV-28 | B1 | Comparison of adverbs | as | as quickly as possible. | rule |
| ADV-29 | A2 | Comparison of adverbs | Phase adverbs — continuation / earlier-than-expected / expected-not-yet; cf. *no longer / not any more* (NEG-07). | He's still asleep; she's already left; have they arrived yet? | rule+LLM |

### COM — Comparison

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| COM-01 | A2 | Comparative form | Inflectional comparative (-er) | taller, bigger, happier. | rule |
| COM-02 | A2 | Comparative form | Periphrastic comparative (more) | more useful, more carefully. | rule |
| COM-03 | B2 | Comparative form | Two-syllable variation | cleverer / more clever; simpler / more simple. | rule+LLM |
| COM-04 | A2 | Comparative form | Irregular comparison | good/better; bad/worse; far/farther-further; little/less; much-many/more. | lexicon |
| COM-05 | A2 | Superlative form | Inflectional superlative (-est) | the tallest. | rule |
| COM-06 | A2 | Superlative form | Periphrastic superlative (most) | the most useful. | rule |
| COM-07 | B1 | Superlative form | Superlative + restriction (in/of/ever + perfect) | the best film I've ever seen; the tallest in the class. | rule+LLM |
| COM-08 | B1 | Standard & modification | Than-clause (case + ellipsis) | taller than me / than I am. | rule+LLM |
| COM-09 | B1 | Standard & modification | Equative as | as tall as his brother. | rule |
| COM-10 | B1 | Standard & modification | Negative equative | not as/so big as. | rule |
| COM-11 | B2 | Standard & modification | Multiples / scaled equative | twice as long; half as expensive; three times the size. | rule+LLM |
| COM-12 | B1 | Standard & modification | Modified comparative | much/far/a lot/a bit/slightly/even/no/any + comparative. | rule |
| COM-13 | B2 | Standard & modification | Proportional (the + comp | The more you practise, the better you get. | lexicon |
| COM-14 | B1 | Standard & modification | Incremental (comp and comp / more and more) | colder and colder; more and more difficult. | lexicon |
| COM-15 | B1 | Standard & modification | Decreasing comparison (less/least; fewer) | less important; the least likely. | rule |
| COM-16 | B1 | Similarity / difference | like / unlike / as | He runs like the wind. Do as I say. | rule+LLM |
| COM-17 | A2 | Similarity / difference | the same as / similar to / different from(-to/-than) | similar to yours. | lexicon |
| COM-18 | B1 | Similarity / difference | so/such | so cold that...; such a mess that... | rule |

### PREP — Prepositions

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| PREP-01 | A1 | Place | in / on / at (enclosure / surface / point) | in the box; on the wall; at the door. | lexicon |
| PREP-02 | A1 | Place | Spatial relations | above/over, below/under, between/among, in front of/behind, beside, opposite. | lexicon |
| PREP-03 | A2 | Place | inside/outside, against, around, along, beyond | around the corner. | lexicon |
| PREP-04 | A1 | Time | at / on / in (point / day / period) | at six; on Monday; in May. | lexicon |
| PREP-05 | A2 | Time | for / since (duration / starting point) | for two hours; since 2010. | rule+LLM |
| PREP-06 | B1 | Time | during / throughout / over / within | during the meeting. | lexicon |
| PREP-07 | B1 | Time | by / until(till) (deadline / up to) | by Friday; until noon. | rule+LLM |
| PREP-08 | A2 | Time | from | from nine to five; two years ago; in an hour. | lexicon |
| PREP-09 | A2 | Movement / direction | to / into / onto / out of / off / from | into the room; out of the car. | lexicon |
| PREP-10 | B1 | Movement / direction | towards / away from / up / down / across / through / along / past / over / under / via | across the bridge. | lexicon |
| PREP-11 | A2 | Other meanings | of (possession/partitive/material/about) | the leg of the table; made of wood. | rule+LLM |
| PREP-12 | A2 | Other meanings | with / without (accompaniment/instrument/manner) | with a knife; without help. | rule+LLM |
| PREP-13 | A2 | Other meanings | by (agent/means/measure) | by train; by 10%; written by her. | rule+LLM |
| PREP-14 | A1 | Other meanings | for (purpose/recipient/duration/exchange) | for you; for lunch; for €5. | rule+LLM |
| PREP-15 | A2 | Other meanings | about / on (topic) | a book about/on grammar. | rule+LLM |
| PREP-16 | B1 | Other meanings | as (role/function) | as a teacher; works as a guide. | rule+LLM |
| PREP-17 | B1 | Other meanings | Complex prepositions (concession/cause/exception) | despite, in spite of, because of, due to, instead of, apart from, except for, according to, thanks to, regarding. | lexicon |
| PREP-18 | A2 | Dependent prepositions | Verb + preposition | depend on, rely on, deal with, apologise for, accuse of, insist on, consist of, refer to, belong to, object to. | lexicon |
| PREP-19 | A2 | Dependent prepositions | Adjective + preposition | good at, interested in, afraid of, keen on, proud of, capable of, aware of, married to, different from, worried about. | lexicon |
| PREP-20 | B1 | Dependent prepositions | Noun + preposition | reason for, solution to, increase in, demand for, effect on, cause of, attitude to(wards), relationship with/between, interest in, need for. | lexicon |
| PREP-21 | B1 | Dependent prepositions | Preposition + -ing (incl | look forward to seeing; object to paying; with a view to. | lexicon |
| PREP-22 | B1 | Syntax of prepositions | Preposition stranding | the man I spoke to; What are you looking at? | rule |
| PREP-23 | C1 | Syntax of prepositions | Pied-piping (fronted preposition) | the man to whom I spoke; in which. | rule |
| PREP-24 | B1 | Syntax of prepositions | Preposition omission | (on) next week; discuss Ø it; enter Ø; reach Ø; this morning. | lexicon |
| PREP-25 | B1 | Syntax of prepositions | Prepositional phrase as postmodifier/complement/adjunct | the key to success; a man of his word. | rule |

## III. Clause & Sentence

### CLS — Basic clause patterns

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| CLS-01 | A1 | Clause types | SV (intransitive) | Birds fly. | rule |
| CLS-02 | A1 | Clause types | SVC (subject complement) | She is a doctor / happy. | rule |
| CLS-03 | A1 | Clause types | SVO (monotransitive) | He read the book. | rule |
| CLS-04 | A2 | Clause types | SVOO (ditransitive) | She gave him a gift. | rule |
| CLS-05 | B1 | Clause types | SVOC (object complement) | They named him captain. | rule |
| CLS-06 | A1 | Clause types | SVA / SVOA (obligatory adverbial) | He lives in Rome. She put it on the shelf. | rule |
| CLS-07 | A1 | Clause types | Canonical SVO word order (baseline) | The committee approved the plan. | rule |
| CLS-08 | A1 | Agreement | Basic subject–verb agreement (3sg -s) | She works. | rule |
| CLS-09 | A2 | Agreement | Coordinate-subject agreement (and = pl; or/nor = proximity) | Tom and Sue are; Either Tom or his friends are. | rule+LLM |
| CLS-10 | B2 | Agreement | Quantifier-headed agreement | Each is; A number of people are; The number of people is; None is/are. | LLM |
| CLS-11 | B2 | Agreement | Measure/amount as singular | Ten years is a long time; Five euros is enough. | LLM |
| CLS-12 | C1 | Agreement | Notional agreement (what-clauses, collectives) | What we need is/are...; The team are. | LLM |
| CLS-13 | A1 | Existential & dummy subjects | Existential there is/are | There's a problem; There are two options. | rule |
| CLS-14 | B1 | Existential & dummy subjects | Existential + modal/perfect/seem | There might be a delay; There seems to be a mistake. | rule |
| CLS-15 | A2 | Existential & dummy subjects | Copular verbs (change & perception) | become/get/turn/go/grow; seem/appear/look/sound/feel/taste/smell + adj. | lexicon |
| CLS-16 | B1 | Existential & dummy subjects | Copular + adj vs adverb confusion | It looks good (not well); She seems nice. | rule+LLM |
| CLS-17 | B1 | Existential & dummy subjects | seem/appear + to-infinitive | He seems to know. | rule |

### QUE — Questions

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| QUE-01 | A1 |  | Yes/no questions (operator: be/have/modal) | Are you ready? Has she left? | rule |
| QUE-02 | A1 |  | Yes/no with do-support | Do you agree? Did they come? | rule |
| QUE-03 | A1 |  | Short answers | Yes, I do. No, she hasn't. | rule |
| QUE-04 | A1 |  | Object wh-questions | What do you want? Who did you see? | rule |
| QUE-05 | A1 |  | Adjunct wh-questions | Where/When/Why/How did they go? | rule |
| QUE-06 | A1 |  | how + adj/adv/much/many/long/often/far | How old are you? How long does it take? | rule |
| QUE-07 | A2 |  | Subject wh-questions (no inversion) | Who called? What happened? | rule |
| QUE-08 | A2 |  | which/what/whose + noun | Which book do you mean? | rule |
| QUE-09 | B1 |  | Preposition placement in questions | Who are you waiting for? For whom...? | rule |
| QUE-10 | A2 |  | What | What's she like? How are you? | rule+LLM |
| QUE-11 | B1 |  | Question tags (reversed polarity) | You're coming, aren't you? | rule+LLM |
| QUE-12 | B2 |  | Same-polarity / special tags | So you're leaving, are you? aren't I; will you (imperative); shall we (let's). | rule+LLM |
| QUE-13 | C1 |  | Tags after negative/indefinite words | Nothing happened, did it? Everyone agrees, don't they? | rule+LLM |
| QUE-14 | B1 |  | Negative questions | Don't you agree? Why didn't you call? | rule+LLM |
| QUE-15 | B1 |  | Indirect/embedded questions (no inversion; if/whether) | Could you tell me where it is? I wonder if she knows. | rule |
| QUE-16 | A2 |  | how come / what if / what about / how about (+ -ing) | What about meeting later? | lexicon |
| QUE-17 | B2 |  | Alternative & echo & rhetorical questions | Tea or coffee? You did what? Who cares? | rule+LLM |

### NEG — Negation

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| NEG-01 | A1 |  | Verb negation (not/n't + operator) | I don't know; she isn't ready; they haven't left. | rule |
| NEG-02 | A1 |  | Contractions | isn't, don't, won't, can't, shan't, mustn't; aren't I. | rule |
| NEG-03 | A2 |  | Non-assertive items in negative scope | I don't have any; I never saw anyone; not...yet/either. | rule+LLM |
| NEG-04 | A2 |  | Negative determiners/pronouns | no money; nobody knew; nothing happened. | rule |
| NEG-05 | B1 |  | no vs not a/not any | I have no time / I don't have any time. | rule+LLM |
| NEG-06 | B1 |  | Semi-negative adverbs | seldom, rarely, hardly (ever), scarcely, barely, little. | lexicon |
| NEG-07 | B1 |  | no longer / not | I no longer work there. | lexicon |
| NEG-08 | A2 |  | Transferred/raised negation | I don't think it'll work | LLM |
| NEG-09 | B1 |  | Negative + either / neither / nor (echoing) | I don't either; Neither do I. | rule |
| NEG-10 | B2 |  | Constituent (local) negation | not a sound; not surprisingly; a not unreasonable request. | rule+LLM |
| NEG-11 | C1 |  | Scope ambiguity (partial vs total) | All that glitters is not gold; Everyone isn't here. | LLM |
| NEG-12 | B2 |  | Litotes / understatement | not bad; not unlike. | LLM |
| NEG-13 | C1 |  | Negative fronting → inversion | Never have I...; No sooner had we... | rule |
| NEG-14 | C1 |  | Independent double negation | You can't not go; it's not that I don't care. | rule |

### IMP — Imperatives & directives

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| IMP-01 | A1 |  | Base-form imperative | Sit down. Turn left. | rule |
| IMP-02 | A1 |  | Negative imperative (don't / do not) | Don't move. | rule |
| IMP-03 | B2 |  | Emphatic / persuasive do | Do come in. Do be careful. | rule |
| IMP-04 | B2 |  | Subject pronoun for emphasis/contrast | You sit here; Somebody call a doctor. | rule+LLM |
| IMP-05 | A2 |  | always / never + imperative | Always read the label. | rule |
| IMP-06 | A1 |  | let's (suggestion) + negatives | Let's go; Let's not argue. | rule |
| IMP-07 | A2 |  | let me / let + object (3rd-person directive) | Let me help; Let them wait. | rule+LLM |
| IMP-08 | B2 |  | Imperative + and/or (conditional) | Hurry, or we'll be late; Move and I'll shout. | rule+LLM |
| IMP-09 | A1 |  | Softened imperative | Please sit down; Just wait here. | rule |

## IV. Connecting Clauses

### COORD — Coordination

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| COORD-01 | A1 |  | and (addition / sequence / result / condition) | I came and saw; Touch it and you'll regret it. | rule+LLM |
| COORD-02 | A1 |  | but / yet (contrast) | small but strong; cheap yet reliable. | rule |
| COORD-03 | A1 |  | or (alternative/exclusion) / nor | tea or coffee; neither rain nor snow. | rule |
| COORD-04 | A2 |  | so (result) / for (reason, formal) | It was late, so we left; He stopped, for he was tired. | rule+LLM |
| COORD-05 | B1 |  | both | both fast and reliable. | rule |
| COORD-06 | B1 |  | either | either Monday or Tuesday. | rule+LLM |
| COORD-07 | B2 |  | not only | not only late but rude; Not only did he..., but... | lexicon |
| COORD-08 | B2 |  | whether | whether you like it or not. | lexicon |
| COORD-09 | C1 |  | Ellipsis/gapping in coordination | I had tea and she Ø coffee. | rule+LLM |

### ASC — Adverbial subordinate clauses

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| ASC-01 | A2 | Time | when / whenever | Call me when you arrive. | lexicon |
| ASC-02 | B1 | Time | while / as (simultaneous) | While I cooked, she read. | lexicon |
| ASC-03 | A2 | Time | before / after / until(till) | Wait until I come back. | lexicon |
| ASC-04 | B1 | Time | since (time starting point) | I've known her since we were kids. | lexicon |
| ASC-05 | B1 | Time | as soon as / once / the moment / by the time / every time | As soon as it rings, answer. | lexicon |
| ASC-06 | B1 | Time | Present tense for future in time clauses (no will) | When he comes, we'll eat. | rule+LLM |
| ASC-07 | C1 | Time | no sooner | No sooner had we left than it rained. | lexicon |
| ASC-08 | A1 | Reason / purpose / result | Reason: because / since / as | We left because it was late. | lexicon |
| ASC-09 | B2 | Reason / purpose / result | Reason (formal): for / seeing (that) / now (that) / in that / given that | Now that you're here, let's start. | lexicon |
| ASC-10 | B1 | Reason / purpose / result | Purpose (finite): so (that) / in order that + modal | I wrote it down so that I'd remember. | lexicon |
| ASC-11 | A2 | Reason / purpose / result | Purpose (non-finite): to / in order to / so as to / for + -ing | I came to help. | rule+LLM |
| ASC-12 | B2 | Reason / purpose / result | Negative purpose: so as not to / in order not to / for fear (that) / lest | He whispered so as not to wake her. | lexicon |
| ASC-13 | B1 | Reason / purpose / result | Result: so | It was so hot that we stayed in. | lexicon |
| ASC-14 | B2 | Reason / purpose / result | Result: so that (no modal) / with the result that | He failed, with the result that... | lexicon |
| ASC-15 | B1 | Concession / contrast | although / though / even though | Although it rained, we went. | lexicon |
| ASC-16 | B2 | Concession / contrast | while / whereas / whilst (contrast) | She likes tea, whereas he prefers coffee. | lexicon |
| ASC-17 | B1 | Concession / contrast | despite / in spite of (+ NP/-ing/the fact that) | Despite the rain, we went. | lexicon |
| ASC-18 | B2 | Concession / contrast | however / whatever / whoever / wherever / no matter how/what | However hard I try... | lexicon |
| ASC-19 | C2 | Concession / contrast | Concessive inversion (Adj/N + as/though + S + V) | Tired as I was, I kept going. | rule+LLM |
| ASC-20 | B1 | Concession / contrast | even if | Even if it rains, we'll go. | lexicon |
| ASC-21 | B2 | Manner / place / proportion | Manner: as / (just) as / the way | Do it as I showed you. | rule+LLM |
| ASC-22 | B2 | Manner / place / proportion | Unreal manner: as if / as though | She acts as if she owned the place. | rule+LLM |
| ASC-23 | B1 | Manner / place / proportion | Place: where / wherever / everywhere | Sit wherever you like. | lexicon |
| ASC-24 | B2 | Manner / place / proportion | Proportion: the + comp | The harder you work, the more you earn. | lexicon |
| ASC-25 | B2 | Reduced adverbial clauses | -ing participle clause | While walking home, I met her. | rule+LLM |
| ASC-26 | C1 | Reduced adverbial clauses | -ed participle clause | If asked, say nothing. | rule+LLM |
| ASC-27 | C1 | Reduced adverbial clauses | Perfect participle clause | Having finished, she left. | rule |
| ASC-28 | C1 | Reduced adverbial clauses | with + NP + complement (absolute) | With the door open, it was cold. | rule+LLM |
| ASC-29 | C1 | Reduced adverbial clauses | Verbless clauses | When in doubt, ask; If necessary, call. | rule+LLM |

### CON — Conditionals

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| CON-01 | A2 |  | Zero conditional | If you heat ice, it melts. | rule+LLM |
| CON-02 | A2 |  | First conditional | If it rains, we'll cancel. | rule+LLM |
| CON-03 | B1 |  | Second conditional (+ If I were you) | If I had time, I would help. | rule+LLM |
| CON-04 | B2 |  | Third conditional | If you had asked, I would have helped. | rule+LLM |
| CON-05 | C1 |  | Mixed (past→present) | If I had saved, I'd be rich now. | rule+LLM |
| CON-06 | C1 |  | Mixed (present→past) | If I were more careful, I wouldn't have lost it. | rule+LLM |
| CON-07 | B1 |  | unless (= if not) | I won't go unless you come. | rule+LLM |
| CON-08 | B2 |  | as/so long as / provided (that) / on condition that | You can stay as long as you're quiet. | lexicon |
| CON-09 | B2 |  | even if / only if / whether | Even if you apologise, I won't forget. | lexicon |
| CON-10 | B1 |  | in case (precaution) | Take an umbrella in case it rains. | rule+LLM |
| CON-11 | B2 |  | suppose / supposing / imagine / what if | Suppose he says no? | lexicon |
| CON-12 | C1 |  | but for / if it weren't/hadn't been for / otherwise | But for your help, I'd have failed. | lexicon |
| CON-13 | C1 |  | Inversion (no if) | Had I known...; Were she to ask...; Should you need anything... | rule |
| CON-14 | B2 |  | Implied/incomplete conditionals | If so; if not; in that case; then. | rule+LLM |
| CON-15 | B2 |  | Imperative/coordination as condition | Do that again and you're out. | rule+LLM |
| CON-16 | C1 |  | if + would/could (politeness/insistence) | If you would wait here, please. | rule+LLM |
| CON-17 | C1 |  | Tentative conditionals (were to / should / happen to) | If you should see her... | rule+LLM |

### REL — Relative clauses

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| REL-01 | A2 | Defining | Defining: who (people) | the man who called. | rule |
| REL-02 | A2 | Defining | Defining: which (things) | the book which won. | rule |
| REL-03 | A2 | Defining | Defining: that (people/things) | the film that I saw. | rule |
| REL-04 | B1 | Defining | Zero relative (object) | the film (Ø) I saw. | rule |
| REL-05 | B1 | Defining | whose (possessive relative) | the author whose book won. | rule |
| REL-06 | B2 | Defining | whom (object, formal) | the colleague whom I trust. | rule |
| REL-07 | B2 | Defining | that-preference contexts (superlative/all/only/indefinite) | the only thing that matters; everything that... | rule+LLM |
| REL-08 | B1 | Non-defining | Non-defining (people/things, commas, no that) | My brother, who lives in Rome, is a chef. | rule+LLM |
| REL-09 | B2 | Non-defining | Sentential relative (which = whole clause) | She passed, which surprised everyone. | rule+LLM |
| REL-10 | C1 | Non-defining | Quantifier + of + which/whom | the students, many of whom were absent. | rule+LLM |
| REL-11 | B2 | Prepositions & adverbs | Stranded preposition in relative | the house (that) she lives in. | rule |
| REL-12 | B2 | Prepositions & adverbs | Pied-piped preposition | the house in which she lives. | rule |
| REL-13 | B1 | Prepositions & adverbs | Relative adverbs (where/when/why) | the place where; the day when; the reason why. | rule |
| REL-14 | C1 | Prepositions & adverbs | whereby (formal) | a system whereby costs are shared. | lexicon |
| REL-15 | B2 | Reduced relatives | -ing participle (active) | the people waiting outside. | rule+LLM |
| REL-16 | B2 | Reduced relatives | -ed participle (passive) | the report written last year. | rule+LLM |
| REL-17 | B2 | Reduced relatives | to-infinitive relative | the next train to arrive; the best way to do it; the first to know. | rule+LLM |
| REL-18 | B1 | Reduced relatives | Adjective/prepositional reduced relative | the people responsible; the woman in the corner. | rule+LLM |

### NCL — Noun / complement clauses

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| NCL-01 | A2 |  | That-clause as object | I think (that) it works. | lexicon |
| NCL-02 | B2 |  | That-clause as subject / extraposed | That he lied is clear; It is clear that he lied. | rule+LLM |
| NCL-03 | B2 |  | Appositive that-clause (after noun) | the fact that he resigned; the idea that... | lexicon |
| NCL-04 | B1 |  | That-clause after adjective | I'm sure that...; glad/aware/afraid that... | lexicon |
| NCL-05 | B1 |  | that-omission rules | I know Ø you're busy | rule+LLM |
| NCL-06 | C1 |  | Subjunctive that-clause | It's essential that he be informed. | lexicon |
| NCL-07 | B1 |  | Wh-nominal clause (subject/object/complement) | What you said matters; I know what he wants. | rule |
| NCL-08 | B1 |  | Wh- + to-infinitive | I don't know what to do / where to go / whether to stay. | rule |
| NCL-09 | B2 |  | Extraposition / anticipatory it (subject & object) | It's no use complaining; I find it odd that... | rule+LLM |
| NCL-10 | B1 |  | if/whether clause (embedded polar) | I wonder whether it's true; Ask if they're open. | lexicon |
| NCL-11 | B2 |  | whether | whether or not to go; the question of whether. | lexicon |
| NCL-12 | C1 |  | Nominal -ing clause | His leaving early surprised us; I appreciate your helping. | rule+LLM |
| NCL-13 | B2 |  | Noun + wh-complement clause | the question (of) whether to stay; no idea how it works | rule |
| NCL-14 | B1 |  | Noun + of + -ing complement clause | the possibility of going; the risk of losing everything | lexicon |

### REP — Reported speech

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| REP-01 | B1 |  | Reported statements + backshift | She said (that) she was tired. | rule+LLM |
| REP-02 | B2 |  | No backshift (still-true / just-said / general truth) | He said the Earth is round. | LLM |
| REP-03 | B1 |  | Pronoun & possessive shift | "I lost my keys" → He said he'd lost his keys. | rule+LLM |
| REP-04 | B2 |  | Deixis shift (this→that, here→there, now→then, today→that day, tomorrow→the next day, ago→before) | this→that, here→there, now→then, today→that day, tomorrow→the next day, ago→before | LLM |
| REP-05 | B1 |  | Reported questions (yes/no → if/whether) | He asked whether I agreed. | rule+LLM |
| REP-06 | B1 |  | Reported wh-questions (statement order) | She asked where I lived. | rule+LLM |
| REP-07 | B1 |  | Reported commands/requests (+ O + (not) to-inf) | She told me to wait; He asked me not to go. | lexicon |
| REP-08 | A2 |  | say vs tell (+ patterns) | say (to sb) that; tell sb (that). | lexicon |
| REP-09 | B2 |  | Illocutionary reporting verbs + patterns | offer/promise/threaten + to; suggest + -ing/that; admit/deny + -ing/that; accuse of; congratulate on; warn (not) to; insist on. | lexicon |
| REP-10 | B1 |  | Modals in reported speech (will→would, can→could, must→had to, may→might) | He said he would come. | rule+LLM |
| REP-11 | B1 |  | Reporting with present tense (news/summaries) | The report says that... | LLM |

## V. Discourse & Information Structure

### FOC — Focus, fronting, inversion, emphasis

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| FOC-01 | B2 |  | It-cleft (subject focus) | It was Maria who solved it. | rule+LLM |
| FOC-02 | B2 |  | It-cleft (object/adjunct focus) | It was the report that I read; It was here that we met. | rule+LLM |
| FOC-03 | C1 |  | It wasn't until | It wasn't until noon that he arrived. | rule+LLM |
| FOC-04 | B2 |  | Wh-cleft / pseudo-cleft | What I need is a break. | rule+LLM |
| FOC-05 | B2 |  | Reversed pseudo-cleft | A break is what I need; That's what I meant. | rule+LLM |
| FOC-06 | B2 |  | All (that) | All I want is the truth; The reason I called is... | rule+LLM |
| FOC-07 | C1 |  | Object/complement fronting | That I can't accept; Brilliant it was not. | rule+LLM |
| FOC-08 | C1 |  | Adverbial fronting (marked theme) | In the corner stood a lamp. | rule+LLM |
| FOC-09 | B2 |  | Directional fronting + subject-verb inversion | Down came the rain; Here comes the bus; There goes the train. | rule |
| FOC-10 | C1 |  | Subject-verb inversion after place/as/than (formal) | On the hill stood a castle; so did her sister. | rule |
| FOC-11 | C1 |  | Negative/restrictive fronting + subject-operator inversion | Never have I...; Not only did he...; Hardly had we...; Only then did I realise. | rule |
| FOC-12 | B1 |  | Emphatic do/did | I do appreciate it; She did warn you. | rule |
| FOC-13 | B1 |  | Additive inversion (so/neither/nor) | So do I; Neither did she. | rule |
| FOC-14 | B1 |  | so/such for emphasis | It was so beautiful; such a mess! | rule |
| FOC-15 | B2 |  | Wh-intensifiers (on earth / ever / the hell) | What on earth is this? | lexicon |
| FOC-16 | C1 |  | End-weight / end-focus (information packaging) | passive/extraposition/existential for end-focus. | LLM |
| FOC-17 | B2 |  | Right/left dislocation (informal) | My brother, he's a doctor; She's clever, your daughter. | rule+LLM |
| FOC-18 | B2 |  | Thing-is focus formula | The thing is(, is) that we're broke; the problem is he never listens. | lexicon |

### ELS — Ellipsis & substitution

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| ELS-01 | B1 |  | VP-ellipsis (post-operator) | I haven't finished but she has Ø; I will if you will. | rule+LLM |
| ELS-02 | B1 |  | Infinitive ellipsis (to) | I'd love to Ø; You can go if you want to Ø. | rule+LLM |
| ELS-03 | C1 |  | Gapping / stripping | I ordered tea and she Ø coffee; She likes jazz, and Bob too. | rule+LLM |
| ELS-04 | B1 |  | Nominal ellipsis | I'll take two Ø; the rich Ø; the first Ø to arrive. | rule+LLM |
| ELS-05 | B2 |  | do / do so / do it / do that substitution | He promised to call and he did (so). | rule+LLM |
| ELS-06 | A2 |  | Clausal so / not | I think so; I hope not; I'm afraid so. | lexicon |
| ELS-07 | B2 |  | the same / such substitution | I'll have the same; such was his anger. | rule+LLM |
| ELS-08 | A1 |  | Response/answer ellipsis | Yes, I do; Me too; Not me; So do I. | rule+LLM |
| ELS-09 | A2 |  | one/ones (nominal substitution) | the blue one; the ones I bought. | rule+LLM |
| ELS-10 | A2 |  | Comparative-clause ellipsis | She's taller than me Ø; faster than expected Ø. | rule+LLM |
| ELS-11 | B1 |  | Situational ellipsis (informal) | Want a coffee? Seen John? | LLM |

### COH — Cohesion & linking adverbials

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| COH-01 | B2 |  | Additive linkers | moreover, in addition, furthermore, besides, what's more, likewise. | lexicon |
| COH-02 | B1 |  | Adversative/contrastive linkers | however, nevertheless, nonetheless, on the other hand, in contrast, conversely, even so, instead. | lexicon |
| COH-03 | B1 |  | Causal linkers | therefore, thus, hence, consequently, as a result, accordingly, for this reason. | lexicon |
| COH-04 | A2 |  | Temporal/sequencing linkers | first(ly), then, next, afterwards, subsequently, finally, meanwhile, eventually, to begin with, in the end. | lexicon |
| COH-05 | A2 |  | Exemplifying linkers | for example, for instance, such as, namely, in particular, to illustrate. | lexicon |
| COH-06 | B2 |  | Reformulating/clarifying linkers | in other words, that is (to say), i.e., or rather, to put it another way. | lexicon |
| COH-07 | B1 |  | Summarising/concluding linkers | in conclusion, to sum up, in short, overall, on the whole, all in all. | lexicon |
| COH-08 | B2 |  | Emphasising linkers | indeed, in fact, as a matter of fact, above all, importantly, clearly. | lexicon |
| COH-09 | B2 |  | Conceding linkers | admittedly, of course, naturally, granted, it is true that... | lexicon |
| COH-10 | B1 |  | Conditional/consequence linkers | otherwise, in that case, if so, if not, under the circumstances. | lexicon |
| COH-11 | B1 |  | Discourse markers (spoken/informal, register-flagged) | well, anyway, by the way, you know, I mean, right, actually. | lexicon |
| COH-12 | B2 |  | Anaphoric reference (pronominal/demonstrative/the-) | ...this approach...; ...such cases...; ...the latter... | LLM |
| COH-13 | C1 |  | the former / the latter / the above / the following / aforementioned | The latter is cheaper. | lexicon |
| COH-14 | B2 |  | Punctuation of connectors (semicolon/comma conventions) | ...; however, ... | rule |

## VI. Spoken & Interactional Grammar

### INS — Inserts (freestanding interactional forms)

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| INS-01 | A1 | Inserts | Interjections | Oh! Wow! Ouch! Ugh! Oops! | lexicon |
| INS-02 | A1 | Inserts | Greetings & farewells | Hi. Good morning. Bye. See you later. Take care. | lexicon |
| INS-03 | A2 | Inserts | Attention signals | Hey! Oi! Look. Listen. Excuse me! | lexicon |
| INS-04 | B1 | Inserts | Response elicitors (freestanding) — Freestanding appeal for confirmation; clause-final invariant tags → QIN-06 | Eh? Right? Okay? You know? | lexicon |
| INS-05 | A2 | Inserts | Response forms / minimal responses & backchannels — Stand-alone C-unit; distinct from elliptical short answers (QUE-03/ELS-08) | Yeah. Mm. Uh huh. Right. Sure. Absolutely. Exactly. | lexicon |
| INS-06 | A2 | Inserts | Hesitators / filled pauses — Spoken transcripts only | er, erm, uh, um | lexicon |
| INS-07 | A1 | Inserts | Polite speech-act formulae | Please. Thanks / thank you. Sorry. Pardon? No problem. You're welcome. | lexicon |
| INS-08 | B2 | Inserts | Expletives & taboo interjections — Sensitive-content flag; moderated + non-taboo (gosh, blimey) variants | Damn! Bloody hell! God! Geez! | lexicon |
| INS-09 | C1 | Inserts | Taboo/expletive intensifiers (pre-adjective/NP) — FOC-15 covers only wh-intensifiers (on earth / the hell) | It's bloody freezing; a damn nuisance. | lexicon |

### VAG — Vague language & approximation

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| VAG-01 | B1 | Vague language | General extenders / coordination tags | and stuff (like that); and things; and so on; and everything; or something; or whatever | lexicon |
| VAG-02 | A2 | Vague language | Vague category nouns & placeholders | thing(s), stuff, thingy, whatsit, what-d'you-call-it | lexicon |
| VAG-03 | B1 | Vague language | Hedging sort of / kind of (+ VP/AdjP/NP) — ART-19 covers only classifying kind/sort/type of + noun | It sort of collapsed; kind of weird. | lexicon |
| VAG-04 | A2 | Vague language | Numeral approximators | about fifty; twenty or so; fiftyish; five or six people; twenty-odd | rule |

### CMT — Comment clauses & parentheticals

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| CMT-01 | B1 | Parentheticals | First-person epistemic parentheticals | It's fine, I think; He's, I guess, about forty; I reckon it'll rain. | rule+LLM |
| CMT-02 | B2 | Parentheticals | Interactive parentheticals (medial/final) — Medial/final position; utterance-initial use → DMG | He was, you know, quite upset; It's over there, you see; It's cold, mind you. | lexicon |
| CMT-03 | B1 | Parentheticals | As-comment clauses | as you know; as I said; as it happens | rule |
| CMT-04 | B2 | Parentheticals | Medial/final reporting clause (incl. inversion) | 'Fine,' he said. 'Come in,' said the old man, 'and sit down.' | rule |

### VOC — Vocatives

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| VOC-01 | A1 | Vocatives | Vocative: name / title | John, are you coming? Thanks, Dr Smith. | rule |
| VOC-02 | A2 | Vocatives | Vocative: kinship & endearment / familiarizers | Mum, where's my bag? Cheers, mate. Come on, love. | lexicon |
| VOC-03 | B1 | Vocatives | Vocative: impersonal, plural & honorifics | you guys; everyone; folks; ladies and gentlemen; sir; madam | lexicon |

### DMG — Discourse markers (item grain)

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| DMG-01 | B1 | Discourse markers | well (utterance launcher) | Well, let's see. Well, I'm not sure. | lexicon |
| DMG-02 | A2 | Discourse markers | oh (change-of-state / receipt) | Oh, I see. Oh right, okay. | lexicon |
| DMG-03 | B1 | Discourse markers | you know (shared-knowledge monitor, initial) | You know, it wasn't that bad. | lexicon |
| DMG-04 | B1 | Discourse markers | I mean (reformulation marker) | I mean, it's not exactly cheap, is it? | lexicon |
| DMG-05 | B2 | Discourse markers | Discourse like (filler / focus / approximator) — Disambiguate from preposition/comparative like (COM-16) and quotative be like (QUO-02) | It was like really strange; there were like fifty people. | LLM |
| DMG-06 | B1 | Discourse markers | so (utterance launcher / topic opener) | So, what happened next? So anyway, we left. | lexicon |
| DMG-07 | B1 | Discourse markers | anyway / anyhow (resumptive / closing) | Anyway, where were we? Anyhow, it worked out. | lexicon |
| DMG-08 | B1 | Discourse markers | right / okay (transition, acceptance, closing) | Right, let's start. Okay, that's settled then. | lexicon |
| DMG-09 | B1 | Discourse markers | Turn-initial and / but | And did you go? But that's not the point. | rule |

### QIN — Interactional questions & tags

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| QIN-01 | B1 | Interactional questions | Declarative question — Statement form + interrogative function; punctuation/prosody proxy | You're leaving? He actually said that? | LLM |
| QIN-02 | C1 | Interactional questions | Statement (copy) tags — Reinforcing same-polarity subject+operator copy; NP tails → FOC-17 | He's mad, he is. I'm not stupid, me. | rule |
| QIN-03 | B2 | Interactional questions | Exclamation tags | Wasn't it brilliant! Isn't she lovely! | rule |
| QIN-04 | A2 | Interactional questions | Follow-up / two-step questions — Discourse-level; needs preceding turn | A: In the drawer. B: Which drawer? | LLM |
| QIN-05 | B1 | Interactional questions | Preface questions | (You) know what? Guess what? You know something? | lexicon |
| QIN-06 | B2 | Interactional questions | Invariant tags — innit flagged vernacular | It's cold, innit? You're coming, yeah? Nice, eh? | lexicon |

### EXC — Exclamatives

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| EXC-01 | A2 | Exclamatives | What-exclamative clause | What a day (it was)! What nonsense! | rule |
| EXC-02 | B1 | Exclamatives | How-exclamative clause | How lovely! How quickly time passes! | rule |
| EXC-03 | A2 | Exclamatives | Verbless / phrasal exclamatives | Nice one! Brilliant! The cheek of it! Some help you were! | rule+LLM |

### QUO — Conversational reporting & quotatives

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| QUO-01 | B2 | Quotatives | Quotative go | And she goes, 'No way.' | lexicon |
| QUO-02 | B2 | Quotatives | Quotative be like / be all | I'm like, 'What?' And he's all, 'Calm down.' | lexicon |
| QUO-03 | B2 | Quotatives | Conversational historic-present reporting — Non-standard I says flagged vernacular | So he says, 'Come back Monday.' (narrative) | rule+LLM |
| QUO-04 | B1 | Quotatives | Past progressive reporting frame | She was saying (that) they might move. | rule |
| QUO-05 | C2 | Quotatives | Free direct & free indirect speech/thought — Written-narrative flavored; hard tier | He'd be back tomorrow, he was sure. Would she come? (narrative) | LLM |

### SIT — Situational ellipsis (subtypes)

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| SIT-01 | B1 | Situational ellipsis | Initial subject ellipsis | Doesn't matter. Serves you right. Told you. | rule |
| SIT-02 | B1 | Situational ellipsis | Subject + operator ellipsis | Want a coffee? Seen John? Been waiting long? | rule |
| SIT-03 | B2 | Situational ellipsis | Operator-only (medial) ellipsis — Shades into vernacular aux-drop | You seen John? She gone already? | rule |
| SIT-04 | B2 | Situational ellipsis | Other situational fragments (copula/determiner/preposition drop) | (Are you) ready? Good film, that. (It's a) pity. (At) about six? | rule+LLM |

### ISB — Insubordination (freestanding subordinate clauses)

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| ISB-01 | B2 | Insubordination | Stand-alone if-clause (polite directive/offer) | If you could just sign here. If you'd like to follow me. | rule |
| ISB-02 | A2 | Insubordination | Stand-alone because/cos-clause | Why not? — Because it's late. Cos I said so. | rule |
| ISB-03 | C1 | Insubordination | Stand-alone which-comment clause | They paid for everything. Which was nice. | rule |

### PSC — Pseudo-coordination & phrasal intensification

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| PSC-01 | B1 | Pseudo-coordination | try and + verb | I'll try and come early. | rule |
| PSC-02 | A2 | Pseudo-coordination | go/come and + verb | Go and get it. Come and see us. | rule |
| PSC-03 | B2 | Pseudo-coordination | nice and / good and + adjective | It's nice and warm in here; when you're good and ready. | lexicon |

### VER — Vernacular grammar (flagged stratum)

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| VER-01 | — | Vernacular | Negative concord (multiple negation) | I didn't do nothing. He never said nothing to nobody. | rule |
| VER-02 | — | Vernacular | ain't (= be/have negative) | He ain't here. I ain't got one. | lexicon |
| VER-03 | — | Vernacular | Non-standard concord | We was there. He don't care. Them things is heavy. | rule |
| VER-04 | — | Vernacular | Demonstrative them | them books; one of them things | rule |
| VER-05 | — | Vernacular | Leveled/regularized verb forms | He seen it. She done it. I knowed him. | lexicon |
| VER-06 | — | Vernacular | Reduced semi-modals gonna / wanna / gotta (transcribed) | I'm gonna leave. You gotta see this. | lexicon |
| VER-07 | — | Vernacular | Double comparative/superlative | more better; the most brightest | rule |

### DYS — Performance phenomena (optional annotation layer)

| ID | CEFR | Family | Construction | Example | Detector |
|---|---|---|---|---|---|
| DYS-01 | — | Dysfluency | Repeats — Performance layer, not competence construct; spoken transcripts only | I I I don't know. | rule |
| DYS-02 | — | Dysfluency | Retrace-and-repair / reformulation — Performance layer | She was — he was already there. | LLM |
| DYS-03 | — | Dysfluency | Incomplete / abandoned utterance — Performance layer | I just thought maybe — | LLM |
| DYS-04 | — | Dysfluency | Syntactic blend (anacoluthon) — Performance layer | That's the one thing is important. | LLM |

