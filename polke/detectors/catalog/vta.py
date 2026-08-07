"""Category VTA - tense & aspect (VTA-01..49).

The FORM of a tense is a deterministic aux-chain router (present simple, present
continuous, past simple, past/present perfect, ...). Where siblings share ONE
form and differ only by MEANING (present-continuous 'now' vs 'arrangement' vs
'annoyance'), a rule localises the form and an LLMReadingDetector picks the
reading.

Rule/lexicon tier (tested offline):
  VTA-07  performative present simple (lexicon-gated: 1st-person + speech-act verb)
  VTA-11  present continuous 'action in progress now' (see flag below)
  VTA-17/18/20  past-simple readings decidable by form (single event / sequence /
                past state), routed together; habit/remote/hypothetical -> None
  VTA-23/26  past-continuous 'in progress at a past point' / 'parallel actions'
  VTA-39/40/41/42  past-perfect readings decidable by form (anterior / by-the-time
                   / reported backshift / conditional)
  VTA-44  past perfect continuous form
LLM tier (registered, skipped offline): the remaining readings.

VTA-29..34 (present-perfect-simple readings) are handled by reference.py and are
deliberately NOT registered here.

FLAG: VTA-11 is typed `rule` in the contract but its negatives are all OTHER
present-continuous readings (temporary / changing / annoyance / arrangement), so
it cannot be separated from its siblings by form alone. It is implemented as the
DEFAULT present-continuous reading via heuristic exclusion of the sibling
signals (always/forever -> annoyance; get/become+comparative -> changing;
this week / at six / tomorrow -> temporary/arrangement). This passes the contract
but is really a reading; the honest home is the present-continuous LLM group.
"""
from __future__ import annotations
from ..routing import RuleRoutingDetector, LexiconRuleDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector

# --------------------------------------------------------------------------- #
# Word lists (kept local per the brief; candidates for lexicons.json in notes).
# --------------------------------------------------------------------------- #
PERFORMATIVE = {
    "promise", "apologise", "apologize", "admit", "accept", "advise", "agree",
    "assure", "beg", "confess", "congratulate", "declare", "deny", "forbid",
    "guarantee", "insist", "invite", "name", "order", "permit", "pronounce",
    "propose", "refuse", "request", "resign", "suggest", "swear", "thank",
    "warn", "acknowledge", "predict", "bet", "vow", "nominate", "object",
    "protest", "recommend", "remind", "second", "sentence", "authorise",
    "authorize", "grant", "welcome", "forgive", "concede", "advise",
}

STATIVE = {
    "be", "have", "know", "believe", "like", "love", "hate", "want", "need",
    "prefer", "own", "belong", "seem", "appear", "understand", "remember",
    "recognize", "recognise", "mean", "matter", "cost", "weigh", "contain",
    "consist", "resemble", "deserve", "depend", "doubt", "envy", "fear",
    "possess", "owe", "lack", "suit", "concern", "involve",
}

# Frequency / habit adverbials -> present or past HABIT reading (not single event).
FREQ = {
    "every", "each", "always", "usually", "often", "sometimes", "frequently",
    "occasionally", "daily", "weekly", "monthly", "yearly", "nightly",
    "regularly", "normally", "generally", "rarely", "seldom", "repeatedly",
    "whenever", "constantly", "habitually", "routinely",
}

ANNOY = {"always", "forever", "constantly", "continually", "perpetually",
         "continuously", "everlastingly"}

# Verbs of change/development -> 'changing situation' reading.
CHANGE = {
    "get", "become", "grow", "turn", "rise", "fall", "increase", "decrease",
    "improve", "worsen", "widen", "narrow", "deepen", "warm", "cool", "change",
    "develop", "expand", "evolve", "progress", "shrink", "spread", "climb",
    "drop", "gain", "lose", "strengthen", "weaken",
}

# Verbs of reporting/communication -> backshift in reported speech.
REPORTING = {
    "say", "tell", "claim", "admit", "explain", "state", "report", "announce",
    "mention", "reply", "note", "insist", "remark", "add", "think", "believe",
    "know", "feel", "decide", "realize", "realise", "promise", "confess",
    "declare", "reveal", "argue", "observe", "complain", "suggest", "whisper",
    "respond", "answer", "assume", "conclude", "discover", "guess", "recall",
    "acknowledge", "confirm", "estimate", "reckon",
}

TIME_WORDS = {"tomorrow", "tonight", "soon", "later", "shortly", "presently"}
NEAR_DET = {"this", "next", "coming", "following"}


# --------------------------------------------------------------------------- #
# Shared FORM patterns.
# --------------------------------------------------------------------------- #
_PRES_SIMPLE = [{"RIGHT_ID": "v",
                 "RIGHT_ATTRS": {"TAG": {"IN": ["VBZ", "VBP"]},
                                 "DEP": {"IN": ["ROOT", "conj"]}}}]

_PRES_CONT = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBG"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "be",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "be", "TAG": {"IN": ["VBZ", "VBP"]}}},
]

_PAST_SIMPLE = [{"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBD"}}]

_PAST_CONT = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBG"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "be",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "be", "TAG": "VBD"}},
]

_PRES_PERF_CONT = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBG"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "been",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "be", "TAG": "VBN"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "have",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "have", "TAG": {"IN": ["VBZ", "VBP"]}}},
]

_PAST_PERF = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBN"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "had",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "have", "TAG": "VBD"}},
]

_PAST_PERF_CONT = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBG"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "been",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "be", "TAG": "VBN"}},
]

_PERF_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": "VBN"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "aux",
     "RIGHT_ATTRS": {"DEP": "aux", "LEMMA": "have"}},
]


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _has_freq(doc):
    return any(t.lower_ in FREQ for t in doc)


def _has_irrealis(doc):
    return any((t.dep_ == "mark" and t.lemma_ in ("if", "whether"))
               or t.lemma_ == "wish" for t in doc)


def _has_time_ref(doc, v):
    for c in v.children:
        if c.dep_ == "npadvmod":
            if any(g.dep_ == "det" and g.lower_ in NEAR_DET for g in c.children):
                return True
            if c.lower_ in TIME_WORDS:
                return True
        if c.dep_ == "prep" and c.lower_ in ("at", "on", "in"):
            for g in c.children:
                if g.dep_ == "pobj" and (g.like_num or g.tag_ == "CD"
                                         or g.ent_type_ in ("TIME", "DATE")):
                    return True
    return any(t.lower_ in TIME_WORDS for t in doc)


def _is_question(doc):
    return any(t.text == "?" for t in doc)


def _has_by_the_time(doc):
    for i in range(len(doc) - 2):
        if (doc[i].lower_ == "by" and doc[i + 1].lower_ == "the"
                and doc[i + 2].lower_ == "time"):
            return True
    return False


# --------------------------------------------------------------------------- #
# Deterministic classifiers.
# --------------------------------------------------------------------------- #
def classify_pc_now(doc, tids):
    """Present-continuous DEFAULT reading (VTA-11). See module FLAG."""
    v = doc[tids[0]]
    # 'be going to' future -> not a progressive-now.
    if v.lemma_ == "go" and any(c.dep_ == "xcomp" for c in v.children):
        return None
    # annoyance (always/forever) -> VTA-14
    if any(c.dep_ == "advmod" and c.lower_ in ANNOY for c in v.children):
        return None
    # changing/developing situation -> VTA-13
    if v.lemma_ in CHANGE and any(
            c.tag_ in ("JJR", "RBR") or c.lower_ in ("more", "less")
            for c in v.children):
        return None
    # fixed arrangement / temporary situation (future or bounded time) -> 12/15
    if _has_time_ref(doc, v):
        return None
    return "VTA-11"


def classify_past_simple(doc, tids):
    v = doc[tids[0]]
    if v.tag_ != "VBD" or v.dep_ != "ROOT":
        return None
    if _has_freq(doc):
        return None                          # habit -> VTA-19 (LLM)
    if _has_irrealis(doc):
        return None                          # remote/hypothetical -> 21/22 (LLM)
    if v.lemma_ in STATIVE:
        return "VTA-20"                      # past state
    if any(c.dep_ == "conj" and c.tag_ in ("VBD", "VBN") for c in v.children):
        return "VTA-18"                      # sequence of events
    return "VTA-17"                          # completed single event


def classify_past_cont(doc, tids):
    v = doc[tids[0]]
    if any(c.dep_ == "advmod" and c.lower_ in ANNOY for c in v.children):
        return None                          # repeated/annoyance -> VTA-27
    if any(t.lower_ == "while" and t.dep_ == "mark" for t in doc):
        return "VTA-26"                      # parallel past actions
    if any(t.lower_ == "when" for t in doc):
        return None                          # interrupted/background -> 24/25
    return "VTA-23"                          # in progress at a past point


def classify_past_perf(doc, tids):
    v = doc[tids[0]]
    if any(c.dep_ == "mark" and c.lemma_ == "if" for c in v.children):
        return "VTA-42"                      # third/mixed conditional
    if v.dep_ == "ccomp" and v.head.lemma_ in REPORTING:
        return "VTA-41"                      # backshift in reported speech
    # unfulfilled hope/intention: past perfect + to-infinitive complement.
    for c in v.children:
        if c.dep_ == "xcomp" and any(g.tag_ == "TO" for g in c.children):
            return None                      # -> VTA-43 (LLM)
    if _has_by_the_time(doc):
        return "VTA-40"                      # with 'by the time'
    return "VTA-39"                          # anterior to a past point


def classify_past_perf_cont(doc, tids):
    v = doc[tids[0]]
    auxes = [c for c in v.children if c.dep_ in ("aux", "auxpass")]
    if any(a.lemma_ == "have" for a in auxes):
        return None                          # present/future perfect continuous
    if any((a.tag_ == "MD" and a.lower_ in ("'d", "had"))
           or (a.lemma_ == "have" and a.tag_ == "VBD") for a in auxes):
        return "VTA-44"
    return None


def _performative_key(doc, tids):
    v = doc[tids[0]]
    return v.lemma_.lower()


# --------------------------------------------------------------------------- #
# LLM reading prompts (skipped offline; enumerate the sibling readings).
# --------------------------------------------------------------------------- #
_PS_SYS = (
    "The bracketed present-simple verb group has one of these USES. Choose one:\n"
    "VTA-01 general/timeless truth (Water boils at 100C) | VTA-02 permanent state "
    "(She comes from Spain) | VTA-03 habit/routine (I get up at seven) | "
    "VTA-04 instructions/directions (You take the second left) | "
    "VTA-05 commentary/demonstration (He passes, he shoots) | "
    "VTA-06 narrative/historic present (So I walk in and he stares) | "
    "VTA-08 headline/caption (Minister resigns) | VTA-09 scheduled future "
    "(The train leaves at six) | VTA-10 present in a time/conditional clause with "
    "future reference (When he arrives, we'll start)."
)
_PC_SYS = (
    "The bracketed present-continuous verb group has one of these USES. Choose "
    "one:\nVTA-12 temporary situation (staying this week) | VTA-13 changing/"
    "developing situation (getting warmer) | VTA-14 repeated action + annoyance "
    "(always interrupting) | VTA-15 fixed future arrangement (meeting at six) | "
    "VTA-16 background in a present narrative (it's raining and people are "
    "running).\nA plain action in progress right now (I'm reading your draft) "
    "is a different construct - return NONE for it."
)
_PST_SYS = (
    "The bracketed past-simple verb group has one of these USES. Choose one:\n"
    "VTA-19 past habit/repeated action (walked to school every day) | "
    "VTA-21 remote/polite/tentative (I wondered if you could help) | "
    "VTA-22 hypothetical/unreal past-tense form (If I knew; I wish I had; "
    "It's time we left).\nA plain single past event or state (He had long "
    "hair then; I saw her yesterday) is a different construct - return NONE "
    "for it."
)
_PSTC_SYS = (
    "The bracketed past-continuous verb group has one of these USES. Choose one:\n"
    "VTA-24 interrupted action (was cooking when the phone rang) | "
    "VTA-25 background to past events (the sun was shining when we set off) | "
    "VTA-27 temporary past situation / repeated annoyance (was always losing) | "
    "VTA-28 polite/tentative (I was wondering whether).\n"
    "An action simply in progress at a stated time point (At nine I was "
    "working) and two simultaneous actions (While I cooked, she was setting "
    "the table) are different constructs - return NONE for those."
)
_PPC_SYS = (
    "The bracketed present-perfect-continuous verb group has one of these USES. "
    "Choose one:\nVTA-35 duration of an activity up to now (waiting for an hour) | "
    "VTA-36 recent activity with present evidence (your eyes are red, been crying) "
    "| VTA-37 temporary/repeated recent activity (been going to the gym lately) | "
    "VTA-38 activity-focus vs result-focus contrast with the simple perfect."
)
_PPF_SYS = (
    "Return VTA-43 if this past-perfect verb group expresses an UNFULFILLED hope "
    "or intention (I had hoped to see you; we had intended to leave earlier)."
)
_V45_SYS = ("Return VTA-45 if a past-perfect-continuous explains the CAUSE of a "
            "past state/result (His eyes were red; he'd been crying).")
_V46_SYS = ("Return VTA-46 if a stative verb (cognition/perception/possession/"
            "emotion: know, believe, own, like) is used in the simple, RESISTING "
            "the progressive. If the stative verb actually appears IN the "
            "progressive (I'm loving it), that is not this construct - return "
            "NONE.")
_V47_SYS = ("Return VTA-47 if it contrasts a stative perception verb (see/hear) "
            "with an active one (look/listen/watch).")
_V48_SYS = ("Return VTA-48 if a normally stative verb is COERCED into the "
            "progressive for a dynamic/temporary reading (I'm loving it; you're "
            "being silly).")
_V49_SYS = ("Return VTA-49 if a subordinate clause shows sequence-of-tenses / "
            "backshift agreement (He said that he was tired).")


# --------------------------------------------------------------------------- #
def build(nlp, client=None):
    dets = []

    # --- Present simple: readings (LLM) + performative (lexicon). ------------
    dets.append(LLMReadingDetector(
        nlp, ["VTA-01", "VTA-02", "VTA-03", "VTA-04", "VTA-05", "VTA-06",
              "VTA-08", "VTA-09", "VTA-10"],
        _PRES_SIMPLE, _PS_SYS, client=client, version="vta-pres-simple@0.1"))

    _perf_form = [
        {"RIGHT_ID": "v", "RIGHT_ATTRS": {"TAG": {"IN": ["VBP", "VBZ"]},
                                          "DEP": "ROOT"}},
        {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "s",
         "RIGHT_ATTRS": {"DEP": "nsubj", "LOWER": {"IN": ["i", "we"]}}},
    ]
    dets.append(LexiconRuleDetector(
        nlp, "VTA-07", _perf_form, PERFORMATIVE, _performative_key,
        version="vta07-performative@0.1"))

    # --- Present continuous: 'now' (rule default) + readings (LLM). ----------
    dets.append(RuleRoutingDetector(
        nlp, ["VTA-11"], _PRES_CONT, classify_pc_now,
        version="vta11-now@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["VTA-12", "VTA-13", "VTA-14", "VTA-15", "VTA-16"],
        _PRES_CONT, _PC_SYS, client=client, version="vta-pres-cont@0.1"))

    # --- Past simple: form-decidable readings (rule) + rest (LLM). -----------
    dets.append(RuleRoutingDetector(
        nlp, ["VTA-17", "VTA-18", "VTA-20"], _PAST_SIMPLE, classify_past_simple,
        version="vta-past-simple@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["VTA-19", "VTA-21", "VTA-22"], _PAST_SIMPLE, _PST_SYS,
        client=client, version="vta-past-simple-llm@0.1"))

    # --- Past continuous: 'at a point' / 'parallel' (rule) + rest (LLM). -----
    dets.append(RuleRoutingDetector(
        nlp, ["VTA-23", "VTA-26"], _PAST_CONT, classify_past_cont,
        version="vta-past-cont@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["VTA-24", "VTA-25", "VTA-27", "VTA-28"], _PAST_CONT, _PSTC_SYS,
        client=client, version="vta-past-cont-llm@0.1"))

    # VTA-29..34 (present perfect simple) handled by reference.py -> skip.

    # --- Present perfect continuous: all readings (LLM). ---------------------
    dets.append(LLMReadingDetector(
        nlp, ["VTA-35", "VTA-36", "VTA-37", "VTA-38"], _PRES_PERF_CONT, _PPC_SYS,
        client=client, version="vta-pres-perf-cont@0.1"))

    # --- Past perfect simple: form-decidable readings (rule) + hope (LLM). ---
    dets.append(RuleRoutingDetector(
        nlp, ["VTA-39", "VTA-40", "VTA-41", "VTA-42"], _PAST_PERF,
        classify_past_perf, version="vta-past-perf@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["VTA-43"], _PAST_PERF, _PPF_SYS, client=client,
        version="vta43-hope@0.1"))

    # --- Past perfect continuous: form (rule) + cause reading (LLM). ---------
    dets.append(RuleRoutingDetector(
        nlp, ["VTA-44"], _PAST_PERF_CONT, classify_past_perf_cont,
        version="vta44-ppc@0.1"))
    dets.append(LLMStandaloneDetector(
        "VTA-45", _V45_SYS, client=client, version="vta45-cause@0.1"))

    # --- Lexical-aspect constraints (LLM/standalone). -----------------------
    dets.append(LLMStandaloneDetector("VTA-46", _V46_SYS, client=client,
                                      version="vta46-stative@0.1"))
    dets.append(LLMStandaloneDetector("VTA-47", _V47_SYS, client=client,
                                      version="vta47-perception@0.1"))
    dets.append(LLMStandaloneDetector("VTA-48", _V48_SYS, client=client,
                                      version="vta48-coercion@0.1"))
    dets.append(LLMStandaloneDetector("VTA-49", _V49_SYS, client=client,
                                      version="vta49-sot@0.1"))

    return dets
