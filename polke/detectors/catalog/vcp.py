"""Category VCP - verb-complementation patterns (VCP-01..40).

The category is a grid of clause frames keyed on the FORM of the verb's
complement(s). Siblings mostly share a shape and differ by (a) the dependency of
the complement (dobj / ccomp / xcomp / oprd / prep+pobj / dative), (b) the tag of
the complement head (JJ vs NN, VB bare-inf vs TO-inf vs VBG vs VBN), and (c) the
lexical class of the matrix verb (seem- vs become-copulas; make/let causatives vs
want/expect object-control; think- vs insist-clause verbs). So most detectors are
``RuleRoutingDetector`` (form + deterministic classify) or ``LexiconRuleDetector``
(form gated by a verb / verb+prep table). Category verb lists live here as Python
constants (data/lexicons.json is shared and left untouched).

Reading-dependent siblings stay in the LLM tier and skip offline:
  VCP-05/06/07  ergative vs middle vs reciprocal (all look like plain intransitive)
  VCP-18        to-inf vs -ing meaning change (stop to rest / stop resting)
  VCP-31/32/33  perception bare-inf / -ing / past-participle readings
  VCP-35        anticipatory-it complement
  VCP-39        cognate object

spaCy en_core_web_sm labels verified empirically. Note the parser is noisy on the
inventory's bare fragments (e.g. object predicatives surface as oprd, ccomp-with-
nsubj, or a second dobj; "paint it red" tags red VBD; "appoint sb director" loses
its verb entirely) - the classifiers absorb those variants.
"""
from __future__ import annotations
from ..rules import DependencyRuleDetector
from ..routing import RuleRoutingDetector, LexiconRuleDetector, CompositeDetector
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector
from ..base import Detector
from ...schema import Annotation, Span

# --------------------------------------------------------------------------- #
# Category lexicons (Python constants; not in shared data/lexicons.json).
# --------------------------------------------------------------------------- #
_WH_TAGS = {"WP", "WDT", "WP$", "WRB"}
_REFLEXIVES = {"myself", "yourself", "himself", "herself", "itself",
               "ourselves", "yourselves", "themselves", "oneself"}

# VCP-01 pure intransitive must reject ergative/inchoative verbs (-> VCP-05).
_ERGATIVE = {
    "open", "close", "shut", "break", "shatter", "crack", "smash", "snap",
    "tear", "split", "burst", "increase", "decrease", "rise", "fall", "drop",
    "grow", "shrink", "melt", "freeze", "boil", "burn", "cook", "dry", "cool",
    "warm", "heat", "change", "improve", "worsen", "move", "shift", "turn",
    "roll", "spin", "bounce", "sink", "float", "start", "stop", "begin", "end",
    "continue", "expand", "contract", "widen", "narrow", "lengthen", "shorten",
    "brighten", "darken", "empty", "fill", "spread", "spill", "scatter", "drown",
    "wake", "bend", "stretch", "slow", "speed", "double", "triple", "form",
}
# VCP-02 obligatory adverbial: directional / locative adverbs.
_LOC_DIR_ADV = {
    "abroad", "home", "here", "there", "away", "back", "out", "outside",
    "inside", "upstairs", "downstairs", "ahead", "apart", "aside", "indoors",
    "outdoors", "nearby", "everywhere", "somewhere", "anywhere", "north",
    "south", "east", "west", "forward", "forwards", "backward", "backwards",
    "up", "down", "in", "on", "off", "over", "past", "round", "around",
    "left", "right",
}

# VCP-03 / VCP-04 copulas (route by class; -> VCP-03 current, VCP-04 resulting).
_COP_CURRENT = {"seem", "appear", "look", "sound", "feel", "taste", "smell"}
_COP_RESULT = {"become", "get", "go", "turn", "grow", "come", "fall", "prove",
               "remain", "stay", "end", "run", "wax"}

# VCP-11 think-class that-clause verbs (finite, no IO, non-mandative).
_THINK_CLASS = {
    "think", "believe", "know", "say", "hope", "suppose", "guess", "feel",
    "reckon", "imagine", "assume", "expect", "claim", "doubt", "realize",
    "realise", "notice", "agree", "mean", "understand", "remember", "forget",
    "decide", "hear", "find", "consider", "argue", "explain", "mention",
    "reply", "add", "worry", "fear", "wish", "pretend", "conclude", "declare",
    "estimate", "predict", "accept", "admit", "deny", "confirm", "state",
    "observe", "report", "reveal", "sense", "trust", "bet", "complain",
    "respond", "acknowledge", "assert", "maintain", "figure", "gather",
}
# VCP-12 mandative subjunctive verbs.
_MANDATIVE = {"insist", "demand", "suggest", "recommend", "require", "propose",
              "request", "urge", "order", "ask", "advise", "move", "direct",
              "stipulate", "command", "prefer", "beg", "desire", "intend",
              "decree", "resolve"}
# VCP-24 V + IO + that-clause verbs.
_ASSURE_CLASS = {"assure", "tell", "convince", "inform", "promise", "remind",
                 "warn", "persuade", "notify", "advise", "show", "teach",
                 "reassure", "guarantee", "satisfy"}

# VCP-15 subject-control to-infinitive verbs (to-inf only).
_TO_INF = {
    "agree", "decide", "hope", "promise", "refuse", "offer", "plan", "manage",
    "learn", "want", "wish", "choose", "attempt", "threaten", "pretend",
    "arrange", "afford", "deserve", "fail", "seem", "appear", "tend", "happen",
    "prove", "claim", "prepare", "volunteer", "care", "dare", "hesitate",
    "long", "swear", "vow", "aim", "expect", "demand", "ask", "consent",
    "resolve", "struggle", "strive", "endeavour", "endeavor", "seek",
}
# VCP-16 gerund-only verbs (-ing only).
_ING_ONLY = {
    "admit", "enjoy", "avoid", "finish", "deny", "consider", "suggest", "mind",
    "keep", "miss", "practise", "practice", "risk", "imagine", "involve",
    "delay", "postpone", "resist", "escape", "fancy", "appreciate", "dislike",
    "quit", "recall", "resent", "understand", "tolerate", "detest", "loathe",
    "contemplate", "advocate", "anticipate", "entail", "acknowledge", "report",
    "recommend", "discuss",
}
# VCP-17 verbs that take to-inf OR -ing with little meaning change.
_BOTH = {"begin", "start", "continue", "cease", "like", "love", "hate",
         "prefer", "bother", "intend", "propose", "neglect", "attempt", "omit",
         "dread", "can't stand"}
# VCP-18 meaning-change verbs (LLM tier).
_MEANING_CHANGE = {"stop", "remember", "forget", "regret", "try", "go on",
                   "mean", "come", "need"}

# VCP-29 object-control / ECM to-infinitive verbs.
_OBJ_CONTROL = {"want", "expect", "tell", "ask", "allow", "enable", "force",
                "cause", "get", "order", "advise", "encourage", "permit",
                "persuade", "remind", "warn", "invite", "teach", "need",
                "prefer", "urge", "require", "request", "beg", "instruct",
                "compel", "oblige", "press", "help", "like", "wish", "expect",
                "trust", "know", "believe", "consider", "declare", "prompt"}
# VCP-30 causative bare-infinitive verbs.
_CAUSATIVE = {"make", "let", "have", "bid", "help"}

# VCP-22 dative-shift (to-NP) verbs.
_DATIVE_TO = {"give", "send", "offer", "lend", "show", "tell", "pass", "hand",
              "teach", "sell", "pay", "bring", "read", "write", "owe",
              "promise", "throw", "award", "grant", "leave", "serve", "post",
              "mail", "feed", "wish", "deny", "sing", "explain", "describe",
              "announce", "introduce", "mention", "suggest", "report", "prove"}
# VCP-23 benefactive (for-NP) verbs.
_BENEFACTIVE = {"buy", "make", "cook", "find", "get", "save", "build", "order",
                "book", "fetch", "choose", "keep", "leave", "pour", "prepare",
                "bake", "knit", "paint", "design", "reserve", "grab", "win",
                "pick", "call", "sing"}

# VCP-28 V + O + as-phrase verbs.
_AS_VERBS = {"regard", "view", "describe", "see", "treat", "define", "know",
             "recognize", "recognise", "class", "classify", "characterize",
             "characterise", "portray", "perceive", "interpret", "identify",
             "accept", "cite", "count", "rate", "brand", "cast", "hail",
             "dismiss", "label", "denounce", "mark"}

# VCP-27 naming / appointing verbs (surface fallback for broken fragments).
_NAMING = {"appoint", "name", "elect", "call", "declare", "crown", "designate",
           "dub", "proclaim", "nominate", "christen", "label", "brand", "term",
           "style", "title", "rename", "deem", "vote", "pronounce", "consider",
           "judge", "make"}

# Verbs that take an object predicative (VCP-26/27) even when spaCy mis-parses
# the predicate; used to gate the evidence-poor objpred branches.
_OBJPRED_VERBS = _NAMING | {"paint", "keep", "find", "drive", "leave", "get",
                            "turn", "render", "wipe", "cut", "hold", "prove",
                            "believe", "presume", "imagine", "want", "like",
                            "prefer", "push", "set", "send"}

# VCP-19 / VCP-20 prepositional-verb table (shared; form splits noun vs -ing).
_PREP_VERB = {
    ("depend", "on"), ("depend", "upon"), ("look", "at"), ("look", "for"),
    ("look", "after"), ("look", "into"), ("listen", "to"), ("wait", "for"),
    ("rely", "on"), ("belong", "to"), ("consist", "of"), ("refer", "to"),
    ("apply", "for"), ("deal", "with"), ("care", "for"), ("hope", "for"),
    ("account", "for"), ("agree", "with"), ("agree", "on"), ("approve", "of"),
    ("believe", "in"), ("benefit", "from"), ("complain", "about"),
    ("cope", "with"), ("laugh", "at"), ("pay", "for"), ("prepare", "for"),
    ("react", "to"), ("result", "in"), ("search", "for"), ("suffer", "from"),
    ("talk", "about"), ("talk", "to"), ("think", "about"), ("think", "of"),
    ("worry", "about"), ("arrive", "at"), ("comment", "on"),
    ("concentrate", "on"), ("decide", "on"), ("insist", "on"), ("object", "to"),
    ("participate", "in"), ("respond", "to"), ("specialize", "in"),
    ("subscribe", "to"), ("sympathize", "with"), ("vote", "for"),
    ("work", "on"), ("succeed", "in"), ("apologize", "for"),
    ("apologise", "for"), ("dream", "of"), ("dream", "about"),
    ("give", "up"), ("carry", "on"), ("keep", "on"), ("go", "on"),
    ("focus", "on"), ("respond", "to"), ("count", "on"), ("bank", "on"),
    ("long", "for"), ("aim", "at"), ("glance", "at"), ("stare", "at"),
    ("smile", "at"), ("shout", "at"), ("aim", "for"), ("head", "for"),
    ("run", "into"), ("come", "across"), ("call", "for"), ("stand", "for"),
}
# VCP-34 V + O + prep + NP table.
_OBJ_PREP = {
    ("accuse", "of"), ("remind", "of"), ("thank", "for"),
    ("congratulate", "on"), ("blame", "for"), ("blame", "on"),
    ("provide", "with"), ("prevent", "from"), ("suspect", "of"),
    ("warn", "about"), ("warn", "of"), ("inform", "of"), ("deprive", "of"),
    ("rob", "of"), ("convince", "of"), ("cure", "of"), ("compare", "with"),
    ("compare", "to"), ("protect", "from"), ("save", "from"), ("spend", "on"),
    ("base", "on"), ("forgive", "for"), ("punish", "for"), ("praise", "for"),
    ("criticize", "for"), ("criticise", "for"), ("add", "to"),
    ("apply", "to"), ("attribute", "to"), ("dedicate", "to"),
    ("devote", "to"), ("introduce", "to"), ("invite", "to"), ("lend", "to"),
    ("prefer", "to"), ("sentence", "to"), ("expose", "to"), ("treat", "for"),
    ("charge", "with"), ("supply", "with"), ("associate", "with"),
    ("combine", "with"), ("replace", "with"), ("share", "with"),
    ("ban", "from"), ("stop", "from"), ("keep", "from"), ("discourage", "from"),
    ("congratulate", "for"), ("remind", "about"), ("tell", "about"),
    ("ask", "for"), ("borrow", "from"), ("steal", "from"), ("translate", "into"),
    ("turn", "into"), ("divide", "into"), ("put", "on"), ("base", "upon"),
}

# VCP-37 light / delexical verb + NP collocations (verb lemma, noun lemma).
_LIGHT_VERB = {
    ("have", "look"), ("have", "rest"), ("have", "shower"), ("have", "bath"),
    ("have", "drink"), ("have", "chat"), ("have", "go"), ("have", "try"),
    ("have", "walk"), ("have", "swim"), ("have", "nap"), ("have", "break"),
    ("have", "word"), ("have", "think"), ("have", "meal"), ("have", "seat"),
    ("take", "look"), ("take", "break"), ("take", "photo"), ("take", "rest"),
    ("take", "walk"), ("take", "seat"), ("take", "nap"), ("take", "shower"),
    ("take", "bath"), ("take", "step"), ("take", "chance"), ("take", "risk"),
    ("take", "turn"), ("take", "bite"), ("take", "sip"), ("take", "breath"),
    ("make", "decision"), ("make", "mistake"), ("make", "effort"),
    ("make", "call"), ("make", "choice"), ("make", "suggestion"),
    ("make", "point"), ("make", "comment"), ("make", "attempt"),
    ("make", "start"), ("make", "move"), ("make", "noise"), ("make", "wish"),
    ("make", "promise"), ("make", "plan"), ("make", "change"),
    ("do", "shopping"), ("do", "washing"), ("do", "cleaning"),
    ("do", "homework"), ("do", "job"), ("do", "work"), ("do", "cooking"),
    ("do", "damage"), ("do", "favour"), ("do", "favor"), ("do", "harm"),
    ("give", "smile"), ("give", "sigh"), ("give", "laugh"), ("give", "look"),
    ("give", "kiss"), ("give", "hug"), ("give", "call"), ("give", "talk"),
    ("give", "shout"), ("give", "cry"), ("give", "answer"), ("give", "speech"),
    ("give", "nod"), ("give", "wave"), ("give", "yawn"), ("give", "cough"),
}
# VCP-38 fixed verb idioms (LEMMA phrases).
_IDIOMS = [
    "take place", "make sense", "pay attention", "take part", "keep an eye on",
    "bear in mind", "take care", "make progress", "take advantage of",
    "get rid of", "make up your mind", "make up one 's mind", "take turns",
    "take charge", "make room", "lose track", "keep track", "take note",
    "make a fuss", "take pride", "make believe", "give way", "take heart",
    "catch sight of", "come to terms", "keep pace", "make headway",
    "run the risk", "take office", "hold office", "break the ice",
    "call the shots", "change your mind", "keep in touch", "lose face",
]

# route tables ------------------------------------------------------------- #
_NONFIN_ROUTE = {}
for _v in _TO_INF:
    _NONFIN_ROUTE[_v] = "VCP-15"
for _v in _ING_ONLY:
    _NONFIN_ROUTE.setdefault(_v, "VCP-16")
for _v in _BOTH:
    _NONFIN_ROUTE[_v] = "VCP-17"       # both-verbs win the shared form
_NONFIN_VERBS = set(_NONFIN_ROUTE)

_COP_ROUTE = {v: "VCP-03" for v in _COP_CURRENT}
_COP_ROUTE.update({v: "VCP-04" for v in _COP_RESULT})
_COP_VERBS = set(_COP_ROUTE)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _wh_in(root):
    return any(t.tag_ in _WH_TAGS for t in root.subtree)


def _nominal_io(v):
    """True if the verb has an indirect-object nominal (dative or a PRON/NOUN dobj)."""
    for c in v.children:
        if c.dep_ == "dative":
            return True
        if c.dep_ == "dobj" and c.pos_ in ("PRON", "NOUN", "PROPN"):
            return True
    return False


def _has_to(tok):
    return any(k.tag_ == "TO" for k in tok.children)


def _child_deps(v):
    return {c.dep_ for c in v.children}


def _is_nominal(tok):
    if tok.pos_ in ("NOUN", "PROPN"):
        return True
    if tok.tag_ in ("NN", "NNS", "NNP", "NNPS"):
        return True
    for k in tok.children:
        if k.dep_ in ("det", "nummod", "poss", "compound") or k.tag_ in ("DT", "CD", "PRP$"):
            return True
    return False


# --------------------------------------------------------------------------- #
# VCP-01 / VCP-02 : intransitive frames (pure vs obligatory adverbial).
# --------------------------------------------------------------------------- #
_ROOT_VERB = [{"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}}]
_COMPLEMENT_DEPS = {"dobj", "dative", "ccomp", "xcomp", "acomp", "attr", "oprd",
                    "prep", "agent", "expl", "prt", "dative"}


def _classify_intrans(doc, ids):
    v = doc[ids[0]]
    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in v.children):
        return None
    deps = _child_deps(v)
    if deps & _COMPLEMENT_DEPS:
        return None
    # adverbials
    advmods = [c for c in v.children if c.dep_ == "advmod"]
    npadv = [c for c in v.children if c.dep_ == "npadvmod"]
    loc = [a for a in advmods if a.lower_ in _LOC_DIR_ADV]
    if npadv or loc:
        return "VCP-02"
    if advmods:                       # non-locative adverb: manner -> still VCP-01
        pass
    if v.lemma_.lower() in _ERGATIVE:
        return None                   # ergative/inchoative -> VCP-05 (LLM)
    return "VCP-01"


# --------------------------------------------------------------------------- #
# VCP-03 / VCP-04 : subject predicative (copular), routed by verb class.
# --------------------------------------------------------------------------- #
_COP_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "p",
     "RIGHT_ATTRS": {"DEP": {"IN": ["oprd", "acomp", "attr"]}}},
]


def _cop_key(doc, ids):
    v = doc[ids[0]]
    if any(c.dep_ == "dobj" for c in v.children):
        return None                    # object present -> complex-transitive, not copular
    lemma = v.lemma_.lower()
    return lemma if lemma in _COP_VERBS else None


# --------------------------------------------------------------------------- #
# VCP-08 : monotransitive V + NP object.
# --------------------------------------------------------------------------- #
_TRANS_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "o", "RIGHT_ATTRS": {"DEP": "dobj"}},
]


def _classify_mono(doc, ids):
    v, obj = doc[ids[0]], doc[ids[1]]
    if obj.lower_ in _REFLEXIVES:
        return None                                   # VCP-09
    if obj.lower_ in ("other", "another", "one"):
        return None                                   # VCP-10 reciprocal
    deps = _child_deps(v)
    if "dative" in deps or "oprd" in deps or "acomp" in deps or "attr" in deps:
        return None                                   # ditransitive / object-predicative
    if len([c for c in v.children if c.dep_ == "dobj"]) >= 2:
        return None                                   # VCP-27 double
    if any(k.dep_ == "poss" for k in obj.children):
        return None                                   # elect her chair -> VCP-27
    if (v.lemma_.lower(), obj.lemma_.lower()) in _LIGHT_VERB:
        return None                                   # VCP-37 light verb
    return "VCP-08"


# --------------------------------------------------------------------------- #
# VCP-09 : reflexive object.
# --------------------------------------------------------------------------- #
def _classify_reflexive(doc, ids):
    obj = doc[ids[1]]
    return "VCP-09" if obj.lower_ in _REFLEXIVES else None


# --------------------------------------------------------------------------- #
# VCP-11 / VCP-12 : that-clause (think-class vs mandative), no IO, non-wh.
# --------------------------------------------------------------------------- #
_CCOMP_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "cc", "RIGHT_ATTRS": {"DEP": "ccomp"}},
]


def _that_key(doc, ids):
    v, cc = doc[ids[0]], doc[ids[1]]
    if _nominal_io(v):
        return None
    if _wh_in(cc):
        return None
    return v.lemma_.lower()


# --------------------------------------------------------------------------- #
# VCP-13 : monotransitive wh-clause (finite).
# --------------------------------------------------------------------------- #
def _classify_whclause(doc, ids):
    v, cc = doc[ids[0]], doc[ids[1]]
    if _nominal_io(v):
        return None                                   # -> VCP-25
    if not _wh_in(cc):
        return None
    return "VCP-13"


# --------------------------------------------------------------------------- #
# VCP-14 : V + wh + to-infinitive.
# --------------------------------------------------------------------------- #
_XCOMP_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "xc", "RIGHT_ATTRS": {"DEP": "xcomp"}},
]


def _classify_whinf(doc, ids):
    xc = doc[ids[1]]
    if _has_to(xc) and _wh_in(xc):
        return "VCP-14"
    return None


# --------------------------------------------------------------------------- #
# VCP-15 / VCP-16 / VCP-17 : subject-control to-inf / -ing / either.
# --------------------------------------------------------------------------- #
def _nonfinite_comp(v):
    """The verb's non-finite complement token, or None. Tolerates the noisy
    'started to rain / raining' fragment where the parser drops the xcomp."""
    for c in v.children:
        if c.dep_ == "xcomp" and (c.tag_ in ("VB", "VBG") or _has_to(c)):
            return c
        if c.dep_ == "dobj" and c.tag_ == "VBG":
            return c
    for c in v.subtree:                                # garbled to-infinitive
        if c is not v and c.tag_ == "VB" and _has_to(c):
            return c
    return None


def _nonfin_key(doc, ids):
    v = doc[ids[0]]
    if v.lemma_.lower() not in _NONFIN_VERBS:
        return None
    comp = _nonfinite_comp(v)
    if comp is None:
        return None
    if any(k.dep_ in ("nsubj", "nsubjpass") for k in comp.children):
        return None                                   # embedded subject -> VCP-29
    return v.lemma_.lower()


def _nonfin_route(key):
    return _NONFIN_ROUTE.get(key)


# --------------------------------------------------------------------------- #
# VCP-19 / VCP-20 : prepositional verb + NP / + -ing.
# --------------------------------------------------------------------------- #
_PREP_NP_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "p", "RIGHT_ATTRS": {"DEP": "prep"}},
    {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "o", "RIGHT_ATTRS": {"DEP": "pobj"}},
]
_PREP_ING_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "p", "RIGHT_ATTRS": {"DEP": "prep"}},
    {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "g", "RIGHT_ATTRS": {"DEP": "pcomp"}},
]


def _prep_key(doc, ids):
    v, p = doc[ids[0]], doc[ids[1]]
    if any(c.dep_ == "dobj" for c in v.children):
        return None                                   # -> VCP-34
    return (v.lemma_.lower(), p.lower_)


# --------------------------------------------------------------------------- #
# VCP-21 : double-object ditransitive (V + IO + DO).
# --------------------------------------------------------------------------- #
_DITRANS_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "io", "RIGHT_ATTRS": {"DEP": "dative"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "do", "RIGHT_ATTRS": {"DEP": "dobj"}},
]


def _classify_ditrans(doc, ids):
    io = doc[ids[1]]
    if io.tag_ == "IN":               # "give a book to him" -> to-dative -> VCP-22
        return None
    return "VCP-21"


# --------------------------------------------------------------------------- #
# VCP-22 / VCP-23 : dative shift (to-NP) / benefactive (for-NP).
# --------------------------------------------------------------------------- #
_SHIFT_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "do", "RIGHT_ATTRS": {"DEP": "dobj"}},
    {"LEFT_ID": "v", "REL_OP": ">>", "RIGHT_ID": "p",
     "RIGHT_ATTRS": {"DEP": {"IN": ["dative", "prep"]}, "TAG": "IN"}},
    {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "po", "RIGHT_ATTRS": {"DEP": "pobj"}},
]


def _shift_key(doc, ids):
    v, p = doc[ids[0]], doc[ids[2]]
    return (v.lemma_.lower(), p.lower_)


def _shift_route(key):
    lemma, prep = key
    if prep == "to" and lemma in _DATIVE_TO:
        return "VCP-22"
    if prep == "for" and lemma in _BENEFACTIVE:
        return "VCP-23"
    return None


# --------------------------------------------------------------------------- #
# VCP-24 : V + IO + that-clause.
# --------------------------------------------------------------------------- #
_IO_CCOMP_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "io",
     "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "dative"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "cc", "RIGHT_ATTRS": {"DEP": "ccomp"}},
]


def _io_that_key(doc, ids):
    v, cc = doc[ids[0]], doc[ids[2]]
    if _wh_in(cc):
        return None                                   # -> VCP-25
    return v.lemma_.lower()


# --------------------------------------------------------------------------- #
# VCP-25 : V + IO + wh-clause.
# --------------------------------------------------------------------------- #
_IO_WH_VERBS = {"tell", "ask", "show", "teach", "remind", "advise", "inform",
                "explain"}


def _classify_io_wh(doc, ids):
    v, cc = doc[ids[0]], doc[ids[2]]
    if v.lemma_.lower() not in _IO_WH_VERBS:
        return None
    return "VCP-25" if _wh_in(cc) else None


# --------------------------------------------------------------------------- #
# VCP-26 / VCP-27 : object predicative (adjective / noun).
# --------------------------------------------------------------------------- #
def _classify_objpred(doc, ids):
    v = doc[ids[0]]
    if v.pos_ != "VERB":
        return None
    kids = list(v.children)
    oprd = [c for c in kids if c.dep_ == "oprd"]
    dobjs = [c for c in kids if c.dep_ == "dobj"]
    ccomps = [c for c in kids if c.dep_ == "ccomp"]

    # A) explicit oprd predicate + a direct object.
    if oprd and dobjs:
        p = oprd[0]
        return "VCP-27" if _is_nominal(p) else "VCP-26"

    # B) predicate raised as ccomp whose nsubj is the object (no to, no bare-inf).
    for cc in ccomps:
        if not any(k.dep_ in ("nsubj", "nsubjpass") for k in cc.children):
            continue
        if cc.pos_ in ("VERB", "AUX"):
            # a verbal ccomp is normally a finite complement clause (VCP-11/13);
            # accept it as a mis-tagged predicate adjective only for verbs that
            # take object predicatives and only when the "clause" is bare.
            if v.lemma_.lower() not in _OBJPRED_VERBS:
                continue
            if any(k.dep_ not in ("nsubj", "nsubjpass") for k in cc.children):
                continue
        if _has_to(cc) or cc.tag_ in ("VB", "VBN"):
            continue                                  # -> VCP-29/30/33
        if cc.pos_ in ("NOUN", "PROPN") or _is_nominal(cc):
            return "VCP-27"
        return "VCP-26"                               # JJ or mis-tagged VBD adjective

    # C/D below have no structural predicate evidence, so they are restricted
    # to the naming/considering verb class.
    if v.lemma_.lower() not in _NAMING:
        return None

    # C) predicate = a second dobj, or a dobj carrying the real object as poss.
    if len(dobjs) >= 2:
        p = dobjs[-1]
        return "VCP-27" if _is_nominal(p) else "VCP-26"
    if len(dobjs) == 1:
        d = dobjs[0]
        if any(k.dep_ == "poss" for k in d.children):
            nominal = d.pos_ in ("NOUN", "PROPN") or d.tag_ in ("NN", "NNS", "NNP", "NNPS")
            return "VCP-27" if nominal else "VCP-26"
        # D) broken fragment "name it X": bare predicate token after the object.
        for k in range(d.i + 1, len(doc)):
            t = doc[k]
            if t.is_space:
                continue
            if t.is_alpha and t.pos_ not in ("ADP", "AUX", "PART", "CCONJ",
                                             "SCONJ", "DET", "PRON"):
                if t.pos_ == "ADJ" or t.tag_ in ("JJ", "JJR", "JJS"):
                    return "VCP-26"
                return "VCP-27"
            break
    return None


class _SurfaceObjPredNP(Detector):
    """Surface fallback for VCP-27 fragments spaCy mis-parses (e.g. rootless
    'appoint sb director'): naming/appointing verb + object NP + predicate noun."""
    detector_type = "rule"

    def __init__(self, nlp, version="vcp27-surface@0.1"):
        from spacy.matcher import Matcher
        self.construct_ids = ["VCP-27"]
        self.version = version
        self._m = Matcher(nlp.vocab)
        self._m.add("np", [[
            {"LEMMA": {"IN": sorted(_NAMING)}},
            {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}, "OP": "+"},
            {"POS": {"IN": ["NOUN", "PROPN"]}},
        ]])

    def match(self, doc, text_id="doc"):
        out, seen = [], set()
        for _mid, start, end in self._m(doc):
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            span = doc[start:end]
            out.append(Annotation(
                text_id=text_id, construct_id="VCP-27",
                span=Span(span.start_char, span.end_char, start, end - 1),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": list(range(start, end)), "matched": span.text}))
        return out


# --------------------------------------------------------------------------- #
# VCP-28 : V + O + as-phrase.
# --------------------------------------------------------------------------- #
_AS_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "o", "RIGHT_ATTRS": {"DEP": "dobj"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "as",
     "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "as"}},
]


def _as_key(doc, ids):
    v = doc[ids[0]]
    lemma = v.lemma_.lower()
    return lemma if lemma in _AS_VERBS else None


# --------------------------------------------------------------------------- #
# VCP-29 : V + O + to-infinitive (object control / ECM).
# --------------------------------------------------------------------------- #
_OBJ_INF_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "cc",
     "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}}},
]


def _objctrl_key(doc, ids):
    v, cc = doc[ids[0]], doc[ids[1]]
    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in cc.children):
        return None
    if not _has_to(cc):
        return None                                   # bare-inf -> VCP-30/31
    return v.lemma_.lower()


# --------------------------------------------------------------------------- #
# VCP-30 : V + O + bare infinitive (causative make/let/have).
# --------------------------------------------------------------------------- #
def _causative_key(doc, ids):
    v, cc = doc[ids[0]], doc[ids[1]]
    if v.lemma_.lower() not in _CAUSATIVE:
        return None
    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in cc.children):
        return None
    if _has_to(cc) or cc.tag_ != "VB":
        return None
    return v.lemma_.lower()


# --------------------------------------------------------------------------- #
# VCP-34 : V + O + prep + NP.
# --------------------------------------------------------------------------- #
_OBJ_PREP_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "o", "RIGHT_ATTRS": {"DEP": "dobj"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "p", "RIGHT_ATTRS": {"DEP": "prep"}},
    {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "po", "RIGHT_ATTRS": {"DEP": "pobj"}},
]


def _obj_prep_key(doc, ids):
    v, p = doc[ids[0]], doc[ids[2]]
    return (v.lemma_.lower(), p.lower_)


# --------------------------------------------------------------------------- #
# VCP-36 : existential / presentational (there + V).
# --------------------------------------------------------------------------- #
_EXPL_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "ex", "RIGHT_ATTRS": {"DEP": "expl"}},
]


def _classify_expl(doc, ids):
    ex = doc[ids[1]]
    return "VCP-36" if ex.lower_ == "there" else None


# --------------------------------------------------------------------------- #
# VCP-37 : light / delexical verb + NP.
# --------------------------------------------------------------------------- #
_LIGHT_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "o", "RIGHT_ATTRS": {"DEP": "dobj"}},
]


def _light_key(doc, ids):
    v, o = doc[ids[0]], doc[ids[1]]
    return (v.lemma_.lower(), o.lemma_.lower())


# --------------------------------------------------------------------------- #
# VCP-40 : catenative chains (stacked non-finite xcomp).
# --------------------------------------------------------------------------- #
_CATENATIVE_FORM = [
    {"RIGHT_ID": "v1", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
    {"LEFT_ID": "v1", "REL_OP": ">", "RIGHT_ID": "v2", "RIGHT_ATTRS": {"DEP": "xcomp"}},
    {"LEFT_ID": "v2", "REL_OP": ">", "RIGHT_ID": "v3", "RIGHT_ATTRS": {"DEP": "xcomp"}},
]


def _classify_catenative(doc, ids):
    return "VCP-40"


# --------------------------------------------------------------------------- #
# LLM-tier system prompts.
# --------------------------------------------------------------------------- #
_INTRANS_SYS = (
    "The bracketed clause is intransitive (a subject and a verb, no direct "
    "object). Choose exactly one reading:\n"
    "VCP-05 ergative/inchoative alternation (a patient subject undergoing a "
    "change that the same verb can express transitively: 'the door opened', "
    "'the glass broke', 'the price increased') | "
    "VCP-06 middle voice (a patient subject with a generic/characterising "
    "reading, usually + adverb or won't/wouldn't: 'this shirt washes easily', "
    "'the book sells well', 'the door won't lock') | "
    "VCP-07 reciprocal intransitive (a plural/conjoined subject acting on each "
    "other: 'they met', 'we argued', 'the two lines intersect').\n"
    "If it is none of these - an ordinary agent-subject intransitive ('she "
    "smiled'), a copular clause ('we got so hot'), or the verb actually has "
    "an object here - return NONE."
)
_MEANING_SYS = (
    "The bracketed verb takes a non-finite complement whose FORM (to-infinitive "
    "vs -ing) changes meaning. Return VCP-18 for verbs like stop/remember/"
    "forget/regret/try/go on/mean where 'stop to rest' != 'stop resting'."
)
_PERCEPTION_SYS = (
    "The bracketed verb has an object followed by a non-finite verb. Choose "
    "one:\nVCP-31 object + bare infinitive, completed perception ('I saw her "
    "leave') | VCP-32 object + -ing, ongoing perception or continuation ('I "
    "caught him cheating', 'she kept me waiting') | VCP-33 object + past "
    "participle, resultative/causative ('I had it repaired', 'I want it "
    "finished').\n"
    "If the matrix verb fits none of these patterns (help/let/make/suppose/"
    "say belong to other constructs), return NONE."
)
_ANTIC_SYS = (
    "Return VCP-35 if the bracketed verb takes an anticipatory/dummy 'it' "
    "object standing in for an extraposed clause: 'I find it hard to "
    "concentrate', 'they made it clear that ...'."
)
_COGNATE_SYS = (
    "Return VCP-39 if this is a cognate-object construction: an otherwise "
    "intransitive verb takes an object morphologically/semantically cognate "
    "with it ('live a good life', \"die a hero's death\", 'smile a faint "
    "smile')."
)

_NSUBJ_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "s",
     "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
]
_OBJ_NONFIN_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "cc",
     "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}}},
]
_IT_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "it",
     "RIGHT_ATTRS": {"LOWER": "it", "DEP": {"IN": ["dobj", "nsubj"]}}},
]
_DOBJ_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "o", "RIGHT_ATTRS": {"DEP": "dobj"}},
]


# --------------------------------------------------------------------------- #
# === Part VI addition (Phase 3): VCP-41 V + for + NP + to-infinitive =======

def _vcp41(doc):
    """The for...to complement of a lexical VERB ("We arranged for him to
    travel"): an advcl with mark 'for' + own subject + to-aux, attached to a
    verb. When the same frame hangs off a copula with an adjective
    ("It's important for us to leave") the head has an acomp and the hit
    belongs to ADJ-21 — that is the sibling exclude, and 'for him, to save
    money' purpose adjuncts parse as prep + separate advcl, never matching.
    """
    from .common import for_to_advcl
    for v in doc:
        if v.pos_ != "VERB":
            continue
        if any(c.dep_ == "acomp" for c in v.children):
            continue                       # adjective predicate -> ADJ-21
        cl = for_to_advcl(v)
        if cl is None:
            # second parse shape (passive to-inf): V > prep for > pobj N
            # with a to-infinitive clause on N ("asked for the meeting to
            # be postponed"). Plain "arranged a trip for him" has no
            # to-clause under the pobj.
            for prep in v.children:
                if prep.dep_ != "prep" or prep.lower_ != "for":
                    continue
                for pobj in prep.children:
                    if pobj.dep_ != "pobj":
                        continue
                    inf = next((k for k in pobj.children
                                if k.dep_ in ("relcl", "acl") and
                                any(g.dep_ == "aux" and g.tag_ == "TO"
                                    for g in k.children)), None)
                    if inf is not None:
                        yield (v.i, inf.i)
            continue
        toks = [c.i for c in cl.children] + [cl.i, v.i]
        yield (min(toks), max(toks))


def build(nlp, client=None):
    dets = []

    # --- intransitive frames -------------------------------------------------
    dets.append(RuleRoutingDetector(nlp, ["VCP-01"], _ROOT_VERB,
                                    _classify_intrans, version="vcp01@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["VCP-02"], _ROOT_VERB,
                                    _classify_intrans, version="vcp02@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-03", "VCP-04"], _COP_FORM,
                                    _COP_VERBS, _cop_key, route=_COP_ROUTE.get,
                                    version="vcp03-04-cop@0.1"))

    # --- monotransitive frames ----------------------------------------------
    dets.append(RuleRoutingDetector(nlp, ["VCP-08"], _TRANS_FORM,
                                    _classify_mono, version="vcp08@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["VCP-09"], _TRANS_FORM,
                                    _classify_reflexive, version="vcp09@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "VCP-10", ["each other", "one another"],
                                      attr="LOWER", version="vcp10-recip@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-11"], _CCOMP_FORM, _THINK_CLASS,
                                    _that_key, version="vcp11-that@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-12"], _CCOMP_FORM, _MANDATIVE,
                                    _that_key, version="vcp12-mandative@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["VCP-13"], _CCOMP_FORM,
                                    _classify_whclause, version="vcp13-wh@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["VCP-14"], _XCOMP_FORM,
                                    _classify_whinf, version="vcp14-whinf@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-15", "VCP-16", "VCP-17"],
                                    _ROOT_VERB, _NONFIN_VERBS, _nonfin_key,
                                    route=_nonfin_route, version="vcp15-17@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-19"], _PREP_NP_FORM, _PREP_VERB,
                                    _prep_key, version="vcp19-prepverb@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-20"], _PREP_ING_FORM, _PREP_VERB,
                                    _prep_key, version="vcp20-preping@0.1"))

    # --- ditransitive frames -------------------------------------------------
    dets.append(RuleRoutingDetector(nlp, ["VCP-21"], _DITRANS_FORM,
                                    _classify_ditrans, version="vcp21-double@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-22", "VCP-23"], _SHIFT_FORM,
                                    {(l, p) for (l, p) in
                                     [(v, "to") for v in _DATIVE_TO] +
                                     [(v, "for") for v in _BENEFACTIVE]},
                                    _shift_key, route=_shift_route,
                                    version="vcp22-23-shift@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-24"], _IO_CCOMP_FORM,
                                    _ASSURE_CLASS, _io_that_key,
                                    version="vcp24-io-that@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["VCP-25"], _IO_CCOMP_FORM,
                                    _classify_io_wh, version="vcp25-io-wh@0.1"))

    # --- complex-transitive frames ------------------------------------------
    dets.append(RuleRoutingDetector(nlp, ["VCP-26"], _ROOT_VERB,
                                    _classify_objpred, version="vcp26-adj@0.1"))
    dets.append(CompositeDetector(
        ["VCP-27"],
        [RuleRoutingDetector(nlp, ["VCP-27"], _ROOT_VERB, _classify_objpred,
                             version="vcp27-np@0.1"),
         _SurfaceObjPredNP(nlp)],
        version="vcp27-np@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-28"], _AS_FORM, _AS_VERBS,
                                    _as_key, version="vcp28-as@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-29"], _OBJ_INF_FORM, _OBJ_CONTROL,
                                    _objctrl_key, version="vcp29-objctrl@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-30"], _OBJ_INF_FORM, _CAUSATIVE,
                                    _causative_key, version="vcp30-causative@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-34"], _OBJ_PREP_FORM, _OBJ_PREP,
                                    _obj_prep_key, version="vcp34-objprep@0.1"))

    # --- other verb constructions -------------------------------------------
    dets.append(RuleRoutingDetector(nlp, ["VCP-36"], _EXPL_FORM,
                                    _classify_expl, version="vcp36-there@0.1"))
    dets.append(LexiconRuleDetector(nlp, ["VCP-37"], _LIGHT_FORM, _LIGHT_VERB,
                                    _light_key, version="vcp37-light@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "VCP-38", _IDIOMS, attr="LEMMA",
                                      version="vcp38-idiom@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["VCP-40"], _CATENATIVE_FORM,
                                    _classify_catenative, version="vcp40-caten@0.1"))

    # --- LLM tier (registered, skipped offline) -----------------------------
    def _intrans_gate(doc, token_ids):
        v = doc[token_ids[0]]
        return not any(c.dep_ in ("dobj", "attr", "acomp", "oprd",
                                  "ccomp", "xcomp") for c in v.children)

    dets.append(LLMReadingDetector(nlp, ["VCP-05", "VCP-06", "VCP-07"],
                                   _NSUBJ_FORM, _INTRANS_SYS, client=client,
                                   version="vcp05-07-intrans@0.2",
                                   gate=_intrans_gate))
    dets.append(LLMReadingDetector(nlp, ["VCP-18"], _XCOMP_FORM, _MEANING_SYS,
                                   client=client, version="vcp18-meaning@0.1"))
    dets.append(LLMReadingDetector(nlp, ["VCP-31", "VCP-32", "VCP-33"],
                                   _OBJ_NONFIN_FORM, _PERCEPTION_SYS,
                                   client=client, version="vcp31-33-percep@0.1"))
    dets.append(LLMReadingDetector(nlp, ["VCP-35"], _IT_FORM, _ANTIC_SYS,
                                   client=client, version="vcp35-antic@0.1"))
    dets.append(LLMReadingDetector(nlp, ["VCP-39"], _DOBJ_FORM, _COGNATE_SYS,
                                   client=client, version="vcp39-cognate@0.1"))

    from .common import Scan
    dets.append(Scan("VCP-41", _vcp41, version="vcp41-for-to@0.1"))
    return dets
