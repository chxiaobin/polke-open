"""Category NOU - nouns (countability, number, possession, modification).

Number and countability sub-types are largely CLOSED lexical classes
(pluralia tantum, invariant/foreign plurals, uncount-looking -ics nouns) ->
surface PhraseLexiconDetector on word lists kept here as module constants.
Regular plurals (NOU-08) and the possessive/genitive family (NOU-13..16,19,20)
are FORM rules over the spaCy possessive clitic (tag POS, dep case, head =
possessor). The meaning-driven rows (count/uncount shift, coercion, collective
agreement, compound vs modifier readings, double/group genitive) are the LLM
tier and are registered but skipped offline.
"""
from __future__ import annotations
from ..lexical import PhraseLexiconDetector
from ..llm import LLMStandaloneDetector
from ...schema import Annotation, Span
from ..base import Detector

# --------------------------------------------------------------------------- #
# Word lists (module constants; recommended for promotion to lexicons.json).
# --------------------------------------------------------------------------- #
UNCOUNT = {
    "advice", "information", "news", "furniture", "luggage", "money",
    "research", "knowledge", "equipment", "baggage", "traffic", "weather",
    "homework", "housework", "work", "progress", "evidence", "accommodation",
    "behaviour", "behavior", "courage", "damage", "education", "employment",
    "fun", "happiness", "health", "help", "honesty", "importance",
    "intelligence", "justice", "laughter", "leisure", "love", "luck",
    "music", "nature", "patience", "peace", "pollution", "poverty",
    "pronunciation", "publicity", "rubbish", "safety", "scenery", "shopping",
    "silence", "spelling", "violence", "wealth", "wisdom", "water", "air",
    "bread", "butter", "cheese", "rice", "sugar", "salt", "milk", "flour",
    "electricity", "energy", "software", "hardware", "data",
}
# Nouns with both a count and an uncount reading (meaning shift): NOU-03.
DUAL = {
    "coffee", "paper", "glass", "experience", "chocolate", "beer", "wine",
    "tea", "cake", "chicken", "iron", "light", "noise", "room", "hair",
    "business", "time", "space", "fire", "language", "cloth", "life",
    "difficulty", "sound",
}
# Pluralia tantum (only plural): NOU-04.
PLURALIA = {
    "scissors", "trousers", "jeans", "glasses", "clothes", "goods", "pants",
    "shorts", "pyjamas", "pajamas", "tights", "leggings", "spectacles",
    "binoculars", "scales", "premises", "belongings", "outskirts",
    "surroundings", "earnings", "savings", "thanks", "congratulations",
    "riches", "remains", "contents", "customs", "stairs", "arms", "ashes",
    "manners", "odds", "particulars", "proceeds", "refreshments", "valuables",
    "whereabouts", "headphones", "pliers", "tongs", "shears", "overalls",
}
# Uncount nouns that look plural / take singular agreement: NOU-05.
ICS_UNCOUNT = {
    "news", "mathematics", "maths", "math", "physics", "economics",
    "politics", "statistics", "linguistics", "ethics", "gymnastics",
    "athletics", "electronics", "genetics", "mechanics", "acoustics",
    "aerodynamics", "phonetics", "logistics", "measles", "mumps", "rabies",
    "diabetes", "billiards", "darts", "dominoes", "checkers", "draughts",
}
# Irregular plurals (both singular and plural citation forms): NOU-09.
IRREGULAR = {
    "man", "men", "woman", "women", "child", "children", "foot", "feet",
    "tooth", "teeth", "mouse", "mice", "goose", "geese", "person", "people",
    "ox", "oxen", "louse", "lice", "die", "dice", "penny", "pence",
    "brother", "brethren",
}
# Zero / invariant plurals: NOU-10.
INVARIANT = {
    "sheep", "fish", "deer", "aircraft", "series", "species", "means",
    "salmon", "trout", "cod", "moose", "offspring", "spacecraft",
    "hovercraft", "swine", "bison", "grouse", "crossroads", "headquarters",
    "barracks", "gallows", "corps", "innings",
}
# Foreign / Latinate plurals (both forms): NOU-11.
FOREIGN = {
    "criterion", "criteria", "phenomenon", "phenomena", "datum", "data",
    "analysis", "analyses", "crisis", "crises", "index", "indices",
    "thesis", "theses", "basis", "bases", "nucleus", "nuclei", "cactus",
    "cacti", "fungus", "fungi", "syllabus", "syllabi", "appendix",
    "appendices", "matrix", "matrices", "formula", "formulae", "curriculum",
    "curricula", "medium", "media", "bacterium", "bacteria", "stimulus",
    "stimuli", "radius", "radii", "alga", "algae", "larva", "larvae",
    "vertebra", "vertebrae", "hypothesis", "hypotheses", "parenthesis",
    "parentheses", "axis", "axes", "oasis", "oases", "memorandum",
    "memoranda", "spectrum", "spectra", "corpus", "corpora",
}
# Time / measure genitive heads: NOU-19.
TIME_MEASURE = {
    "day", "week", "month", "year", "hour", "minute", "second", "moment",
    "night", "morning", "afternoon", "evening", "decade", "century",
    "pound", "dollar", "euro", "cent", "mile", "kilometre", "kilometer",
    "metre", "meter", "yard", "foot", "stone", "ton", "tonne", "kilo",
}
# Partitive nouns for uncount: NOU-06 ("<part> of ...").
PARTITIVES = [
    "piece", "slice", "item", "bit", "loaf", "sheet", "drop", "grain",
    "lump", "bar", "blade", "pane", "clap", "flash", "ball", "cube",
    "pinch", "spot", "strip", "scrap", "speck", "rasher", "sliver",
    "morsel", "dash", "gust", "stroke", "spell", "pane",
]
# Nationality / group nouns used as "the + ADJ" plurals: NOU-24.
NATIONALITIES = [
    "french", "dutch", "british", "english", "irish", "welsh", "scottish",
    "spanish", "swedish", "danish", "finnish", "polish", "turkish",
    "chinese", "japanese", "vietnamese", "portuguese", "swiss", "lebanese",
    "cornish", "flemish", "maltese",
]
COMPOUND_PLURALS = [
    "passers - by", "passer - by", "mothers - in - law", "mother - in - law",
    "brothers - in - law", "brother - in - law", "sisters - in - law",
    "sister - in - law", "sons - in - law", "son - in - law",
    "daughters - in - law", "daughter - in - law", "fathers - in - law",
    "father - in - law", "runners - up", "runner - up", "runners - up",
    "grown - ups", "grown - up", "commanders - in - chief",
    "editors - in - chief", "men - of - war", "attorneys - general",
    "courts - martial", "lookers - on", "passers - through",
    "poets - laureate", "notaries - public", "heirs - apparent",
    "lieutenant - colonels", "men - at - arms", "ladies - in - waiting",
]

_EXCL_COUNT = UNCOUNT | DUAL | PLURALIA | ICS_UNCOUNT


def _ann(text_id, cid, doc, lo, hi, ver, dtype="rule", matched=None):
    span = doc[lo:hi + 1]
    return Annotation(
        text_id=text_id, construct_id=cid,
        span=Span(span.start_char, span.end_char, lo, hi),
        detector_type=dtype, detector_version=ver, confidence=1.0,
        evidence={"tokens": list(range(lo, hi + 1)),
                  "matched": matched or span.text})


def _regular_plural(lem, surf):
    """Is *surf* a regularly-inflected plural of *lem*?"""
    if surf == lem:
        return False
    if lem + "s" == surf:
        return True
    if lem + "es" == surf:
        return True
    if lem.endswith("y") and lem[:-1] + "ies" == surf:
        return True
    if lem.endswith("f") and lem[:-1] + "ves" == surf:
        return True
    if lem.endswith("fe") and lem[:-2] + "ves" == surf:
        return True
    if lem.endswith("o") and (lem + "es" == surf or lem + "s" == surf):
        return True
    return False


class _CountableNoun(Detector):
    """NOU-01: a noun with an overt countable signal (a/an/one, a number, or a
    plural form) that is NOT uncount / dual / pluralia / -ics-uncount."""
    construct_ids = ["NOU-01"]
    detector_type = "rule"
    version = "nou01-countable@0.1"

    def match(self, doc, text_id="doc"):
        for t in doc:
            if t.tag_ not in ("NN", "NNS"):
                continue
            if t.text.lower() in _EXCL_COUNT or t.lemma_.lower() in _EXCL_COUNT:
                continue
            plural = t.tag_ == "NNS"
            has_art = any(c.dep_ == "det" and c.lower_ in ("a", "an", "one")
                          for c in t.children)
            has_num = any(c.dep_ == "nummod" for c in t.children)
            if plural or has_art or has_num:
                return [_ann(text_id, "NOU-01", doc, t.i, t.i, self.version)]
        return []


class _RegularPlural(Detector):
    """NOU-08: regularly-inflected plural common noun, excluding invariant
    nouns and compound plurals (hyphenated)."""
    construct_ids = ["NOU-08"]
    detector_type = "rule"
    version = "nou08-regplural@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.tag_ != "NNS":
                continue
            surf = t.text.lower()
            if surf in INVARIANT or surf in FOREIGN or surf in IRREGULAR:
                continue
            # part of a hyphenated compound -> NOU-12, not a simple plural
            if (t.i + 1 < len(doc) and doc[t.i + 1].tag_ == "HYPH") or \
               (t.i > 0 and doc[t.i - 1].tag_ == "HYPH"):
                continue
            if _regular_plural(t.lemma_.lower(), surf):
                out.append(_ann(text_id, "NOU-08", doc, t.i, t.i, self.version))
        return out


class _PossessiveFamily(Detector):
    """Serves the possessive-clitic genitives (NOU-13/14/15/19/20).

    One scan over possessive clitics (tag POS, head = possessor); each id gets
    its own detector instance sharing this class but pinned to a single cid, so
    the contract sees one detector per id. ``cid`` selects the branch.
    """
    detector_type = "rule"

    def __init__(self, cid, version):
        self.construct_ids = [cid]
        self.version = version
        self._cid = cid

    def _classify(self, doc, clitic, poss):
        s = clitic.text  # "'s" or "'"
        tag = poss.tag_
        lem = poss.lemma_.lower()
        dep = poss.dep_
        time_measure = lem in TIME_MEASURE
        # NOU-19 time/measure genitive (owns the time/measure possessors)
        if time_measure and dep == "poss":
            return "NOU-19"
        # NOU-20 local/elliptical genitive: no following head noun (possessor is
        # itself an argument), and not the "of X's" double genitive.
        if dep in ("pobj", "dobj", "nsubj", "nsubjpass", "dative", "attr",
                   "conj", "ROOT"):
            if dep == "pobj" and poss.head.lower_ == "of":
                return None  # double genitive -> NOU-17 (LLM)
            return "NOU-20"
        # remaining branches require a possessive relation to a head noun
        if dep != "poss":
            return None
        # exclude group genitive "the King of Spain's" (possessor after "of")
        if poss.i >= 1 and doc[poss.i - 1].lower_ == "of":
            return None
        if tag == "NNS":
            # regular-plural possessor + bare apostrophe -> NOU-14; irregular
            # plural + 's -> NOU-15.
            return "NOU-14" if s == "'" else "NOU-15"
        if tag in ("NN", "NNP") and s == "'s":
            return "NOU-13"
        return None

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.tag_ != "POS":
                continue
            poss = t.head
            cid = self._classify(doc, t, poss)
            if cid != self._cid:
                continue
            lo, hi = min(poss.i, t.i), max(poss.i, t.i)
            out.append(_ann(text_id, cid, doc, lo, hi, self.version,
                            dtype="lexicon" if cid in ("NOU-19",) else "rule"))
        return out


class _OfGenitive(Detector):
    """NOU-16: of-genitive (noun + of + noun-NP), excluding the double genitive
    (of + possessive pronoun / of + N's)."""
    construct_ids = ["NOU-16"]
    detector_type = "rule"
    version = "nou16-ofgen@0.1"
    _POSS_PRON = {"mine", "yours", "hers", "ours", "theirs", "his"}
    # quantifier/measure heads make a PARTITIVE of-phrase, not a genitive
    _QUANT_HEADS = {"bit", "lot", "lots", "kind", "sort", "type", "load",
                    "loads", "couple", "plenty", "bunch", "deal", "ton",
                    "tons", "heap", "heaps", "pile", "amount", "number",
                    "rest", "half", "majority", "none", "stack", "mass",
                    "pair", "series", "range", "variety", "cup", "glass",
                    "bottle", "piece", "slice"}

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ != "of" or t.dep_ != "prep":
                continue
            head = t.head
            if head.pos_ not in ("NOUN", "PROPN"):
                continue
            if head.lemma_.lower() in self._QUANT_HEADS:
                continue
            pobjs = [c for c in t.children if c.dep_ == "pobj"]
            if not pobjs:
                continue
            obj = pobjs[0]
            if obj.lower_ in self._POSS_PRON:
                continue  # double genitive
            if any(c.tag_ == "POS" for c in obj.children):
                continue  # "of John's" double genitive
            if obj.pos_ not in ("NOUN", "PROPN", "PRON"):
                continue
            out.append(_ann(text_id, "NOU-16", doc, head.i, obj.i, self.version))
        return out


# === Part VI additions (Phase 3): NOU-25 apposition, NOU-26 plural premod ==

def _nou25(doc):
    """Appositive NP: the parser's appos dep over nouny heads ("my brother,
    a doctor", "the novelist John le Carre", "we teachers"). Excludes: lists
    (conj, not appos), vocatives ("John, are you coming?" parses npadvmod),
    dysfluent identical repeats ("I I" -> appos of the same form), and the
    vernacular 'them things' NP (VER-04's demonstrative them)."""
    for t in doc:
        if t.dep_ != "appos":
            continue
        h = t.head
        if t.pos_ not in ("NOUN", "PROPN", "PRON") \
                or h.pos_ not in ("NOUN", "PROPN", "PRON"):
            continue
        if t.lower_ == h.lower_:
            continue                       # "I I I" repeat, not apposition
        if h.lower_ == "them":
            continue                       # them things -> VER-04
        lo, hi = min(h.i, t.i), max(h.i, t.i)
        yield (lo, hi)


def _nou26(doc):
    """Plural noun as premodifier: NNS compound before a head noun
    ("sports car", "drugs policy", "arms race"); singular premodifiers
    ("shoe shop", "three-hour meeting") are NOU-22's pattern."""
    for t in doc:
        if t.tag_ == "NNS" and t.dep_ == "compound" and t.i < t.head.i \
                and t.head.pos_ in ("NOUN", "PROPN"):
            yield (t.i, t.head.i)


def build(nlp, client=None):
    dets = []

    # --- rule tiers -------------------------------------------------------- #
    dets.append(_CountableNoun())               # NOU-01
    dets.append(_RegularPlural())               # NOU-08
    dets.append(_OfGenitive())                  # NOU-16
    for cid, ver in (("NOU-13", "nou13-poss-sg@0.1"),
                     ("NOU-14", "nou14-poss-pl@0.1"),
                     ("NOU-15", "nou15-poss-irr@0.1"),
                     ("NOU-19", "nou19-time-gen@0.1"),
                     ("NOU-20", "nou20-local-gen@0.1")):
        dets.append(_PossessiveFamily(cid, ver))

    # --- surface lexicon tiers -------------------------------------------- #
    dets.append(PhraseLexiconDetector(nlp, "NOU-04", sorted(PLURALIA),
                                      version="nou04-pluralia@0.1"))
    # "news" also appears in the generic uncount list (NOU-02's illustrative
    # example, and a NOU-05 negative), so trigger NOU-05 on the -ics nouns only.
    dets.append(PhraseLexiconDetector(nlp, "NOU-05", sorted(ICS_UNCOUNT - {"news"}),
                                      version="nou05-ics@0.1"))
    dets.append(PhraseLexiconDetector(
        nlp, "NOU-06", [f"{p} of" for p in PARTITIVES],
        version="nou06-partitive@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "NOU-09", sorted(IRREGULAR),
                                      version="nou09-irregular@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "NOU-10", sorted(INVARIANT),
                                      version="nou10-invariant@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "NOU-11", sorted(FOREIGN),
                                      version="nou11-foreign@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "NOU-12", COMPOUND_PLURALS,
                                      version="nou12-compound@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "NOU-24",
                                      [f"the {n}" for n in NATIONALITIES],
                                      version="nou24-nationality@0.1"))

    # --- LLM tier (registered structurally; skipped offline) --------------- #
    _llm = {
        "NOU-02": "an uncountable (mass) noun used without a plural or a/an",
        "NOU-03": "a noun used in its count vs uncount reading (meaning shift, "
                  "e.g. a coffee / coffee, a paper / paper)",
        "NOU-07": "an uncountable noun coerced to a count reading (two coffees, "
                  "three beers)",
        "NOU-17": "a double genitive (a friend of mine / of John's)",
        "NOU-18": "a group genitive - the 's attaches to the END of a multi-"
                  "word phrase rather than to its head noun (the King of "
                  "Spain's visit; someone else's idea; the man next door's "
                  "car)",
        "NOU-21": "a noun+noun compound (bus stop, coffee table)",
        "NOU-22": "a singular noun used as a modifier of another noun, "
                  "including measure nouns that stay singular (a three-hour "
                  "meeting, a ten-pound note, a shoe shop)",
        "NOU-23": "a collective noun with singular-vs-plural (notional) "
                  "agreement (the team is / are)",
    }
    for cid, desc in _llm.items():
        dets.append(LLMStandaloneDetector(
            cid, f"Decide whether the sentence contains: {desc}. Return JSON "
            f'{{"construct_id":"{cid}"|"NONE","confidence":0..1,'
            f'"rationale":"..."}}.', client=client,
            version=f"{cid.lower()}-llm@0.1"))
    from .common import Scan
    dets.append(Scan("NOU-25", _nou25, version="nou25-appos@0.1"))
    dets.append(Scan("NOU-26", _nou26, version="nou26-plural-premod@0.1"))
    return dets
