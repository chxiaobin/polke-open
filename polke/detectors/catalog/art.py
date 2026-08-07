"""Category ART - articles (a/an, the, zero) across their uses.

The reference-driven rows (unique referents, instruments, decades, names &
places, zero-article idioms) are CLOSED lexical sets -> surface matchers with
word lists kept here. a/an-by-sound (ART-05), exclamative a (ART-04) and
the+superlative/ordinal (ART-10) are small FORM rules. The zero-article rows use
a spaCy Matcher that treats the inventory's Ø marker as an OPTIONAL token so the
same pattern fires on natural text (have lunch) and on the marked examples
(have Ø lunch). The choice rows that hinge on discourse/meaning (first vs
subsequent mention, generic, situational, institutional purpose sense) are the
LLM tier.
"""
from __future__ import annotations
import re
from ..lexical import PhraseLexiconDetector
from ..llm import LLMStandaloneDetector
from ...schema import Annotation, Span
from ..base import Detector

# --------------------------------------------------------------------------- #
# Word lists.
# --------------------------------------------------------------------------- #
RATE_NOUNS = {
    "day", "hour", "week", "month", "year", "minute", "second", "night",
    "head", "person", "dozen", "kilo", "litre", "liter", "gallon", "mile",
    "pound", "time", "capita", "week", "session", "game", "round",
}
FREQ_WORDS = {"twice", "once", "thrice", "per"}

UNIQUE = [
    "sun", "moon", "earth", "world", "sky", "sea", "ground", "universe",
    "government", "internet", "web", "environment", "future", "past",
    "present", "police", "army", "navy", "equator", "horizon", "ozone",
    "atmosphere", "countryside", "weather", "north", "south", "east",
    "west", "globe", "cosmos", "public",
]
INSTRUMENTS = [
    "piano", "guitar", "violin", "flute", "drums", "cello", "saxophone",
    "trumpet", "harp", "clarinet", "oboe", "viola", "bass", "accordion",
    "banjo", "harmonica", "trombone", "tuba", "keyboard",
]
PERIODS = {
    "renaissance", "enlightenment", "reformation", "restoration",
    "depression", "revolution", "antiquity", "sixties", "seventies",
    "eighties", "nineties", "twenties", "thirties", "forties", "fifties",
    "middle", "dark", "cold", "industrial", "victorian", "elizabethan",
}
_DECADE_RE = re.compile(r"^\d{2,4}s$")

ORDINALS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "twentieth", "hundredth", "last", "next",
}
ADJ_UNIQUE = {"only", "same", "very", "sole", "main", "chief", "principal"}

# zero-article idiom lexicons (ART-16/18/19/23)
HAVE_VERBS = ["have", "eat", "cook", "serve", "make", "take", "get", "skip"]
MEALS = ["breakfast", "lunch", "dinner", "supper", "tea", "brunch", "dessert"]
TRANSPORT = ["car", "bus", "train", "plane", "bike", "bicycle", "boat", "ship",
             "foot", "taxi", "underground", "tube", "metro", "coach", "ferry",
             "air", "sea", "rail", "road", "horseback"]
TIMEWORDS = ["night", "noon", "midnight", "dawn", "dusk", "sunset", "sunrise",
             "midday", "daybreak", "nightfall", "lunchtime", "dinnertime"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday"]
STUDY_VERBS = ["study", "learn", "teach", "take", "do", "research", "read"]
SPEAK_VERBS = ["speak", "learn", "teach", "study", "translate", "understand"]
SUBJECTS = ["biology", "chemistry", "physics", "mathematics", "maths", "math",
            "history", "geography", "economics", "philosophy", "psychology",
            "sociology", "medicine", "law", "art", "music", "literature",
            "science", "engineering", "linguistics", "geology", "botany",
            "astronomy", "anatomy", "accounting", "architecture"]
LANGUAGES = ["french", "spanish", "german", "english", "chinese", "japanese",
             "russian", "italian", "arabic", "portuguese", "dutch", "greek",
             "latin", "korean", "hindi", "polish", "turkish", "swedish"]
SPORTS_GAMES = ["tennis", "football", "soccer", "basketball", "chess", "poker",
                "cricket", "golf", "rugby", "baseball", "hockey", "volleyball",
                "cards", "badminton", "squash", "bridge", "darts", "billiards",
                "snooker", "polo", "netball", "handball"]
INSTITUTION_NOUNS = ["hospital", "prison", "jail", "bed", "school", "church",
                     "college", "university", "court", "sea", "work", "home",
                     "town", "class", "camp", "market", "table", "sight",
                     "office", "power", "port"]

# names & places (ART-20/21/22)
THE_COUNTRIES = ["the USA", "the US", "the UK", "the United States",
                 "the United Kingdom", "the Netherlands", "the Philippines",
                 "the Gambia", "the Sudan", "the Congo", "the Bahamas",
                 "the Maldives", "the Emirates", "the UAE", "the USSR",
                 "the Czech Republic", "the Ukraine", "the Vatican"]
ZERO_COUNTRIES = ["Ø France", "Ø Germany", "Ø Spain", "Ø China", "Ø Japan",
                  "Ø England", "Ø Italy", "Ø Brazil", "Ø India", "Ø Russia"]
THE_GEO = ["the Thames", "the Nile", "the Amazon", "the Rhine", "the Danube",
           "the Mississippi", "the Atlantic", "the Pacific", "the Arctic",
           "the Mediterranean", "the Baltic", "the Alps", "the Himalayas",
           "the Andes", "the Rockies", "the Sahara", "the Gobi",
           "the Pacific Ocean", "the Red Sea", "the Dead Sea"]
ZERO_GEO = ["Ø Lake", "Ø Mount", "Ø Mont"]
THE_MEDIA = ["the BBC", "the Times", "the Guardian", "the Sun", "the Observer",
             "the Economist", "the Hilton", "the Ritz", "the Savoy",
             "the NHS", "the UN", "the EU", "the FBI", "the CIA",
             "the Pentagon", "the Kremlin", "the Louvre"]
ZERO_MEDIA = ["Ø Time magazine", "Ø Oxford University", "Ø Cambridge University",
              "Ø Harvard University", "Ø Newsweek", "Ø Buckingham Palace",
              "Ø Heathrow Airport", "Ø Kings College"]


def _ann(text_id, cid, doc, lo, hi, ver, dtype="rule", matched=None):
    span = doc[lo:hi + 1]
    return Annotation(
        text_id=text_id, construct_id=cid,
        span=Span(span.start_char, span.end_char, lo, hi),
        detector_type=dtype, detector_version=ver, confidence=1.0,
        evidence={"tokens": list(range(lo, hi + 1)),
                  "matched": matched or span.text})


class _MatcherDetector(Detector):
    """Surface Matcher (supports OP quantifiers, e.g. an optional Ø marker).
    Longest-match, non-overlapping."""
    detector_type = "lexicon"

    def __init__(self, nlp, cid, patterns, version):
        from spacy.matcher import Matcher
        self.construct_ids = [cid]
        self.version = version
        self._m = Matcher(nlp.vocab)
        self._m.add("p", patterns)

    def match(self, doc, text_id="doc"):
        raw = sorted(self._m(doc), key=lambda x: (-(x[2] - x[1]), x[1]))
        accepted, out = [], []
        for _mid, start, end in raw:
            if any(start >= a and end <= b for a, b in accepted):
                continue
            accepted.append((start, end))
            out.append(_ann(text_id, self.construct_ids[0], doc, start, end - 1,
                             self.version, dtype="lexicon"))
        return out


class _PerRate(Detector):
    """ART-03: a/an in the distributive 'per' sense (twice a day; 60 km an
    hour) - a rate noun preceded by a number/frequency word."""
    construct_ids = ["ART-03"]
    detector_type = "lexicon"
    version = "art03-rate@0.1"

    def match(self, doc, text_id="doc"):
        for t in doc:
            if t.tag_ != "DT" or t.lower_ not in ("a", "an"):
                continue
            head = t.head
            if head.lemma_.lower() not in RATE_NOUNS:
                continue
            sent = t.sent
            if any((w.like_num or w.lower_ in FREQ_WORDS) and w.i < t.i
                   for w in sent):
                lo, hi = t.i, max(t.i, head.i)
                return [_ann(text_id, "ART-03", doc, lo, hi, self.version,
                             dtype="lexicon")]
        return []


class _Exclamative(Detector):
    """ART-04: exclamative a/an (What a day! Such a mess!)."""
    construct_ids = ["ART-04"]
    detector_type = "rule"
    version = "art04-excl@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ in ("what", "such") and t.i + 1 < len(doc) \
                    and doc[t.i + 1].lower_ in ("a", "an"):
                out.append(_ann(text_id, "ART-04", doc, t.i, t.i + 1,
                                self.version))
        return out


class _ArticleBySound(Detector):
    """ART-05: a/an chosen by sound - fires on the pedagogically salient cases
    where the article and the following word's spelling diverge (a university,
    an hour, an MP)."""
    construct_ids = ["ART-05"]
    detector_type = "rule"
    version = "art05-sound@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.tag_ != "DT" or t.lower_ not in ("a", "an"):
                continue
            if t.i + 1 >= len(doc):
                continue
            nxt = doc[t.i + 1]
            fa = next((c for c in nxt.text if c.isalpha()), None)
            if not fa:
                continue
            vowel = fa.lower() in "aeiou"
            if (t.lower_ == "a" and vowel) or (t.lower_ == "an" and not vowel):
                out.append(_ann(text_id, "ART-05", doc, t.i, t.i + 1,
                                self.version))
        return out


class _TheSuperlative(Detector):
    """ART-10: the + superlative / ordinal / only / same / very."""
    construct_ids = ["ART-10"]
    detector_type = "rule"
    version = "art10-superl@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ != "the" or t.tag_ != "DT" or t.i + 1 >= len(doc):
                continue
            nxt = doc[t.i + 1]
            if nxt.tag_ == "JJS" or nxt.lower_ in ORDINALS \
                    or nxt.lower_ in ADJ_UNIQUE:
                out.append(_ann(text_id, "ART-10", doc, t.i, nxt.i,
                                self.version))
        return out


class _TheDecade(Detector):
    """ART-14: the + decade (the 1990s) or named period (the Renaissance)."""
    construct_ids = ["ART-14"]
    detector_type = "lexicon"
    version = "art14-decade@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ != "the" or t.tag_ != "DT" or t.i + 1 >= len(doc):
                continue
            nxt = doc[t.i + 1]
            if _DECADE_RE.match(nxt.text.lower()) or nxt.lower_ in PERIODS:
                out.append(_ann(text_id, "ART-14", doc, t.i, nxt.i,
                                self.version, dtype="lexicon"))
        return out


def _zero(head, lemmas=None, orth_in=None):
    """A pattern list matching an optional Ø marker then the target token."""
    if orth_in is not None:
        tgt = {"LOWER": {"IN": orth_in}}
    else:
        tgt = {"LEMMA": {"IN": lemmas}}
    return [head, {"ORTH": "Ø", "OP": "?"}, tgt]


def build(nlp, client=None):
    dets = []

    # --- rule / form tiers ------------------------------------------------- #
    dets.append(_PerRate())          # ART-03
    dets.append(_Exclamative())      # ART-04
    dets.append(_ArticleBySound())   # ART-05
    dets.append(_TheSuperlative())   # ART-10
    dets.append(_TheDecade())        # ART-14

    # --- definite-the lexical sets ---------------------------------------- #
    dets.append(PhraseLexiconDetector(nlp, "ART-09", [f"the {w}" for w in UNIQUE],
                                      version="art09-unique@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ART-12",
                                      [f"the {w}" for w in INSTRUMENTS],
                                      version="art12-instrument@0.1"))

    # --- zero-article idioms (Ø optional) --------------------------------- #
    dets.append(_MatcherDetector(nlp, "ART-16", [
        _zero({"LEMMA": {"IN": HAVE_VERBS}}, lemmas=MEALS),
        _zero({"LOWER": "by"}, lemmas=TRANSPORT),
        _zero({"LOWER": "at"}, lemmas=TIMEWORDS),
        _zero({"LOWER": "on"}, orth_in=WEEKDAYS),
    ], "art16-meals-transport@0.1"))
    dets.append(_MatcherDetector(nlp, "ART-18", [
        _zero({"LEMMA": {"IN": STUDY_VERBS}}, lemmas=SUBJECTS),
        _zero({"LEMMA": {"IN": SPEAK_VERBS}}, orth_in=LANGUAGES),
        _zero({"LEMMA": "play"}, orth_in=SPORTS_GAMES),
    ], "art18-subjects@0.1"))
    dets.append(_MatcherDetector(nlp, "ART-19", [
        [{"LEMMA": {"IN": ["kind", "sort", "type"]}}, {"LOWER": "of"},
         {"ORTH": "Ø", "OP": "?"}, {"POS": {"IN": ["NOUN", "PROPN"]}}],
    ], "art19-kindof@0.1"))
    dets.append(_MatcherDetector(nlp, "ART-23", [
        _zero({"LOWER": {"IN": ["in", "at", "to", "into", "from"]}},
              lemmas=INSTITUTION_NOUNS),
    ], "art23-fixed@0.1"))

    # --- names & places lexical sets -------------------------------------- #
    dets.append(PhraseLexiconDetector(nlp, "ART-20", THE_COUNTRIES + ZERO_COUNTRIES,
                                      version="art20-countries@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ART-21", THE_GEO + ZERO_GEO,
                                      version="art21-geography@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ART-22", THE_MEDIA + ZERO_MEDIA,
                                      version="art22-institutions@0.1"))

    # --- LLM tier (registered structurally; skipped offline) --------------- #
    _llm = {
        "ART-01": "indefinite a/an introducing a first-mention / non-specific "
                  "singular count noun (I saw a dog); NOT the classifying use "
                  "after be (she's a teacher - a different construct)",
        "ART-02": "indefinite a/an classifying (occupation or category: she's a "
                  "teacher)",
        "ART-06": "anaphoric the (second mention of something introduced "
                  "earlier in the text); NOT situational shared-knowledge the "
                  "(shut the door) and NOT the identified by a following "
                  "modifier - those are different constructs",
        "ART-07": "cataphoric the (definite because post-modified: the man who "
                  "called)",
        "ART-08": "situational the (shared-knowledge referent: shut the door); "
                  "NOT anaphoric the (second mention) and NOT the identified "
                  "by a post-modifier (the book that you lent me) - those are "
                  "different constructs",
        "ART-11": "generic the + singular naming a whole class (the tiger is "
                  "endangered)",
        "ART-13": "the + adjective/nationality denoting a group (the rich, the "
                  "unemployed, the Chinese)",
        "ART-15": "zero article with a generic plural or uncountable noun (dogs "
                  "bark; I like music); NOT fixed institutional phrases (have "
                  "lunch, go to school, by bus) - those are different "
                  "constructs",
        "ART-17": "zero article with an institution in its purpose sense "
                  "(in hospital / at school as a patient / pupil)",
        "ART-24": "the contrasting uses of 'most' (most people / the most "
                  "expensive / most of the people)",
    }
    for cid, desc in _llm.items():
        dets.append(LLMStandaloneDetector(
            cid, f"Decide whether the sentence contains: {desc}. Return JSON "
            f'{{"construct_id":"{cid}"|"NONE","confidence":0..1,'
            f'"rationale":"..."}}.', client=client,
            version=f"{cid.lower()}-llm@0.1"))
    return dets
