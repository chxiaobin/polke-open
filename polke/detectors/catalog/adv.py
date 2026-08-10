"""Category ADV - adverbs & adverbials.

Two shapes dominate:
- Closed word-class *types* (place, definite/indefinite time & frequency,
  viewpoint, stance, linking, and the irregular/flat adverbs) -> surface
  ``PhraseLexiconDetector`` tables. Many of the inventory's positive cases are
  bare comma lists ("here, there, outside, ...") with no clean parse, so surface
  matching is exactly right and the sibling negatives are disjoint word sets.
- *Position / degree* rules (front / mid / end position, comparatives, too/
  enough, as...as, intensifiers) -> a small token-predicate rule engine
  (``_TokenRule``) that inspects dep/head/tag.

ADV-02 (flat citation forms) vs ADV-03 (same flat form used adverbially by a
verb) share one lexicon and are split purely by whether the head is a VERB.

LLM tier (skipped offline): ADV-10 degree, ADV-11 focusing, ADV-18 order of
manner-place-time, ADV-19 scope of only/even, ADV-21 ever in Q/neg,
ADV-25 quite/rather downtoners, ADV-29 phase adverbs (still/already/yet).
ADV-04 has no positive contract case (citation of paired forms) - implemented on
the distinct -ly members; flagged in NOTES.
"""
from __future__ import annotations
from typing import Callable
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector
from ..base import Detector
from ...schema import Annotation, Span


# --------------------------------------------------------------------------- #
# Word lists (kept local per the brief; candidates for lexicons.json in NOTES). #
# --------------------------------------------------------------------------- #

# Flat adverbs: same form as the adjective (ADV-02 citation / ADV-03 in use).
_FLAT = {"fast", "hard", "late", "early", "daily", "well", "high", "low",
         "deep", "near", "straight", "wrong", "right", "fair", "fine", "loud",
         "quick", "slow", "long", "far", "clean", "close", "tight", "flat",
         "direct", "sharp"}

# -ly words that are NOT regular manner de-adjectival adverbs (exclude from
# ADV-01, which is "adjective + -ly"): time/frequency/degree/adjective -ly forms.
_LY_NOT_MANNER = {"early", "daily", "weekly", "monthly", "yearly", "hourly",
                  "nightly", "quarterly", "only", "likely", "lonely", "friendly",
                  "lively", "deadly", "lovely", "ugly", "silly", "holy", "jolly",
                  "kindly", "cowardly", "elderly", "orderly", "leisurely"}

# ADV-04: the -ly member of a pair whose meaning differs from the flat form.
_DISTINCT_LY = {"hardly", "lately", "nearly", "highly", "deeply", "shortly",
                "presently", "scarcely", "barely", "directly", "closely",
                "coldly", "widely", "freely", "rightly", "wrongly", "roughly",
                "shortly", "flatly", "sharply", "justly"}

# ADV-06 place / direction.
_PLACE = {"here", "there", "everywhere", "somewhere", "anywhere", "nowhere",
          "elsewhere", "outside", "inside", "indoors", "outdoors", "upstairs",
          "downstairs", "abroad", "overseas", "away", "back", "home", "nearby",
          "ahead", "aboard", "ashore", "underground", "downtown", "uptown",
          "uphill", "downhill", "northward", "southward", "sideways",
          "backward", "backwards", "forward", "forwards", "onward", "onwards"}

# ADV-07 definite time.
_DEF_TIME = {"yesterday", "today", "tomorrow", "now", "then", "tonight",
             "soon", "recently", "currently", "nowadays", "afterwards",
             "afterward", "earlier", "later", "shortly", "immediately",
             "eventually", "presently", "meanwhile", "beforehand", "lately"}

# ADV-08 indefinite frequency.
_INDEF_FREQ = {"always", "usually", "often", "sometimes", "rarely", "never",
               "seldom", "frequently", "occasionally", "normally", "generally",
               "regularly", "constantly", "continually", "repeatedly", "ever",
               "periodically", "intermittently", "sporadically", "typically",
               "ordinarily", "commonly", "habitually", "invariably"}

# ADV-09 definite frequency (single tokens + multiword phrases).
_DEF_FREQ_WORDS = {"daily", "weekly", "monthly", "yearly", "annually", "hourly",
                   "nightly", "quarterly", "fortnightly", "biweekly"}
_DEF_FREQ_PHRASES = ["twice a week", "once a week", "every day", "every week",
                     "every year", "every month", "every hour", "every morning",
                     "every night", "every evening", "twice a day", "once a day",
                     "three times a day", "twice a year", "once a month",
                     "once a year", "several times a day", "two times a week"]

# ADV-12 viewpoint / domain (a -ly adjunct scoping the whole proposition).
_DOMAIN = {"financially", "technically", "politically", "economically",
           "scientifically", "legally", "biologically", "chemically",
           "physically", "morally", "socially", "environmentally", "culturally",
           "historically", "geographically", "statistically", "theoretically",
           "logically", "grammatically", "mathematically", "medically",
           "commercially", "industrially", "ideologically", "psychologically",
           "ethically", "linguistically", "architecturally", "structurally",
           "financially", "academically", "professionally"}

# ADV-13 comment / stance disjuncts.
_STANCE = {"frankly", "obviously", "fortunately", "apparently", "honestly",
           "clearly", "surely", "presumably", "arguably", "admittedly",
           "luckily", "unfortunately", "evidently", "certainly", "undoubtedly",
           "hopefully", "sadly", "curiously", "interestingly", "surprisingly",
           "understandably", "regrettably", "thankfully", "seriously",
           "personally", "supposedly", "allegedly", "reportedly", "ideally",
           "ironically", "predictably", "inevitably", "strangely", "oddly",
           "amazingly", "remarkably", "notably", "importantly", "significantly",
           "unbelievably", "mercifully", "curiously", "annoyingly"}

# ADV-14 linking adverbs (conjuncts).
_LINK_WORDS = {"however", "therefore", "moreover", "furthermore", "consequently",
               "nevertheless", "nonetheless", "thus", "hence", "accordingly",
               "meanwhile", "otherwise", "besides", "instead", "likewise",
               "similarly", "additionally", "anyway", "namely", "indeed",
               "conversely", "subsequently", "alternatively", "firstly",
               "secondly", "finally", "lastly", "overall"}
_LINK_PHRASES = ["for example", "for instance", "in addition",
                 "on the other hand", "as a result", "in contrast",
                 "in conclusion", "on the contrary", "in fact", "in short"]

# Degree modifiers of a comparative (ADV-22).
_MODCOMP = {"much", "far", "lot", "bit", "slightly", "even", "no", "any",
            "rather", "still", "way", "loads", "considerably", "significantly",
            "marginally", "somewhat", "substantially", "infinitely"}

# Intensifiers of a verb (ADV-23).
_INTENS_VERB = {"totally", "completely", "strongly", "entirely", "absolutely",
                "fully", "utterly", "thoroughly", "greatly", "deeply", "badly",
                "heartily", "firmly", "readily", "sincerely", "wholeheartedly",
                "highly", "widely", "severely", "seriously", "vehemently"}

# very / really / so + adjective/adverb (ADV-24).
_VRS = {"very", "really", "so", "extremely", "terribly", "awfully",
        "incredibly", "remarkably", "super", "dead", "mighty", "hugely",
        "immensely", "dreadfully", "jolly", "real", "insanely", "ridiculously"}

# Flat adverbs that also form a synthetic comparative/superlative (ADV-27).
_FLAT_COMP = {"fast", "hard", "late", "early", "near", "high", "low", "deep",
              "long", "soon", "quick", "slow", "loud", "straight", "close",
              "tight", "clear", "wide"}

_ADVP = {"ADJ", "ADV"}


# --------------------------------------------------------------------------- #
# Tiny token-predicate rule engine.                                            #
# --------------------------------------------------------------------------- #
class _TokenRule(Detector):
    """Emit a one-token annotation for every token satisfying ``predicate``."""

    def __init__(self, cid: str, predicate: Callable, version: str,
                 detector_type: str = "rule"):
        self.construct_ids = [cid]
        self._pred = predicate
        self.version = version
        self.detector_type = detector_type

    def match(self, doc, text_id: str = "doc"):
        out = []
        for t in doc:
            if self._pred(doc, t):
                out.append(Annotation(
                    text_id=text_id, construct_id=self.construct_ids[0],
                    span=Span(t.idx, t.idx + len(t.text), t.i, t.i),
                    detector_type=self.detector_type,
                    detector_version=self.version, confidence=1.0,
                    evidence={"tokens": [t.i], "matched": t.text}))
        return out


def _is_adverb(t):
    return t.pos_ == "ADV" or t.tag_ in ("RB", "RBR", "RBS")


def _subject_before(head, i):
    return any(c.dep_ in ("nsubj", "nsubjpass", "expl") and c.i < i
               for c in head.children)


def _n_adverbials(head):
    return sum(1 for c in head.children
               if c.dep_ in ("advmod", "npadvmod", "prep"))


_FOCUS = {"only", "even", "just", "merely", "simply", "also", "too", "either",
          "alone", "particularly", "especially", "mainly", "purely"}


def _is_imperative(v):
    return v.tag_ == "VB" and v.dep_ == "ROOT" and \
        not any(c.dep_ in ("nsubj", "nsubjpass") for c in v.children)


# ---- individual predicates ------------------------------------------------- #
def _p_regular_ly(doc, t):
    # ADV-01: adjective + -ly manner adverb.
    return (t.tag_ in ("RB", "RBR", "RBS") and t.lower_.endswith("ly")
            and t.lower_ not in _LY_NOT_MANNER and len(t.lower_) > 3)


def _p_manner(doc, t):
    # ADV-05: manner adverb modifying a verb ("how").
    if t.dep_ != "advmod" or t.head.pos_ not in ("VERB", "AUX"):
        return False
    if t.lower_ in (_PLACE | _DEF_TIME | _INDEF_FREQ | _DEF_FREQ_WORDS
                    | _STANCE | _LINK_WORDS | _FOCUS | {"very", "so", "too",
                    "quite", "rather", "not", "n't", "never", "always"}):
        return False
    return t.lower_.endswith("ly") or t.lower_ in _FLAT


def _p_front(doc, t):
    # ADV-15: sentence-initial adverbial modifying the clause.
    if t.i != 0 or t.pos_ != "ADV" or t.head.pos_ not in ("VERB", "AUX"):
        return False
    if t.lower_ in _INDEF_FREQ and _is_imperative(t.head):
        return False   # leave frequency-imperative to ADV-20
    return True


def _p_mid(doc, t):
    # ADV-16: adverb between subject and its verb (after operator / before verb).
    if t.pos_ != "ADV" or t.dep_ != "advmod" or t.lower_ in _FOCUS:
        return False
    if t.head.pos_ not in ("VERB", "AUX") or t.i >= t.head.i:
        return False
    return _subject_before(t.head, t.i)


def _p_end(doc, t):
    # ADV-17: single adverb after its verb at clause end.
    if t.pos_ != "ADV" or t.dep_ != "advmod" or t.lower_ in _FOCUS:
        return False
    if t.head.pos_ not in ("VERB", "AUX") or t.i <= t.head.i:
        return False
    return _n_adverbials(t.head) == 1


def _p_freq_imperative(doc, t):
    # ADV-20: frequency adverb with an imperative.
    return (t.lower_ in _INDEF_FREQ and t.dep_ == "advmod"
            and _is_imperative(t.head))


def _p_mod_comparative(doc, t):
    # ADV-22: much/far/a bit/slightly/even + comparative.
    if t.dep_ != "advmod" or t.lemma_.lower() not in _MODCOMP:
        return False
    h = t.head
    return h.tag_ in ("JJR", "RBR") or h.lower_ in ("more", "less")


def _p_intens_verb(doc, t):
    # ADV-23: completely/totally/strongly + verb.
    return (t.dep_ == "advmod" and t.head.pos_ == "VERB"
            and t.lemma_.lower() in _INTENS_VERB)


def _p_vrs(doc, t):
    # ADV-24: very/really/so + adjective/adverb.
    return (t.dep_ == "advmod" and t.lemma_.lower() in _VRS
            and t.head.pos_ in _ADVP)


def _p_too_enough(doc, t):
    # ADV-26: too + adj/adv  OR  adj/adv + enough.
    if t.lower_ == "too" and t.dep_ == "advmod" and t.head.pos_ in _ADVP:
        return True
    if t.lower_ == "enough" and t.head.pos_ in _ADVP:
        return True
    return False


def _p_comp_superl_adv(doc, t):
    # ADV-27: comparative / superlative adverb.
    if t.pos_ == "ADV" and t.tag_ in ("RBR", "RBS"):
        return True
    if t.lower_ in ("more", "most", "less", "least") and t.dep_ == "advmod" \
            and t.head.pos_ == "ADV":
        return True
    if t.tag_ in ("JJR", "JJS") and t.lemma_.lower() in _FLAT_COMP:
        return True
    return False


class _AsAsRule(Detector):
    """ADV-28: as + adverb + as (equative comparison of an adverb)."""
    detector_type = "rule"

    def __init__(self):
        self.construct_ids = ["ADV-28"]
        self.version = "adv28-asas@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.pos_ != "ADV":
                continue
            as_kids = [c for c in t.children if c.lower_ == "as"]
            if any(c.i < t.i for c in as_kids) and any(c.i > t.i for c in as_kids):
                lo = min(c.i for c in as_kids if c.i < t.i)
                hi = max(c.i for c in as_kids if c.i > t.i)
                span = doc[lo:hi + 1]
                out.append(Annotation(
                    text_id=text_id, construct_id="ADV-28",
                    span=Span(span.start_char, span.end_char, lo, hi),
                    detector_type=self.detector_type,
                    detector_version=self.version, confidence=1.0,
                    evidence={"tokens": list(range(lo, hi + 1)),
                              "matched": span.text}))
        return out


# ---- gates for the flat-adverb split -------------------------------------- #
_RESPONSE_FLATS = {"well", "right", "okay", "fine", "sure", "alright"}


def _gate_flat_citation(doc, start, end):
    # ADV-02: flat form NOT used as a verb's manner adverb (citation / adjectival).
    t = doc[start]
    if t.dep_ == "advmod" and t.head.pos_ == "VERB":
        return False
    # discourse/response uses of well/right/okay are INS/DMG, not flat adverbs
    if t.lower_ in _RESPONSE_FLATS:
        if t.dep_ in ("intj", "discourse"):
            return False
        first = next((x for x in t.sent if not x.is_punct and not x.is_space),
                     None)
        if first is t:
            return False
    return True


def _gate_flat_adverbial(doc, start, end):
    # ADV-03: flat form used adverbially by a verb.
    t = doc[start]
    return t.dep_ == "advmod" and t.head.pos_ == "VERB"


def build(nlp, client=None):
    dets = []

    # -- Formation ------------------------------------------------------------
    dets.append(_TokenRule("ADV-01", _p_regular_ly, "adv01-ly@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ADV-02", sorted(_FLAT),
                                      gate=_gate_flat_citation,
                                      version="adv02-flat-cite@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ADV-03", sorted(_FLAT),
                                      gate=_gate_flat_adverbial,
                                      version="adv03-flat-adv@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ADV-04", sorted(_DISTINCT_LY),
                                      version="adv04-pairs@0.1"))

    # -- Types (closed classes) ----------------------------------------------
    dets.append(_TokenRule("ADV-05", _p_manner, "adv05-manner@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ADV-06", sorted(_PLACE),
                                      version="adv06-place@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ADV-07", sorted(_DEF_TIME),
                                      version="adv07-deftime@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ADV-08", sorted(_INDEF_FREQ),
                                      version="adv08-indeffreq@0.1"))
    dets.append(PhraseLexiconDetector(
        nlp, "ADV-09", sorted(_DEF_FREQ_WORDS) + _DEF_FREQ_PHRASES,
        version="adv09-deffreq@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ADV-12", sorted(_DOMAIN),
                                      version="adv12-domain@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "ADV-13", sorted(_STANCE),
                                      version="adv13-stance@0.1"))
    dets.append(PhraseLexiconDetector(
        nlp, "ADV-14", sorted(_LINK_WORDS) + _LINK_PHRASES,
        version="adv14-linking@0.1"))

    # -- Position & scope (rules) --------------------------------------------
    dets.append(_TokenRule("ADV-15", _p_front, "adv15-front@0.1"))
    dets.append(_TokenRule("ADV-16", _p_mid, "adv16-mid@0.1"))
    dets.append(_TokenRule("ADV-17", _p_end, "adv17-end@0.1"))
    dets.append(_TokenRule("ADV-20", _p_freq_imperative, "adv20-freqimp@0.1"))

    # -- Degree details (rules) ----------------------------------------------
    dets.append(_TokenRule("ADV-22", _p_mod_comparative, "adv22-modcomp@0.1"))
    dets.append(_TokenRule("ADV-23", _p_intens_verb, "adv23-intensverb@0.1",
                           detector_type="lexicon"))
    dets.append(_TokenRule("ADV-24", _p_vrs, "adv24-vrs@0.1"))
    dets.append(_TokenRule("ADV-26", _p_too_enough, "adv26-tooenough@0.1"))

    # -- Comparison of adverbs (rules) ---------------------------------------
    dets.append(_TokenRule("ADV-27", _p_comp_superl_adv, "adv27-compsup@0.1"))
    dets.append(_AsAsRule())

    # -- LLM tier: reading/scope decided by meaning (skipped offline) ---------
    def _reading(cid, form, desc):
        sys = (f"Decide whether the marked adverb is: {desc}. "
               f'Return JSON {{"construct_id":"{cid}","confidence":0..1,'
               f'"rationale":"..."}} or use construct_id "NONE".')
        return LLMReadingDetector(nlp, [cid], form, sys, client=client,
                                  version=f"{cid.lower()}-llm@0.1")

    _adv_anchor = [{"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": "ADV"}}]
    dets.append(_reading(
        "ADV-10", _adv_anchor,
        "a degree adverb grading an adjective/adverb "
        "(very/too/quite/rather/extremely/fairly/pretty)"))
    dets.append(_reading(
        "ADV-11",
        [{"RIGHT_ID": "a", "RIGHT_ATTRS":
          {"LOWER": {"IN": sorted(_FOCUS)}}}],
        "a focusing adverb (only/even/just/also/too/as well/either) restricting "
        "or highlighting a constituent"))
    dets.append(_reading(
        "ADV-18", [{"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}}],
        "the clause ordering manner then place then time adverbials"))
    dets.append(_reading(
        "ADV-21", [{"RIGHT_ID": "a", "RIGHT_ATTRS": {"LOWER": "ever"}}],
        "'ever' used in a question or a negative/non-assertive context"))
    dets.append(_reading(
        "ADV-29",
        [{"RIGHT_ID": "a", "RIGHT_ATTRS":
          {"LOWER": {"IN": ["still", "already", "yet"]}}}],
        "a phase adverb: still (continuation) / already (earlier than expected) "
        "/ yet (expected but not yet)"))

    dets.append(LLMStandaloneDetector(
        "ADV-19",
        'Decide whether the sentence turns on the SCOPE of a focusing adverb '
        '(only/even/just): which constituent it associates with. Return JSON '
        '{"construct_id":"ADV-19"|"NONE","confidence":0..1,"rationale":"..."}.',
        client=client, version="adv19-llm@0.1"))
    dets.append(LLMStandaloneDetector(
        "ADV-25",
        'Decide whether the sentence contains a downtoning degree adverb '
        '(quite/rather/fairly/pretty), including the quite ambiguity '
        '(=fairly vs =completely). Return JSON {"construct_id":"ADV-25"|"NONE",'
        '"confidence":0..1,"rationale":"..."}.',
        client=client, version="adv25-llm@0.1"))

    return dets
