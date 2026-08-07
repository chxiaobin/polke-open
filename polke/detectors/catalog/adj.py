"""Category ADJ - adjectives (ADJ-01..20).

Position:
  ADJ-01 attributive (amod before a noun, minus the attributive-only and
         postpositive classes), ADJ-02 predicative (acomp), ADJ-03
         attributive-only lexicon (main/former/sheer), ADJ-04 predicative-only
         a-adjectives + well/ill, ADJ-05 postpositive (indef-pronoun+adj,
         noun+postpositive-adj, fixed titles).
Order/gradability/form (LLM tier unless morphological): ADJ-13 compound
  (hyphenated pre/post-nominal), ADJ-15 comparative/superlative (JJR/JJS).
Complementation: ADJ-16 adj+to-infinitive, ADJ-18 adj+that-clause, ADJ-19
  adj+preposition, ADJ-20 too/enough/so.
LLM tier (registered, skipped offline): ADJ-06 order, ADJ-07 coordinate vs
  cumulative, ADJ-08 gradable+very, ADJ-09 non-gradable+absolutely, ADJ-10
  classifying, ADJ-11 -ing participial, ADJ-12 -ed participial, ADJ-14 adj as
  noun, ADJ-17 tough-movement.

spaCy en_core_web_sm labels verified empirically: attributive adj is amod
(precedes head noun); predicative adj is acomp; a-adjectives tag JJ; the
'too/so' intensifier is RB(advmod), 'enough' is RB(advmod) postposed; adj+to
is adj>xcomp(VERB)>aux(TO); hyphens are HYPH punctuation splitting the compound.
"""
from __future__ import annotations
from ..base import Detector
from ..routing import RuleRoutingDetector, LexiconRuleDetector, CompositeDetector
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector
from ...schema import Annotation, Span

# Attributive-only adjectives (no predicative use).
_ATTR_ONLY = {"main", "former", "sheer", "mere", "utter", "chief", "principal",
              "sole", "only", "elder", "inner", "outer", "upper", "lone",
              "eventual", "outright", "sheer", "downright", "very"}
# Predicative-only: a-adjectives + health words.
_PRED_ONLY = ["asleep", "afraid", "alive", "alone", "aware", "awake",
              "ashamed", "aghast", "adrift", "afloat", "aback", "aflame",
              "aglow", "akin", "aloof", "amiss", "askew", "awash", "unwell",
              "well", "ill", "poorly", "content", "aloof"]
# Postpositive adjectives that follow their noun.
_POSTPOS_ADJ = {"present", "involved", "concerned", "responsible", "available",
                "elect", "galore", "aplenty", "incarnate", "proper",
                "designate", "unknown", "affected"}
_POSTPOS_PHRASES = ["attorney general", "secretary general", "court martial",
                    "heir apparent", "president elect", "notary public",
                    "body politic", "poet laureate", "time immemorial",
                    "director general", "surgeon general", "sum total"]
_INDEF_PRON = {"something", "anything", "everything", "nothing", "somebody",
               "anybody", "everybody", "nobody", "someone", "anyone", "everyone"}
# Tough-movement adjectives (ADJ-17): excluded from ADJ-16 when a raised
# subject is present.
_TOUGH_ADJ = {"hard", "easy", "difficult", "tough", "impossible", "tricky",
              "simple", "awkward", "pleasant", "unpleasant", "dangerous",
              "safe", "comfortable", "expensive", "cheap", "nice", "good",
              "bad", "painful", "fun", "interesting", "boring", "convenient",
              "possible", "straightforward"}
# Adjectives that license a that-clause (ADJ-18).
_THAT_ADJ = {"sure", "certain", "glad", "happy", "sorry", "afraid", "aware",
             "confident", "convinced", "hopeful", "worried", "surprised",
             "pleased", "sad", "proud", "angry", "disappointed", "positive",
             "clear", "obvious", "evident", "apparent", "likely", "possible",
             "probable", "true", "lucky", "fortunate", "amazed", "astonished",
             "delighted", "grateful", "thankful", "anxious", "concerned",
             "doubtful", "adamant", "insistent", "keen"}
# Adjective + dependent preposition pairs (ADJ-19).
_ADJ_PREP = {
    ("good", "at"), ("bad", "at"), ("brilliant", "at"), ("hopeless", "at"),
    ("interested", "in"), ("involved", "in"), ("successful", "in"),
    ("afraid", "of"), ("fond", "of"), ("proud", "of"), ("aware", "of"),
    ("capable", "of"), ("full", "of"), ("jealous", "of"), ("scared", "of"),
    ("tired", "of"), ("sick", "of"), ("guilty", "of"), ("short", "of"),
    ("sure", "of"), ("certain", "of"), ("ashamed", "of"), ("terrified", "of"),
    ("responsible", "for"), ("famous", "for"), ("good", "for"), ("ready", "for"),
    ("suitable", "for"), ("sorry", "for"), ("grateful", "for"), ("known", "for"),
    ("keen", "on"), ("dependent", "on"), ("based", "on"), ("intent", "on"),
    ("similar", "to"), ("married", "to"), ("used", "to"), ("kind", "to"),
    ("close", "to"), ("related", "to"), ("addicted", "to"), ("accustomed", "to"),
    ("different", "from"), ("angry", "with"), ("angry", "at"), ("bored", "with"),
    ("pleased", "with"), ("familiar", "with"), ("busy", "with"),
    ("worried", "about"), ("excited", "about"), ("nervous", "about"),
    ("happy", "about"), ("concerned", "about"), ("curious", "about"),
    ("crazy", "about"), ("serious", "about"),
}


class _Scan(Detector):
    def __init__(self, cid, fn, detector_type="rule", version="0.1"):
        self.construct_ids = [cid]
        self._fn = fn
        self.detector_type = detector_type
        self.version = version

    def match(self, doc, text_id="doc"):
        out, seen = [], set()
        for lo, hi in self._fn(doc):
            if (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            span = doc[lo:hi + 1]
            out.append(Annotation(
                text_id=text_id, construct_id=self.construct_ids[0],
                span=Span(span.start_char, span.end_char, lo, hi),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text}))
        return out


# --- position -------------------------------------------------------------- #
_AMOD_FORM = [{"RIGHT_ID": "a", "RIGHT_ATTRS": {"DEP": "amod"}}]
_ACOMP_FORM = [{"RIGHT_ID": "a", "RIGHT_ATTRS": {"DEP": "acomp", "TAG": "JJ"}}]


def _classify_attributive(doc, tids):
    a = doc[tids[0]]
    if a.tag_ != "JJ":
        return None                                   # comp/sup -> ADJ-15
    if a.head.pos_ not in ("NOUN", "PROPN"):
        return None
    if a.i >= a.head.i:
        return None                                   # postposed -> ADJ-05
    if a.lemma_.lower() in _ATTR_ONLY:
        return None                                   # -> ADJ-03
    if a.head.lower_ in _INDEF_PRON:
        return None                                   # -> ADJ-05
    return "ADJ-01"


def _classify_attr_only(doc, tids):
    a = doc[tids[0]]
    return "ADJ-03" if a.lemma_.lower() in _ATTR_ONLY else None


def _adj05(doc):
    for t in doc:
        if t.pos_ == "ADJ" and t.i > 0 and doc[t.i - 1].lower_ in _INDEF_PRON:
            yield (t.i, t.i)
        if t.lower_ in _POSTPOS_ADJ and t.i > 0 \
                and doc[t.i - 1].pos_ in ("NOUN", "PROPN"):
            yield (doc[t.i - 1].i, t.i)


# --- form ------------------------------------------------------------------- #
def _adj13(doc):
    for a in doc:
        if a.pos_ == "ADJ" and a.dep_ in ("amod", "acomp", "ROOT", "conj"):
            sub = list(a.subtree)
            if any(x.tag_ == "HYPH" or x.text == "-" for x in sub):
                lo = min(x.i for x in sub)
                hi = max(x.i for x in sub)
                yield (lo, hi)


_JJRS_FORM = [{"RIGHT_ID": "a", "RIGHT_ATTRS": {"TAG": {"IN": ["JJR", "JJS"]}}}]


# --- complementation -------------------------------------------------------- #
_TO_INF_FORM = [
    {"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": "ADJ"}},
    {"LEFT_ID": "a", "REL_OP": ">", "RIGHT_ID": "v",
     "RIGHT_ATTRS": {"DEP": "xcomp", "POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "to",
     "RIGHT_ATTRS": {"TAG": "TO", "DEP": "aux"}},
]


def _classify_to_inf(doc, tids):
    a = doc[tids[0]]
    if any(c.lower_ in ("too", "enough") for c in a.children):
        return None                                   # -> ADJ-20
    if a.dep_ == "acomp" and a.lemma_.lower() in _TOUGH_ADJ:
        if any(c.dep_ in ("nsubj", "nsubjpass") for c in a.head.children):
            return None                               # tough-movement -> ADJ-17
    return "ADJ-16"


def _adj18(doc):
    for a in doc:
        if a.pos_ == "ADJ" and a.lemma_.lower() in _THAT_ADJ:
            has_ccomp = any(c.dep_ in ("ccomp", "acl") for c in a.children)
            has_that = any(t.lower_ == "that" and a.i < t.i <= a.i + 3
                           for t in doc)
            if has_ccomp or has_that:
                yield (a.i, a.i)


_ADJPREP_FORM = [
    {"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": "ADJ"}},
    {"LEFT_ID": "a", "REL_OP": ">", "RIGHT_ID": "p",
     "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "prt", "agent"]}}},
]


def _adjprep_key(doc, tids):
    a = p = None
    for i in tids:
        tok = doc[i]
        if tok.dep_ in ("prep", "prt", "agent"):
            p = tok
        elif tok.pos_ == "ADJ":
            a = tok
    if a is None or p is None:
        return None
    return (a.lemma_.lower(), p.lower_)


_TOO_FORM = [{"RIGHT_ID": "x",
              "RIGHT_ATTRS": {"LOWER": {"IN": ["too", "so", "enough"]}}}]


def _classify_toenoughso(doc, tids):
    t = doc[tids[0]]
    if t.dep_ == "advmod" and t.head.pos_ in ("ADJ", "ADV"):
        return "ADJ-20"
    return None


# --- LLM tier forms --------------------------------------------------------- #
_TWOAMOD_FORM = [
    {"RIGHT_ID": "n", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "n", "REL_OP": ">", "RIGHT_ID": "a1",
     "RIGHT_ATTRS": {"DEP": "amod"}},
    {"LEFT_ID": "n", "REL_OP": ">", "RIGHT_ID": "a2",
     "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "conj"]}}},
]
_AMODJJ_FORM = [{"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": "ADJ"}}]
# -ing participial adjectives are tagged VBG in predicative but often plain
# JJ in attributive position (an interesting book) — accept both.
_ING_FORM = [{"RIGHT_ID": "a", "RIGHT_ATTRS": {
    "TAG": {"IN": ["VBG", "JJ"]}, "LOWER": {"REGEX": "ing$"}}}]
_ED_FORM = [{"RIGHT_ID": "a", "RIGHT_ATTRS": {"TAG": {"IN": ["VBN", "JJ"]}}}]
_THEADJ_FORM = [
    {"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": "ADJ"}},
    {"LEFT_ID": "a", "REL_OP": ">", "RIGHT_ID": "d",
     "RIGHT_ATTRS": {"LOWER": "the", "DEP": "det"}},
]
_TOUGH_FORM = [
    {"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": "ADJ", "DEP": "acomp"}},
    {"LEFT_ID": "a", "REL_OP": ">", "RIGHT_ID": "v",
     "RIGHT_ATTRS": {"DEP": "xcomp", "POS": "VERB"}},
]


# === Part VI addition (Phase 3): ADJ-21 Adj + for + NP + to-infinitive =====

def _adj21(doc):
    """The for...to frame licensed by an adjective, incl. extraposed "It's
    important for us to leave" / "I'm eager for it to end": the advcl(mark
    for + subj + to) hangs off a copula that carries the acomp adjective.
    Verbal heads without an acomp ("We arranged for him to travel") are
    VCP-41 — the mirror of vcp.py's exclude."""
    from .common import for_to_advcl
    for be in doc:
        adj = next((c for c in be.children if c.dep_ == "acomp"
                    and c.pos_ == "ADJ"), None)
        if adj is None:
            continue
        cl = for_to_advcl(be)
        if cl is None:
            cl = for_to_advcl(adj)
        if cl is None:
            continue
        toks = [c.i for c in cl.children] + [cl.i, adj.i]
        yield (min(toks), max(toks))


def build(nlp, client=None):
    dets = []

    # ADJ-01 attributive, ADJ-02 predicative, ADJ-03 attributive-only.
    dets.append(RuleRoutingDetector(nlp, ["ADJ-01"], _AMOD_FORM,
                                    _classify_attributive, span="match",
                                    version="adj01-attributive@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["ADJ-02"], _ACOMP_FORM,
                                    lambda d, t: "ADJ-02", span="match",
                                    version="adj02-predicative@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["ADJ-03"], _AMOD_FORM,
                                    _classify_attr_only, span="match",
                                    version="adj03-attronly@0.1"))
    # ADJ-04 predicative-only a-adjectives (surface lexicon).
    dets.append(PhraseLexiconDetector(nlp, "ADJ-04", _PRED_ONLY,
                                      version="adj04-predonly@0.1"))
    # ADJ-05 postpositive: indef-pronoun+adj / noun+postpositive-adj / titles.
    dets.append(CompositeDetector("ADJ-05", [
        _Scan("ADJ-05", _adj05, detector_type="lexicon"),
        PhraseLexiconDetector(nlp, "ADJ-05", _POSTPOS_PHRASES),
    ], detector_type="lexicon", version="adj05-postpositive@0.1"))

    # ADJ-13 compound (hyphenated), ADJ-15 comparative/superlative form.
    dets.append(_Scan("ADJ-13", _adj13, detector_type="lexicon",
                      version="adj13-compound@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["ADJ-15"], _JJRS_FORM,
                                    lambda d, t: "ADJ-15", span="match",
                                    version="adj15-comparison@0.1"))

    # ADJ-16 adj+to-inf, ADJ-18 adj+that, ADJ-19 adj+prep, ADJ-20 too/enough/so.
    dets.append(RuleRoutingDetector(nlp, ["ADJ-16"], _TO_INF_FORM,
                                    _classify_to_inf, span="match",
                                    version="adj16-toinf@0.1"))
    dets.append(_Scan("ADJ-18", _adj18, detector_type="lexicon",
                      version="adj18-thatclause@0.1"))
    dets.append(LexiconRuleDetector(nlp, "ADJ-19", _ADJPREP_FORM, _ADJ_PREP,
                                    _adjprep_key, span="match",
                                    version="adj19-adjprep@0.1"))
    dets.append(RuleRoutingDetector(nlp, ["ADJ-20"], _TOO_FORM,
                                    _classify_toenoughso, span="match",
                                    version="adj20-too-enough-so@0.1"))

    # --- LLM tier (registered, skipped offline) ---
    dets.append(LLMReadingDetector(
        nlp, ["ADJ-06"], _TWOAMOD_FORM,
        "Two or more attributive adjectives premodify the bracketed noun. Return "
        "ADJ-06 if they follow the canonical order "
        "(opinion-size-age-shape-colour-origin-material-purpose), e.g. a large "
        "round wooden table. Comma-separated coordinated adjectives (a long, "
        "hot summer) are a different construct - return NONE for those.",
        client=client, version="adj06-order@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["ADJ-07"], _AMODJJ_FORM,
        "Return ADJ-07 if the noun phrase has comma-separated coordinate "
        "adjectives (a long, hot summer) rather than an unpunctuated cumulative "
        "string.",
        client=client, version="adj07-coordinate@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["ADJ-08"], _AMODJJ_FORM,
        "Return ADJ-08 if the bracketed adjective is gradable and a scalar "
        "degree modifier such as very/quite/rather is actually present with it "
        "in the text (very tired). A gradable adjective with no degree word "
        "returns NONE.",
        client=client, version="adj08-gradable@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["ADJ-09"], _AMODJJ_FORM,
        "Return ADJ-09 if the bracketed adjective is non-gradable/absolute and "
        "an intensifier like absolutely/completely/utterly is actually present "
        "with it in the text (absolutely freezing). Without such an "
        "intensifier, or with plain very + gradable adjective, return NONE.",
        client=client, version="adj09-nongradable@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["ADJ-10"], _AMODJJ_FORM,
        "Return ADJ-10 if the bracketed adjective is a classifying/relational "
        "(non-gradable) adjective naming a type (nuclear power, a daily "
        "routine), not a qualitative one.",
        client=client, version="adj10-classifying@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["ADJ-11"], _ING_FORM,
        "Return ADJ-11 if the bracketed -ing form is a participial adjective "
        "describing the source of a feeling (an interesting book; the news is "
        "boring).",
        client=client, version="adj11-ing@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["ADJ-12"], _ED_FORM,
        "Return ADJ-12 if the bracketed -ed form is a participial adjective "
        "describing the experiencer of a feeling (I'm interested/bored).",
        client=client, version="adj12-ed@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["ADJ-14"], _THEADJ_FORM,
        "Return ADJ-14 if the + adjective is used as a noun denoting a class or "
        "abstraction (the poor; the impossible; the unknown).",
        client=client, version="adj14-asnoun@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["ADJ-17"], _TOUGH_FORM,
        "Return ADJ-17 if this is tough-movement: the subject is the notional "
        "object of the infinitive (The book is hard to read = it is hard to read "
        "the book).",
        client=client, version="adj17-tough@0.1"))
    from .common import Scan
    dets.append(Scan("ADJ-21", _adj21, version="adj21-for-to@0.1"))
    return dets
