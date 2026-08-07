"""Category COM - comparison (COM-01..18).

Routing is on the morphology of the comparative/superlative marker:
  -er/-est inflection  -> JJR/JJS/RBR/RBS token (COM-01/05).
  more/most            -> RBR/RBS degree adverb on an adjective/adverb
                          (COM-02/06), gated to exclude the two-syllable
                          variable adjectives (clever/simple...) that are the
                          COM-03 hybrid tier.
  irregular            -> better/worse/best/worst/further/farther... lexicon
                          (COM-04); less/least/fewer are the decreasing set
                          (COM-15); more/most as quantifiers (not degree
                          adverbs) also fall to COM-04.
  as...as / so...as    -> equative (COM-09) vs negated equative (COM-10).
  much/far/slightly... -> modified comparative (COM-12).
  the +comp,the +comp  -> proportional (COM-13); comp and comp -> incremental
                          (COM-14).
  so/such (+that)      -> COM-18.
LLM tier (registered, skipped offline): COM-03 two-syllable variation,
COM-07 superlative+restriction, COM-08 than-clause, COM-11 multiples/scaled,
COM-16 like/unlike/as.

spaCy en_core_web_sm labels verified empirically: 'more'/'most'/'less' are
RBR/RBS ADV with dep advmod; inflected comparatives are JJR/RBR, superlatives
JJS/RBS; equative 'as' is RB(advmod) on the pivot adjective plus a second
IN 'as'; 'such' is PDT(predet).
"""
from __future__ import annotations
from ..base import Detector
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector
from ...schema import Annotation, Span

# good/bad/far irregulars. less/least/fewer -> COM-15; more/most -> COM-02/06
# (as degree adverbs) or COM-04 (as quantifiers, handled in _com04).
_IRREGULAR = {"better", "worse", "best", "worst", "further", "farther",
              "furthest", "farthest", "elder", "eldest"}
# two-syllable adjectives that take BOTH -er and 'more' -> COM-03 (LLM tier);
# excluded from the deterministic COM-01/02/05/06 so those stay precise.
_TWO_SYLL = {"clever", "simple", "narrow", "quiet", "gentle", "common",
             "pleasant", "cruel", "polite", "stupid", "shallow", "handsome",
             "noble", "humble", "subtle", "feeble", "friendly", "likely",
             "lovely", "little", "profound"}
_COMP_TAGS = {"JJR", "RBR", "JJS", "RBS"}
_MOD_WORDS = {"much", "far", "slightly", "even", "still", "way", "somewhat",
              "considerably", "significantly", "marginally", "rather", "no",
              "any", "lot", "bit", "loads", "miles"}
_DECREASE = {"less", "least", "fewer", "fewest", "lesser"}


class _Scan(Detector):
    """Fire a construct id on each (lo, hi) token span yielded by ``fn(doc)``."""

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


def _is_neg(doc):
    return any(t.dep_ == "neg" or t.lower_ in ("not", "never", "n't", "no")
              for t in doc)


def _has_multiple(doc):
    if any(t.lower_ in ("twice", "half") for t in doc):
        return True
    return any(t.lower_ in ("times", "time") and t.i > 0 and doc[t.i - 1].like_num
              for t in doc)


def _equative_info(doc):
    """(equative, so_equative) - an adj/adv flanked by 'as ... as', or a 'so
    ADJ ... as' pattern."""
    as_idx = [t.i for t in doc if t.lower_ == "as"]
    equative = False
    for t in doc:
        if t.pos_ in ("ADJ", "ADV"):
            if any(i < t.i for i in as_idx) and any(i > t.i for i in as_idx):
                equative = True
    so_equative = False
    for t in doc:
        if t.lower_ == "so" and t.dep_ == "advmod" and t.head.pos_ in ("ADJ", "ADV"):
            if any(i > t.head.i for i in as_idx):
                so_equative = True
    return equative, so_equative


# --- morphology of the comparative marker ----------------------------------- #
def _com01(doc):
    for t in doc:
        if (t.tag_ in ("JJR", "RBR") and t.lower_ not in ({"more", "less"} | _IRREGULAR)
                and t.lower_.endswith("er") and len(t.text) > 3
                and t.lemma_.lower() not in _TWO_SYLL):
            yield (t.i, t.i)


def _com02(doc):
    for t in doc:
        if (t.lower_ == "more" and t.tag_ in ("RBR", "JJR") and t.dep_ == "advmod"
                and t.head.pos_ in ("ADJ", "ADV")
                and t.head.lemma_.lower() not in _TWO_SYLL):
            # skip the COM-14 'more and more' reduplication
            if t.i + 2 < len(doc) and doc[t.i + 1].lower_ == "and" \
                    and doc[t.i + 2].lower_ == "more":
                continue
            lo, hi = sorted((t.i, t.head.i))
            yield (lo, hi)


def _com04(doc):
    for t in doc:
        w = t.lower_
        if w in _IRREGULAR:
            yield (t.i, t.i)
        elif w in ("more", "most"):
            # quantifier use (much/many -> more), NOT the periphrastic degree
            # adverb modifying an adjective/adverb (that is COM-02/06).
            if not (t.tag_ in ("RBR", "RBS") and t.head.pos_ in ("ADJ", "ADV")):
                yield (t.i, t.i)


def _com05(doc):
    for t in doc:
        if (t.tag_ in ("JJS", "RBS")
                and t.lower_ not in ("most", "least", "best", "worst",
                                     "furthest", "farthest")
                and t.lower_.endswith("est")
                and t.lemma_.lower() not in _TWO_SYLL):
            yield (t.i, t.i)


def _com06(doc):
    for t in doc:
        if (t.lower_ == "most" and t.tag_ in ("RBS", "JJS") and t.dep_ == "advmod"
                and t.head.pos_ in ("ADJ", "ADV")
                and t.head.lemma_.lower() not in _TWO_SYLL):
            lo, hi = sorted((t.i, t.head.i))
            yield (lo, hi)


# --- standard & modification ------------------------------------------------ #
def _com09(doc):
    equative, so_eq = _equative_info(doc)
    if equative and not _is_neg(doc) and not _has_multiple(doc):
        idx = [t.i for t in doc if t.lower_ == "as"]
        if idx:
            yield (min(idx), max(idx))


def _com10(doc):
    equative, so_eq = _equative_info(doc)
    if (equative or so_eq) and _is_neg(doc):
        idx = [t.i for t in doc if t.lower_ in ("as", "so")]
        if idx:
            yield (min(idx), max(idx))


def _com12(doc):
    for t in doc:
        if (t.lower_ in _MOD_WORDS and t.dep_ in ("advmod", "npadvmod")
                and t.head.pos_ in ("ADJ", "ADV")):
            lo, hi = sorted((t.i, t.head.i))
            yield (lo, hi)


def _com13(doc):
    the_comp = [t for t in doc if t.lower_ == "the"
                and t.head.tag_ in _COMP_TAGS]
    if len(the_comp) >= 2:
        idx = [t.i for t in the_comp] + [t.head.i for t in the_comp]
        yield (min(idx), max(idx))


def _com14(doc):
    for i in range(len(doc) - 2):
        if doc[i + 1].lower_ == "and":
            a, b = doc[i].lower_, doc[i + 2].lower_
            if a == b and (doc[i].tag_ in _COMP_TAGS or a in ("more", "less")):
                yield (i, i + 2)


def _com15(doc):
    for t in doc:
        if t.lower_ in _DECREASE and t.tag_ in (_COMP_TAGS | {"JJ", "RB", "DT"}):
            yield (t.i, t.i)


def _com18(doc):
    for t in doc:
        if t.lower_ == "so" and t.dep_ == "advmod" and t.head.pos_ in ("ADJ", "ADV"):
            adj = t.head
            has_that = any(c.lower_ == "that" and c.dep_ == "mark"
                           for c in adj.children) \
                or any(x.lower_ == "that" and x.i > adj.i for x in doc)
            if has_that:
                yield (t.i, adj.i)
        if t.lower_ == "such" and t.tag_ in ("PDT", "DT") \
                and t.dep_ in ("predet", "det"):
            yield (t.i, t.i)


# --- LLM tier forms --------------------------------------------------------- #
_ADJ_FORM = [{"RIGHT_ID": "a", "RIGHT_ATTRS": {"POS": {"IN": ["ADJ", "ADV"]}}}]
_THAN_FORM = [{"RIGHT_ID": "t", "RIGHT_ATTRS": {"LOWER": "than"}}]
_AS_FORM = [{"RIGHT_ID": "t", "RIGHT_ATTRS": {"LOWER": {"IN": ["as", "times", "twice", "half"]}}}]
_LIKE_FORM = [{"RIGHT_ID": "t", "RIGHT_ATTRS": {"LOWER": {"IN": ["like", "unlike", "as"]}}}]


def build(nlp, client=None):
    dets = [
        _Scan("COM-01", _com01, version="com01-inflect@0.1"),
        _Scan("COM-02", _com02, version="com02-more@0.1"),
        _Scan("COM-04", _com04, detector_type="lexicon", version="com04-irregular@0.1"),
        _Scan("COM-05", _com05, version="com05-inflect@0.1"),
        _Scan("COM-06", _com06, version="com06-most@0.1"),
        _Scan("COM-09", _com09, version="com09-equative@0.1"),
        _Scan("COM-10", _com10, version="com10-negequative@0.1"),
        _Scan("COM-12", _com12, version="com12-modified@0.1"),
        _Scan("COM-13", _com13, detector_type="lexicon", version="com13-proportional@0.1"),
        _Scan("COM-14", _com14, detector_type="lexicon", version="com14-incremental@0.1"),
        _Scan("COM-15", _com15, version="com15-decreasing@0.1"),
        _Scan("COM-18", _com18, version="com18-so-such@0.1"),
    ]

    # COM-17 the same as / similar to / different from(-to/-than) etc.
    dets.append(PhraseLexiconDetector(
        nlp, "COM-17",
        ["the same as", "same as", "similar to", "different from",
         "different to", "different than", "identical to", "equal to",
         "comparable to", "in contrast to", "as opposed to"],
        version="com17-similarity@0.1"))

    # --- LLM tier (registered, skipped offline) ---
    dets.append(LLMReadingDetector(
        nlp, ["COM-03"], _ADJ_FORM,
        "The bracketed adjective is two syllables and idiomatically allows "
        "BOTH -er/-est and more/most (clever, narrow, simple, quiet: cleverer "
        "/ more clever). Return COM-03 only for such both-form adjectives; "
        "adjectives with one standard form (happy -> happier; useful -> more "
        "useful) and adverbs return NONE.",
        client=client, version="com03-twosyll@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["COM-07"], _ADJ_FORM,
        "The bracketed superlative carries a restriction: a following in/of "
        "phrase or an ever + present-perfect relative (the best film I've ever "
        "seen; the tallest in the class). Return COM-07.",
        client=client, version="com07-restriction@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["COM-08"], _THAN_FORM,
        "The bracketed comparative has a than-standard. Return COM-08 if this is "
        "a than-clause/phrase comparison (taller than me / than I am), covering "
        "pronoun case and ellipsis.",
        client=client, version="com08-than@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["COM-11"], _AS_FORM,
        "Return COM-11 if this is a multiple/scaled equative: twice/half/three "
        "times + as + adjective, or N times the + noun (three times the size).",
        client=client, version="com11-multiples@0.1"))
    dets.append(LLMReadingDetector(
        nlp, ["COM-16"], _LIKE_FORM,
        "Return COM-16 if like/unlike (preposition of similarity, 'He runs like "
        "the wind') or manner/role 'as' ('Do as I say') expresses "
        "similarity/difference.",
        client=client, version="com16-like@0.1"))
    return dets
