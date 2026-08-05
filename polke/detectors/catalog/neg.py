"""Category NEG - negation (NEG-01..13).

Rule/lexicon tier (tested offline):
  NEG-01 verb negation (not/n't on an operator, with a nominal subject and no
         non-assertive item / negative determiner -> those are NEG-03/04/05).
  NEG-02 the contraction forms themselves (a metalinguistic list of >=2 negative
         contractions, or the special inverted "aren't I").
  NEG-04 negative determiner "no" + noun, or a negative pronoun (nobody/nothing).
  NEG-06 semi-negative adverbs (seldom/rarely/hardly/scarcely/barely/little).
  NEG-07 no longer / not any longer / no more.
  NEG-09 echoing either / neither / nor.
  NEG-13 negative-adverbial fronting + subject-operator inversion.
LLM tier (registered, skipped offline):
  NEG-03 non-assertive items in negative scope; NEG-05 no vs not a/not any;
  NEG-08 transferred/raised negation; NEG-10 constituent (local) negation;
  NEG-11 scope ambiguity (partial vs total); NEG-12 litotes.

spaCy en_core_web_sm labels verified: "not"/"n't" is ``neg``; "no" determiner is
DT ``det``; negative pronouns nobody/nothing parse as NN; fronted "never"/"no"
attach as ``neg``/``advmod`` with the operator (aux/MD) preceding the subject.
"""
from __future__ import annotations
from ...schema import Annotation, Span
from ..base import Detector
from ..lexical import PhraseLexiconDetector
from ..routing import RuleRoutingDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector

_NEG_WORDS = {"not", "n't", "n’t"}
_NONASSERTIVE = {"any", "anybody", "anyone", "anything", "anywhere", "either"}
_ANY_SERIES = {"any", "anybody", "anyone", "anything", "anywhere"}
_NEG_PRONOUNS = {"nobody", "nothing", "none", "nowhere", "noone"}
_NEG13_FRONT = {"never", "rarely", "seldom", "hardly", "scarcely", "barely",
                "little", "nowhere", "no", "not", "neither", "nor"}


def _has_any_series(doc):
    return any(t.lower_ in _ANY_SERIES for t in doc)


def _has_no_det(doc):
    return any(t.lower_ == "no" and t.tag_ == "DT" for t in doc)


def _is_operator(t):
    return t.tag_ == "MD" or (t.lemma_ in ("do", "have", "be")
                              and t.tag_ in ("VBD", "VBP", "VBZ", "VB"))


def fronted_inversion(doc, front_set):
    """Sentence opens with an adverbial in ``front_set`` and inverts an operator
    before the subject.  Returns (lo, hi) span or None."""
    if len(doc) < 2 or doc[0].lower_ not in front_set:
        return None
    j = 1
    while j < len(doc) and doc[j].tag_ in ("RB", "RBR", "RBS"):
        j += 1
    if j >= len(doc) or not _is_operator(doc[j]):
        return None
    # a subject should follow the fronted operator (true inversion)
    if not any(t.pos_ in ("PRON", "NOUN", "PROPN") for t in doc[j + 1:j + 4]):
        return None
    return 0, j


# --------------------------------------------------------------------------- #
# NEG-01: verb negation not/n't on an operator.
# --------------------------------------------------------------------------- #
_NEG_FORM = [{"RIGHT_ID": "n",
              "RIGHT_ATTRS": {"DEP": "neg", "LOWER": {"IN": ["not", "n't"]}}}]


def _classify_neg01(doc, token_ids):
    neg = doc[token_ids[0]]
    h = neg.head
    if h.pos_ not in ("VERB", "AUX"):
        return None
    is_be = h.lemma_ == "be"
    has_aux = any(c.dep_ in ("aux", "auxpass") for c in h.children)
    if not (is_be or has_aux):
        return None
    subj = [c for c in h.children if c.dep_ in ("nsubj", "nsubjpass")]
    if not any(s.pos_ in ("PRON", "NOUN", "PROPN") for s in subj):
        return None
    if _has_any_series(doc) or _has_no_det(doc):
        return None                       # -> NEG-03 / NEG-05
    return "NEG-01"


# --------------------------------------------------------------------------- #
# NEG-02: the negative contractions themselves.
# --------------------------------------------------------------------------- #
class _Contractions(Detector):
    construct_ids = ["NEG-02"]
    detector_type = "rule"
    version = "neg02-contractions@0.1"

    def _emit(self, doc, text_id, lo, hi, matched):
        span = doc[lo:hi + 1]
        return Annotation(
            text_id=text_id, construct_id="NEG-02",
            span=Span(span.start_char, span.end_char, lo, hi),
            detector_type=self.detector_type, detector_version=self.version,
            confidence=1.0, evidence={"tokens": list(range(lo, hi + 1)),
                                      "matched": matched})

    def match(self, doc, text_id="doc"):
        negs = [t for t in doc if t.dep_ == "neg" and t.text.lower() == "n't"]
        out, seen = [], set()
        # (a) a metalinguistic list of >=2 negative contractions
        if len(negs) >= 2:
            idxs = sorted(t.i for t in negs) + sorted(t.head.i for t in negs)
            lo, hi = min(idxs), max(idxs)
            out.append(self._emit(doc, text_id, lo, hi, doc[lo:hi + 1].text))
            seen.add((lo, hi))
        # (b) the special inverted tag "aren't I"
        for t in negs:
            h = t.head
            if h.lemma_ in ("be", "do", "have") or h.tag_ == "MD":
                subj = [c for c in h.children
                        if c.dep_ == "nsubj" and c.i > t.i]
                if subj and subj[0].lower_ in ("i", "i."):
                    lo, hi = min(h.i, t.i), max(subj[0].i, t.i)
                    if (lo, hi) not in seen:
                        seen.add((lo, hi))
                        out.append(self._emit(doc, text_id, lo, hi,
                                              doc[lo:hi + 1].text))
        return out


# --------------------------------------------------------------------------- #
# NEG-04: negative determiner "no" + noun, or a negative pronoun.
# --------------------------------------------------------------------------- #
class _NegDeterminer(Detector):
    construct_ids = ["NEG-04"]
    detector_type = "rule"
    version = "neg04-det@0.1"

    def match(self, doc, text_id="doc"):
        if _has_any_series(doc):
            return []                     # "no X / not any X" contrast -> NEG-05
        out, seen = [], set()
        for t in doc:
            lo = hi = None
            if t.lower_ in _NEG_PRONOUNS and t.pos_ in ("NOUN", "PRON", "PROPN"):
                lo = hi = t.i
            elif t.lower_ == "no" and t.tag_ == "DT" and t.dep_ == "det":
                lo, hi = min(t.i, t.head.i), max(t.i, t.head.i)
            if lo is None or (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            span = doc[lo:hi + 1]
            out.append(Annotation(
                text_id=text_id, construct_id="NEG-04",
                span=Span(span.start_char, span.end_char, lo, hi),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0, evidence={"tokens": list(range(lo, hi + 1)),
                                          "matched": span.text}))
        return out


# --------------------------------------------------------------------------- #
# NEG-09: echoing either / neither / nor.
# --------------------------------------------------------------------------- #
class _NegEither(Detector):
    construct_ids = ["NEG-09"]
    detector_type = "rule"
    version = "neg09-either@0.1"

    def match(self, doc, text_id="doc"):
        has_neg = any(t.dep_ == "neg" for t in doc)
        out, seen = [], set()
        for t in doc:
            hit = t.lower_ in ("neither", "nor") or (t.lower_ == "either"
                                                     and has_neg)
            if not hit or t.i in seen:
                continue
            seen.add(t.i)
            out.append(Annotation(
                text_id=text_id, construct_id="NEG-09",
                span=Span(t.idx, t.idx + len(t), t.i, t.i),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0, evidence={"tokens": [t.i], "matched": t.text}))
        return out


# --------------------------------------------------------------------------- #
# NEG-13: negative-adverbial fronting + inversion.
# --------------------------------------------------------------------------- #
class _NegFronting(Detector):
    construct_ids = ["NEG-13"]
    detector_type = "rule"
    version = "neg13-fronting@0.1"

    def match(self, doc, text_id="doc"):
        hit = fronted_inversion(doc, _NEG13_FRONT)
        if not hit:
            return []
        lo, hi = hit
        span = doc[lo:hi + 1]
        return [Annotation(
            text_id=text_id, construct_id="NEG-13",
            span=Span(span.start_char, span.end_char, lo, hi),
            detector_type=self.detector_type, detector_version=self.version,
            confidence=1.0, evidence={"tokens": list(range(lo, hi + 1)),
                                      "matched": span.text})]


# --------------------------------------------------------------------------- #
# LLM tier (registered; skipped offline).
# --------------------------------------------------------------------------- #
_NEG03_SYS = ("Return NEG-03 if the negation licenses a non-assertive item "
              "(any/anybody/anything/ever/yet/either) inside its scope.")
_NEG05_SYS = ("Contrast of negative forms: return NEG-05 if the sentence uses "
              "'no + noun' where 'not a/not any + noun' would be equivalent.")
_NEG10_SYS = ("Return NEG-10 for constituent (local) negation - 'not' negating a "
              "phrase rather than the verb (not a sound; not surprisingly; a not "
              "unreasonable request).")
_NEG08_SYS = ('Return NEG-08 for negative-raising/transferred negation: the '
              'negation is on a matrix verb (think/believe/suppose/expect) but '
              'semantically negates the complement ("I don\'t think it\'ll work").')
_NEG11_SYS = ("Return NEG-11 if the sentence is ambiguous between partial and "
              "total negation (a quantifier such as all/every/everyone under the "
              "scope of not).")
_NEG12_SYS = ("Return NEG-12 for litotes/understatement: a negated negative or "
              "negated strong term used for positive/moderating effect (not bad; "
              "not unlike).")


# === Part VI addition (Phase 3): NEG-14 independent double negation ========

def _neg14(doc):
    """Two verbal negators with independent scope: (a) stacked on one verb
    group ("You ca n't not go" -> go has two neg children), or (b) "it's not
    that ..." + an internally negated ccomp ("It's not that I don't care").
    Litotes ("not bad", "not unlike") has a single negator, and negative
    concord ("didn't do nothing") pairs a negator with an indefinite, not a
    second not/n't -> VER-01."""
    for v in doc:
        negs = [c for c in v.children if c.dep_ == "neg"]
        if len(negs) >= 2:
            lo = min(c.i for c in negs)
            hi = max([c.i for c in negs] + [v.i])
            yield (lo, hi)
            continue
        # not that + negated clause
        if negs and v.lemma_ == "be":
            cc = next((c for c in v.children if c.dep_ == "ccomp"), None)
            if cc is not None and any(k.dep_ == "neg" for k in cc.children) \
                    and any(k.dep_ == "mark" and k.lower_ == "that"
                            for k in cc.children):
                yield (negs[0].i, cc.i)


def build(nlp, client=None):
    dets = [
        RuleRoutingDetector(nlp, ["NEG-01"], _NEG_FORM, _classify_neg01,
                            version="neg01-verb@0.1"),
        _Contractions(),
        _NegDeterminer(),
        PhraseLexiconDetector(nlp, "NEG-06",
                              ["seldom", "rarely", "hardly", "scarcely",
                               "barely", "little"], version="neg06-adv@0.1"),
        PhraseLexiconDetector(nlp, "NEG-07",
                              ["no longer", "not any longer", "no more",
                               "not anymore", "not any more"],
                              version="neg07-nolonger@0.1"),
        _NegEither(),
        _NegFronting(),
        # LLM tier
        LLMReadingDetector(nlp, ["NEG-03"], _NEG_FORM, _NEG03_SYS,
                           client=client, version="neg03-nonassertive@0.1"),
        LLMReadingDetector(nlp, ["NEG-05"],
                           [{"RIGHT_ID": "d",
                             "RIGHT_ATTRS": {"LOWER": "no", "TAG": "DT"}}],
                           _NEG05_SYS, client=client, version="neg05-no@0.1"),
        LLMReadingDetector(nlp, ["NEG-10"],
                           [{"RIGHT_ID": "n",
                             "RIGHT_ATTRS": {"LOWER": {"IN": ["not", "n't"]}}}],
                           _NEG10_SYS, client=client, version="neg10-const@0.1"),
        LLMStandaloneDetector("NEG-08", _NEG08_SYS, client=client,
                              version="neg08-raising@0.1"),
        LLMStandaloneDetector("NEG-11", _NEG11_SYS, client=client,
                              version="neg11-scope@0.1"),
        LLMStandaloneDetector("NEG-12", _NEG12_SYS, client=client,
                              version="neg12-litotes@0.1"),
    ]
    from .common import Scan
    dets.append(Scan("NEG-14", _neg14, version="neg14-double-neg@0.1"))
    return dets
