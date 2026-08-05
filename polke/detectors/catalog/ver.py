"""Category VER — vernacular grammar (flagged stratum), Part VI.

Phase 2 implements the three lexicon-tier members over the tokenisations
en_core_web_sm actually produces (verified: ain't -> ai + n't, gonna ->
gon + na, gotta -> got + ta, but wanna and innit stay whole):

- VER-02 ain't: fixed token sequences.
- VER-05 leveled verb forms: regularized preterites (knowed, catched, ...)
  fire on sight; participle-as-preterite (He seen it) fires only when the
  parser tags the form VBD with no auxiliary — "He has seen it." is VBN
  with aux and stays out.
- VER-06 reduced semi-modals gonna/wanna/gotta as transcribed; the full
  forms "going to / got to / have to" never produce these token sequences.

VER-01/03/04/07 are rule tier -> Phase 3.
"""
from __future__ import annotations
from ...registry import lexicons
from ...schema import Annotation, Span
from ..base import Detector
from ..spoken import TokenSequenceDetector


class _LeveledVerbForms(Detector):
    detector_type = "lexicon"
    version = "ver05-leveled@0.1"
    construct_ids = ["VER-05"]

    def __init__(self):
        lx = lexicons()["vernacular_forms"]
        self._regularized = set(lx["leveled_regularized"])
        self._participles = set(lx["leveled_participle_preterites"])

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            hit = False
            if t.lower_ in self._regularized and t.pos_ in ("VERB", "AUX", "NOUN"):
                hit = True  # sm sometimes noun-tags unknown forms like "knowed"
            elif t.lower_ in self._participles and t.tag_ == "VBD" \
                    and not any(k.dep_ in ("aux", "auxpass") for k in t.children):
                hit = True  # participle used as preterite: "He seen it."
            if hit:
                out.append(Annotation(
                    text_id=text_id, construct_id="VER-05",
                    span=Span(t.idx, t.idx + len(t.text), t.i, t.i),
                    detector_type=self.detector_type,
                    detector_version=self.version, confidence=1.0,
                    evidence={"tokens": [t.i], "matched": t.text}))
        return out


# --- rule tier (Phase 3) ---------------------------------------------------

_NEG_INDEFINITES = {"nothing", "nobody", "nowhere", "none", "no one"}
_PLURAL_PRON = {"we", "you", "they", "them"}
_SG3_PRON = {"he", "she", "it"}


def _ver01(doc):
    """VER-01 negative concord: a verbal negator (n't/not/never) plus a
    negative INDEFINITE (nothing/nobody/nowhere/none) in the same sentence.
    Standard non-assertive items (any/anyone) and stacked verbal negation
    ("can't not go" -> NEG-14) have no negative indefinite."""
    for sent in doc.sents:
        negs = [t for t in sent
                if t.dep_ == "neg" or t.lower_ == "never"]
        indefs = [t for t in sent if t.lower_ in _NEG_INDEFINITES
                  and t.dep_ != "neg"]
        if negs and indefs:
            lo = min(t.i for t in negs + indefs)
            hi = max(t.i for t in negs + indefs)
            yield (lo, hi)


def nonstandard_concord(v):
    """VER-03 core predicate on a finite verb token. Shared with CLS-08: a
    True here means the clause shows systematic vernacular concord and must
    NOT be scored against the standard-agreement construct (see cls.py)."""
    subj = next((c for c in v.children if c.dep_ in ("nsubj", "nsubjpass")),
                None)
    if subj is None:
        return False
    plural_subj = (subj.tag_ == "NNS"
                   or (subj.tag_ == "PRP" and subj.lower_ in _PLURAL_PRON))
    sg3_subj = (subj.tag_ in ("NN", "NNP")
                or (subj.tag_ == "PRP" and subj.lower_ in _SG3_PRON))
    # we/you/they/them + was ; plural NP + is/was
    if v.lower_ in ("was", "is") and plural_subj and v.lower_ != "were":
        if v.lower_ == "was" or v.tag_ == "VBZ":
            return True
    # 3sg subject + don't (do-support without -s)
    if any(c.dep_ == "aux" and c.lower_ == "do" and c.tag_ == "VBP"
           for c in v.children) \
            and any(c.dep_ == "neg" for c in v.children) and sg3_subj:
        return True
    return False


def _ver03(doc):
    for v in doc:
        if v.pos_ not in ("VERB", "AUX"):
            continue
        if nonstandard_concord(v):
            subj = next(c for c in v.children
                        if c.dep_ in ("nsubj", "nsubjpass"))
            aux = [c for c in v.children if c.dep_ in ("aux", "neg")]
            lo = min([subj.i, v.i] + [a.i for a in aux])
            hi = max([subj.i, v.i] + [a.i for a in aux])
            yield (lo, hi)


def _ver04(doc):
    """VER-04 demonstrative them: 'them' + plural noun as one NP. In the
    ditransitive near-miss "I gave them books" the parser makes 'them' a
    dative argument, which is the exclude."""
    for t in doc:
        if t.lower_ != "them" or t.dep_ == "dative":
            continue
        nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
        if nxt is not None and nxt.tag_ == "NNS":
            yield (t.i, t.i + 1)


def _ver07(doc):
    """VER-07 double comparative/superlative: more/most + an already
    inflected form ("more better", "the most brightest")."""
    for t in doc:
        if t.lower_ not in ("more", "most"):
            continue
        nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
        if nxt is not None and nxt.tag_ in ("JJR", "JJS", "RBR", "RBS"):
            yield (t.i, t.i + 1)


def build(nlp, client=None):
    from .common import Scan
    lx = lexicons()["vernacular_forms"]
    reduced = lexicons()["reduced_forms"]
    return [
        TokenSequenceDetector("VER-02", lx["aint_sequences"],
                              version="ver02-aint@0.1"),
        _LeveledVerbForms(),
        TokenSequenceDetector("VER-06", reduced["semi_modal_sequences"],
                              version="ver06-reduced@0.1"),
        Scan("VER-01", _ver01, version="ver01-neg-concord@0.1"),
        Scan("VER-03", _ver03, version="ver03-concord@0.1"),
        Scan("VER-04", _ver04, version="ver04-them@0.1"),
        Scan("VER-07", _ver07, version="ver07-double-comp@0.1"),
    ]
