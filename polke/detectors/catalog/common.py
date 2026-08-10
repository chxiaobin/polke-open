"""Shared pattern fragments and aux-chain helpers for the category detectors."""
from __future__ import annotations
from ...registry import lexicons
from ...schema import Annotation, Span
from ..base import Detector


def shift_annotation(a: Annotation, sent) -> Annotation:
    """Move an annotation produced against ``sent.as_doc()`` back into
    document coordinates."""
    sp = a.span
    a.span = Span(sp.start_char + sent.start_char,
                  sp.end_char + sent.start_char,
                  sp.token_start + sent.start,
                  sp.token_end + sent.start)
    ev = a.evidence or {}
    if isinstance(ev.get("tokens"), list):
        ev["tokens"] = [i + sent.start for i in ev["tokens"]]
    return a


class SentenceScoped(Detector):
    """Base for detectors whose logic assumes doc == ONE sentence (edge
    anchoring such as doc[0] / len(doc)-1, or doc-wide any(...) gates).

    match() renders each sentence as its own doc, runs the subclass's
    match_sent() on it, and shifts the results back into document
    coordinates. The contract suite uses single-sentence docs, which cannot
    expose doc-scope bugs — but production input is 40-utterance documents,
    where a doc-scoped edge check silently disables the detector everywhere
    except the first/last line (this happened: see docs/detector_gaps.md
    and the Spoken BNC dev review)."""

    def match_sent(self, doc, text_id="doc"):
        raise NotImplementedError

    def match(self, doc, text_id="doc"):
        out = []
        for sent in doc.sents:
            for a in self.match_sent(sent.as_doc(), text_id=text_id):
                out.append(shift_annotation(a, sent))
        return out


class Scan(SentenceScoped):
    """Rule detector from a plain generator: fn(doc) yields (lo, hi) token
    index pairs, where doc is ONE SENTENCE (sentence-scoped via
    SentenceScoped, so fn may safely anchor at doc[0] / len(doc)-1). The
    workhorse for token-scan rules where a DependencyMatcher pattern would
    be less readable."""

    def __init__(self, cid, fn, detector_type="rule", version="0.1"):
        self.construct_ids = [cid]
        self._fn = fn
        self.detector_type = detector_type
        self.version = version

    def match_sent(self, doc, text_id="doc"):
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


def last_token(sent):
    """The sentence's last non-space token (utterance-per-line input keeps
    the trailing newline token inside the sentence), or None."""
    for t in reversed(sent):
        if not t.is_space:
            return t
    return None


def ends_with(sent, mark):
    """True iff the sentence's final non-space token is ``mark``."""
    t = last_token(sent)
    return t is not None and t.text == mark


def for_to_advcl(head):
    """The for...to complement frame: an advcl child of ``head`` carrying
    mark 'for', its own overt subject, and a to-infinitive aux — the parse
    en_core_web_sm gives 'We arranged [for him to travel]' and 'It's
    important [for us to leave]' alike (VCP-41 vs ADJ-21 route on the head)."""
    for cl in head.children:
        if cl.dep_ != "advcl":
            continue
        kids = list(cl.children)
        if any(k.dep_ == "mark" and k.lower_ == "for" for k in kids) \
                and any(k.dep_ in ("nsubj", "nsubjpass") for k in kids) \
                and any(k.dep_ == "aux" and k.tag_ == "TO" for k in kids):
            return cl
    return None

# A passive participle (VBN) that has a be/get auxiliary. The anchor is "verb".
PASSIVE_FORM = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ap",
     "RIGHT_ATTRS": {"DEP": "auxpass"}},
]


def aux_chain(verb):
    """The aux + auxpass children of a (participle) verb token."""
    return [k for k in verb.children if k.dep_ in ("aux", "auxpass")]


def has_be_auxpass(verb):
    """True iff the participle's passive auxiliary is a form of *be* (excludes
    get-passive and bare participles). This is the be-passive gate."""
    return any(k.dep_ == "auxpass" and k.lemma_ == "be" for k in verb.children)


def pair_set(name, key="pairs"):
    return {tuple(p) for p in lexicons()[name][key]}


def lemma_set(name, key="lemmas"):
    return set(lexicons()[name][key])
