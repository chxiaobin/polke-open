"""Category ASC - adverbial subordinate clauses.

Most siblings are identified by the conjunction that introduces the adverbial
clause -> surface ``PhraseLexiconDetector`` on the connective, gated by
``_subord`` (the connective really heads a subordinate clause: a ``mark`` or a
``WRB`` advmod whose head is an advcl/ccomp/acl/relcl). This keeps prepositional
uses ("before dinner"), interrogatives ("When did you go?") and clause-final
linkers out.

The genuinely ambiguous connectives are handled explicitly:
- ``as`` is simultaneous-time (ASC-02) OR reason (ASC-08); both are guarded to
  reject the "so as", "as soon as", "as if", "as ... as" frames.
- ``since`` is time (ASC-04) OR reason (ASC-08) - both fire (see NOTES).
- ``while``/``as`` appear in time, contrast and reduced-clause families.
Correlative/rule shapes (ASC-13 so...that result, ASC-24 the...the proportion,
ASC-27 perfect participle) are small custom rules.

Prepositional concessives (ASC-17 despite/in spite of) and inversion triggers
(ASC-07 no sooner) are NOT clause-gated - they are distinctive fixed phrases.

LLM tier (skipped offline): ASC-06 present-for-future in time clauses, ASC-11
non-finite purpose, ASC-19 concessive inversion, ASC-21 manner as/the way,
ASC-22 unreal as if/as though, ASC-25 -ing / ASC-26 -ed / ASC-28 with-absolute /
ASC-29 verbless reduced clauses.
"""
from __future__ import annotations
from ..lexical import PhraseLexiconDetector
from ..llm import LLMReadingDetector
from ..base import Detector
from ...schema import Annotation, Span

_CLAUSE = {"advcl", "ccomp", "acl", "relcl", "pcomp"}
# tokens that turn a bare "as" into a different connective frame
_AS_VETO_PREV = {"so", "soon", "long", "far", "as", "just"}
_AS_VETO_NEXT = {"if", "though", "long", "far"}


def _subord(doc, start, end):
    """True if some token in [start,end) subordinates a clause."""
    for i in range(start, end):
        t = doc[i]
        if t.dep_ == "mark" and t.head.dep_ in _CLAUSE:
            return True
        if t.tag_ == "WRB" and t.dep_ == "advmod" and t.head.dep_ in _CLAUSE:
            return True
    return False


def _prev(doc, start):
    return doc[start - 1].lower_ if start > 0 else ""


def _next(doc, end):
    return doc[end].lower_ if end < len(doc) else ""


def _as_ok(doc, start, end):
    return (_prev(doc, start) not in _AS_VETO_PREV
            and _next(doc, end) not in _AS_VETO_NEXT)


# --- guarded gates ---------------------------------------------------------- #
def _gate_time_as(doc, start, end):        # ASC-02 while / whilst / as
    if not _subord(doc, start, end):
        return False
    txt = doc[start:end].text.lower()
    if txt in ("while", "whilst"):
        return True
    if txt == "as":
        return _as_ok(doc, start, end)
    return False


def _gate_reason(doc, start, end):         # ASC-08 because / since / as
    if not _subord(doc, start, end):
        return False
    txt = doc[start:end].text.lower()
    if txt in ("because", "since"):
        return True
    if txt == "as":
        return _as_ok(doc, start, end)
    return False


def _gate_concess_though(doc, start, end):  # ASC-15 although / though / even though
    if not _subord(doc, start, end):
        return False
    if doc[start:end].text.lower() == "though" and _prev(doc, start) == "as":
        return False   # "as though" -> ASC-22
    return True


def _gate_however(doc, start, end):        # ASC-18 however + adj/adv, wh-ever, no matter
    txt = doc[start:end].text.lower()
    if txt == "however":
        nxt = doc[end] if end < len(doc) else None
        return nxt is not None and nxt.pos_ in ("ADJ", "ADV")
    return True


# --- custom rule detectors -------------------------------------------------- #
class _ResultSoThat(Detector):
    """ASC-13: so + adj/adv ... that  (also such + N ... that) result clause."""
    detector_type = "lexicon"

    def __init__(self):
        self.construct_ids = ["ASC-13"]
        self.version = "asc13-sothat@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ == "so" and t.dep_ == "advmod" and t.head.pos_ in ("ADJ", "ADV"):
                anchor = t.head
            elif t.lower_ == "such" and t.head.pos_ in ("NOUN", "PROPN"):
                anchor = t.head
            else:
                continue
            that = next((u for u in doc if u.lower_ == "that" and u.dep_ == "mark"
                         and u.i > t.i and (u.head == anchor
                                            or u.head.head == anchor
                                            or u.head.head == anchor.head)), None)
            if that is None:
                continue
            lo, hi = t.i, that.i
            span = doc[lo:hi + 1]
            out.append(Annotation(
                text_id=text_id, construct_id="ASC-13",
                span=Span(span.start_char, span.end_char, lo, hi),
                detector_type=self.detector_type, detector_version=self.version,
                confidence=1.0,
                evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text}))
        return out


class _Proportion(Detector):
    """ASC-24: the + comparative ... the + comparative."""
    detector_type = "rule"

    def __init__(self):
        self.construct_ids = ["ASC-24"]
        self.version = "asc24-thethe@0.1"

    def match(self, doc, text_id="doc"):
        hits = [t for t in doc if t.lower_ == "the"
                and (t.head.tag_ in ("JJR", "RBR")
                     or t.head.lower_ in ("more", "less", "fewer"))]
        if len(hits) < 2:
            return []
        lo = hits[0].i
        hi = max(h.head.i for h in hits[:2])
        span = doc[lo:hi + 1]
        return [Annotation(
            text_id=text_id, construct_id="ASC-24",
            span=Span(span.start_char, span.end_char, lo, hi),
            detector_type=self.detector_type, detector_version=self.version,
            confidence=1.0,
            evidence={"tokens": list(range(lo, hi + 1)), "matched": span.text})]


class _PerfectParticiple(Detector):
    """ASC-27: Having + past participle adverbial clause."""
    detector_type = "rule"

    def __init__(self):
        self.construct_ids = ["ASC-27"]
        self.version = "asc27-having@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if (t.lower_ == "having" and t.tag_ == "VBG"
                    and t.dep_ == "aux" and t.head.tag_ == "VBN"
                    and t.head.dep_ in ("advcl", "ccomp")):
                lo, hi = min(t.i, t.head.i), max(t.i, t.head.i)
                span = doc[lo:hi + 1]
                out.append(Annotation(
                    text_id=text_id, construct_id="ASC-27",
                    span=Span(span.start_char, span.end_char, lo, hi),
                    detector_type=self.detector_type,
                    detector_version=self.version, confidence=1.0,
                    evidence={"tokens": list(range(lo, hi + 1)),
                              "matched": span.text}))
        return out


def build(nlp, client=None):
    dets = []

    def lex(cid, phrases, gate=None, version=None):
        return PhraseLexiconDetector(nlp, cid, phrases, gate=gate,
                                     version=version or f"{cid.lower()}@0.1")

    # -- Time -----------------------------------------------------------------
    dets.append(lex("ASC-01", ["when", "whenever"], gate=_subord))
    dets.append(lex("ASC-02", ["while", "whilst", "as"], gate=_gate_time_as))
    dets.append(lex("ASC-03", ["before", "after", "until", "till"], gate=_subord))
    dets.append(lex("ASC-04", ["since"], gate=_subord))
    dets.append(lex("ASC-05",
                    ["as soon as", "once", "the moment", "by the time",
                     "every time", "each time", "the minute", "the instant",
                     "directly", "immediately"], gate=_subord))
    dets.append(lex("ASC-07", ["no sooner"]))   # inversion trigger, no gate

    # -- Reason / purpose / result -------------------------------------------
    dets.append(lex("ASC-08", ["because", "since", "as"], gate=_gate_reason))
    dets.append(lex("ASC-09",
                    ["now that", "seeing that", "seeing as", "in that",
                     "given that", "insofar as", "inasmuch as"], gate=_subord))
    dets.append(lex("ASC-10", ["so that", "in order that"], gate=_subord))
    dets.append(lex("ASC-12",
                    ["so as not to", "in order not to", "for fear that",
                     "for fear", "lest"]))       # distinctive fixed phrases
    dets.append(_ResultSoThat())                 # ASC-13
    dets.append(lex("ASC-14", ["with the result that", "with the result"]))

    # -- Concession / contrast ------------------------------------------------
    dets.append(lex("ASC-15", ["although", "though", "even though"],
                    gate=_gate_concess_though))
    dets.append(lex("ASC-16", ["whereas", "whilst", "while"], gate=_subord))
    dets.append(lex("ASC-17",
                    ["despite", "in spite of", "notwithstanding"]))
    dets.append(lex("ASC-18",
                    ["however", "whatever", "whoever", "wherever", "whichever",
                     "no matter how", "no matter what", "no matter who",
                     "no matter when", "no matter where", "no matter which"],
                    gate=_gate_however))
    dets.append(lex("ASC-20", ["even if"], gate=_subord))

    # -- Manner / place / proportion -----------------------------------------
    dets.append(lex("ASC-23", ["where", "wherever", "everywhere", "anywhere"],
                    gate=_subord))
    dets.append(_Proportion())                   # ASC-24

    # -- Reduced adverbial clauses -------------------------------------------
    dets.append(_PerfectParticiple())            # ASC-27

    # -- LLM tier (hybrid_rule_llm): reading decided by meaning; skip offline --
    def _reading(cid, form, desc):
        sys = (f"Decide whether the marked clause is: {desc}. "
               f'Return JSON {{"construct_id":"{cid}","confidence":0..1,'
               f'"rationale":"..."}} or construct_id "NONE".')
        return LLMReadingDetector(nlp, [cid], form, sys, client=client,
                                  version=f"{cid.lower()}-llm@0.1")

    _advcl = [{"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "advcl"}}]
    _advcl_mark = _advcl + [
        {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "m",
         "RIGHT_ATTRS": {"DEP": "mark"}}]

    dets.append(_reading(
        "ASC-06", _advcl_mark,
        "a time clause using the present tense (no 'will') to refer to the "
        "future, e.g. 'When he comes, we'll eat'"))
    dets.append(_reading(
        "ASC-11",
        _advcl + [{"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "to",
                   "RIGHT_ATTRS": {"TAG": "TO"}}],
        "a non-finite purpose clause: to / in order to / so as to / for + -ing"))
    dets.append(_reading(
        "ASC-19",
        [{"RIGHT_ID": "a", "RIGHT_ATTRS":
          {"DEP": "advcl", "TAG": {"IN": ["JJ", "RB", "NN"]}}}],
        "concessive inversion: Adj/N + as/though + subject + verb, e.g. "
        "'Tired as I was, ...'"))
    dets.append(_reading(
        "ASC-21",
        _advcl + [{"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "m",
                   "RIGHT_ATTRS": {"DEP": "mark", "LOWER": "as"}}],
        "a manner clause: (just) as / the way someone does something"))
    dets.append(_reading(
        "ASC-22",
        _advcl + [{"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "m",
                   "RIGHT_ATTRS": {"DEP": "mark", "LOWER": {"IN": ["if", "though"]}}}],
        "an unreal manner clause introduced by 'as if' / 'as though'"))
    dets.append(_reading(
        "ASC-25", [{"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "advcl", "TAG": "VBG"}}],
        "a reduced -ing participle adverbial clause, e.g. 'While walking home, ...'"))
    dets.append(_reading(
        "ASC-26", [{"RIGHT_ID": "v", "RIGHT_ATTRS": {"DEP": "advcl", "TAG": "VBN"}}],
        "a reduced -ed participle adverbial clause, e.g. 'If asked, ...'"))
    dets.append(_reading(
        "ASC-28",
        _advcl + [{"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "m",
                   "RIGHT_ATTRS": {"DEP": "mark", "LOWER": "with"}}],
        "an absolute 'with + NP + complement' clause, e.g. 'With the door open, ...'"))
    dets.append(_reading(
        "ASC-29",
        [{"RIGHT_ID": "m", "RIGHT_ATTRS": {"TAG": "WRB"}}],
        "a verbless reduced clause, e.g. 'When in doubt, ask' / 'If necessary, call'"))

    return dets
