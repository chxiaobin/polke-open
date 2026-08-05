"""Category SIT — situational-ellipsis subtypes, Part VI.

Patterns are written against what en_core_web_sm ACTUALLY produces for the
elliptical inputs (inspected first, per the integration plan):

- "Doesn't matter." -> does/aux + matter/ROOT, no nsubj (SIT-01: finite,
  subjectless); "Told you." -> Told/VBD/ROOT, no nsubj.
- "Want a coffee?" -> Want/VB/ROOT no nsubj; "Seen John?" -> Seen/VBD(!);
  "Been waiting long?" -> waiting/ROOT with aux Been (SIT-02: subjectless
  AND operatorless, interrogative). The "?" requirement is the imperative
  exclude: "Look at this picture." is verb-initial and subjectless too but
  ends in ".".
- "You seen John?" -> seen/VBD/ROOT WITH nsubj, no aux (SIT-03: subject
  present, operator dropped; the verb form is a participle-preterite from
  the vernacular_forms lexicon, which is why SIT-03 xrefs VER-03).

SIT-04 (other situational fragments) is hybrid tier -> Phase 4.
"""
from __future__ import annotations
from ...registry import lexicons
from .common import Scan

_FINITE_TAGS = {"VBZ", "VBD", "VBP"}


def _root_and_subj(sent):
    root = sent.root
    subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
    return root, subj


def _sit01(doc):
    """Initial subject ellipsis: finite verb group, no subject, declarative."""
    for sent in doc.sents:
        if sent[-1].text == "?":
            continue                       # interrogative -> SIT-02 territory
        root, subj = _root_and_subj(sent)
        if root.pos_ not in ("VERB", "AUX") or subj:
            continue
        finite = root.tag_ in ("VBZ", "VBD") or any(
            c.dep_ in ("aux", "auxpass") and c.tag_ in _FINITE_TAGS
            for c in root.children)
        if not finite:
            continue                       # bare VB with no finite aux = imperative
        first = next((t for t in sent if not t.is_punct), None)
        if first is None or (first.i != root.i and first.head.i != root.i
                             and first.head.head.i != root.i):
            continue                       # verb group must open the unit
        yield (first.i, root.i)


def _sit02(doc):
    """Subject + operator ellipsis: verb-initial, subjectless, interrogative.
    The '?' is the imperative exclude ("Look at this picture.")."""
    for sent in doc.sents:
        if sent[-1].text != "?":
            continue
        root, subj = _root_and_subj(sent)
        if root.pos_ not in ("VERB", "AUX") or subj:
            continue
        # no finite do/have/be operator: "Do you want...?" has one (and a
        # subject); "Been waiting long?" has nonfinite Been which is fine
        if any(c.dep_ == "aux" and c.tag_ in _FINITE_TAGS and c.lower_ != "been"
               for c in root.children):
            continue
        first = next((t for t in sent if not t.is_punct), None)
        if first is None:
            continue
        if first.i != root.i and first.head.i != root.i:
            continue                       # unit opens with the verb group
        yield (first.i, root.i)


def _sit03(doc):
    """Operator-only (medial) ellipsis: overt subject + bare participle
    preterite ("You seen John?"). The form set is the shared
    leveled_participle_preterites lexicon (aux-drop shades into VER)."""
    forms = set(lexicons()["vernacular_forms"]["leveled_participle_preterites"])
    for sent in doc.sents:
        root = sent.root
        if root.lower_ not in forms or root.tag_ not in ("VBD", "VBN"):
            continue
        subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
        if not subj:
            continue                       # no subject -> SIT-02
        if any(c.dep_ in ("aux", "auxpass") for c in root.children):
            continue                       # "Have you seen John?" is standard
        yield (subj[0].i, root.i)


_SIT04_SYS = (
    "The utterance is a VERBLESS fragment. Decide whether it is SIT-04 — a "
    "situational fragment where a copula, determiner or preposition is "
    "recoverable from the speech situation:\n"
    "Examples: \"Ready?\" (= Are you ready?) | \"Good film, that.\" | "
    "\"Pity.\" (= It's a pity.) | \"About six?\" (= At about six?)\n\n"
    "NOT SIT-04 (return NONE):\n"
    "- Freestanding response tokens / interjections / greetings / politeness "
    "formulae (INS-*): \"Yeah.\" | \"Thanks.\" | \"Hi.\" | \"Wow!\"\n"
    "- Bare vocatives: \"John?\"\n"
    "- Verbless exclamative judgements (EXC-03): \"Nice one!\"\n"
    'Return only JSON: {"construct_id": "SIT-04"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')


class _SituationalFragment:
    """SIT-04 (hybrid): the rule proposes sentences with NO verbal token at
    all (subject/operator ellipses SIT-01..03 and full clauses never
    qualify); the LLM separates true situational fragments from freestanding
    inserts (INS-*) and verbless exclamatives (EXC-03)."""
    detector_type = "hybrid_rule_llm"
    version = "sit04-fragment@0.1"
    construct_ids = ["SIT-04"]

    def __init__(self, client=None):
        from ..llm import DummyClient
        self._client = client or DummyClient()

    def llm_tasks(self, doc, text_id="doc"):
        from ...schema import Span
        tasks = []
        for sent in doc.sents:
            if any(t.pos_ in ("VERB", "AUX") for t in sent):
                continue
            if not any(not t.is_punct for t in sent):
                continue
            sp = Span(sent.start_char, sent.end_char, sent.start, sent.end - 1)
            tasks.append(lambda user=sent.text, sp=sp:
                         self._judge(user, sp, text_id))
        return tasks

    def _judge(self, user, sp, text_id):
        from ...schema import Annotation
        res = self._client.classify(_SIT04_SYS, user, ["SIT-04", "NONE"])
        if res.get("construct_id") != "SIT-04":
            return None
        return Annotation(
            text_id=text_id, construct_id="SIT-04", span=sp,
            detector_type=self.detector_type, detector_version=self.version,
            confidence=float(res.get("confidence", 0.0)),
            model=getattr(self._client, "model", None),
            evidence={"rationale": res.get("rationale", "")})

    def match(self, doc, text_id="doc"):
        anns = (t() for t in self.llm_tasks(doc, text_id=text_id))
        return [a for a in anns if a is not None]


def build(nlp, client=None):
    return [
        Scan("SIT-01", _sit01, version="sit01-subj-ellipsis@0.1"),
        Scan("SIT-02", _sit02, version="sit02-subj-op-ellipsis@0.1"),
        Scan("SIT-03", _sit03, version="sit03-op-ellipsis@0.1"),
        _SituationalFragment(client=client),
    ]
