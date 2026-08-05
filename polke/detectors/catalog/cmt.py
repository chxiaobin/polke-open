"""Category CMT — comment clauses & parentheticals, Part VI.

CMT-02 (lexicon): interactive parentheticals, medial/final, comma-preceded —
initial "You know, ..." is DMG-03 and integrated "You know the answer" has
no comma. CMT-03 (rule): as-comment clauses — an advcl with mark *as* whose
verb is a discourse/comment lemma (know, say, mention, happen ...) and whose
subject is a personal pronoun; manner "as I showed you" (show) and temporal
"as I was leaving" (leave) fail the lemma set, and "works as a nurse" is a
prep, not a clause. CMT-04 (rule): medial/final reporting clause including
inversion — a reporting verb PRECEDED by a closing quote in the same
sentence ("'Fine,' he said."); initial-frame "He said, 'Fine.'" has the
quote after the verb (REP-01). CMT-01 is hybrid tier -> Phase 4.
"""
from __future__ import annotations
from ...registry import lexicons
from ..llm import LLMReadingDetector
from ..spoken import MedialFinalParentheticalDetector
from .common import Scan, lemma_set

# CMT-01 (hybrid): one FORM — a first-person epistemic verb — two readings.
# The rule proposes every "I + think/guess/reckon/..." span; the LLM routes
# between the parenthetical comment (CMT-01) and the true main clause with a
# that-complement (NCL-01). ncl.py registers NCL-01's own detector later
# (alphabetical order), so NCL-01's contract stays with its original owner.
_CMT01_FORM = [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": [
        "think", "guess", "reckon", "suppose", "imagine", "suspect",
        "bet", "expect", "gather", "figure"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "s",
     "RIGHT_ATTRS": {"DEP": "nsubj", "LOWER": "i"}},
]

_CMT01_SYS = (
    "The marked span [[ ]] is a first-person epistemic verb phrase (I think / "
    "I guess / I reckon ...). Decide which construct it realises:\n\n"
    "CMT-01 — first-person epistemic PARENTHETICAL: the I-verb unit is a "
    "comment hedging the host clause, in medial or final position, or "
    "clause-initial WITHOUT 'that'. The host clause stands on its own.\n"
    "Examples: \"It's fine, I think.\" | \"He's, I guess, about forty.\" | "
    "\"I reckon it'll rain.\"\n\n"
    "NCL-01 — MAIN CLAUSE + that-complement: the I-verb is the matrix "
    "predicate and takes an explicit 'that'-clause object.\n"
    "Examples: \"I think that it's fine.\" | \"I guess that we should leave "
    "now.\"\n\n"
    "Decision rule: an overt 'that' right after the verb -> NCL-01; comma-"
    "delimited medial/final placement -> CMT-01; initial without 'that' -> "
    "CMT-01.\n"
    'Return only JSON: {"construct_id": "CMT-01"|"NCL-01"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')

_COMMENT_LEMMAS = {"know", "say", "mention", "note", "happen", "see",
                   "suggest", "expect", "indicate", "understand", "recall",
                   "remember", "gather", "hear", "fear", "hope", "imagine",
                   "explain", "promise", "admit", "report", "turn"}
_COMMENT_SUBJ = {"you", "i", "we", "it"}

_EXTRA_SPEECH = {"reply", "ask", "shout", "whisper", "add", "murmur",
                 "mutter", "answer", "cry", "call", "exclaim", "insist",
                 "continue", "repeat", "snap", "sigh", "laugh", "moan",
                 "demand", "warn", "beg", "yell", "growl", "grumble"}


def _cmt03(doc):
    for t in doc:
        if t.dep_ != "advcl" or t.lemma_.lower() not in _COMMENT_LEMMAS:
            continue
        mark = next((c for c in t.children
                     if c.dep_ == "mark" and c.lower_ == "as"), None)
        if mark is None:
            continue
        subj = next((c for c in t.children if c.dep_ == "nsubj"), None)
        if subj is not None and subj.lower_ not in _COMMENT_SUBJ:
            continue
        # parenthetical: the clause is comma-delimited on at least one side
        end = max([t.i] + [c.i for c in t.subtree])
        after = doc[end + 1] if end + 1 < len(doc) else None
        before = doc[mark.i - 1] if mark.i > 0 else None
        sent = t.sent
        delimited = ((after is not None and after.text == ",")
                     or (before is not None and before.text == ",")
                     or mark.i == sent.start)
        if not delimited:
            continue
        yield (mark.i, end)


def _cmt04(doc):
    reporting = lemma_set("reporting_verbs") | _EXTRA_SPEECH
    for sent in doc.sents:
        closers = [t.i for t in sent if t.tag_ == "''"]
        if not closers:
            continue
        for t in sent:
            if t.pos_ != "VERB" or t.lemma_.lower() not in reporting:
                continue
            if not any(i < t.i for i in closers):
                continue                    # quote must PRECEDE the verb
            subj = next((c for c in t.children if c.dep_ == "nsubj"), None)
            lo = min(t.i, subj.i) if subj is not None else t.i
            hi = max(t.i, subj.i) if subj is not None else t.i
            yield (lo, hi)


def build(nlp, client=None):
    return [
        MedialFinalParentheticalDetector(
            nlp, "CMT-02", lexicons()["comment_parentheticals"]["entries"],
            version="cmt02-parenth@0.1"),
        Scan("CMT-03", _cmt03, version="cmt03-as-comment@0.1"),
        Scan("CMT-04", _cmt04, version="cmt04-report-clause@0.1"),
        LLMReadingDetector(nlp, ["CMT-01", "NCL-01"], _CMT01_FORM, _CMT01_SYS,
                           client=client, version="cmt01-epistemic@0.1"),
    ]
