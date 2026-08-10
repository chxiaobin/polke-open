"""Category QIN — interactional questions & tags, Part VI.

Lexicon tier: QIN-05 preface questions (freestanding formula + "?") and
QIN-06 invariant tags (host clause + comma + tag + "?"; freestanding "Eh?"
is INS-04, reversed-polarity operator tags are QUE-11 and never lexicon
items). Rule tier: QIN-02 statement (copy) tags — the tail after the final
comma is a bare personal pronoun or pronoun + operator echo ("he is",
"me"); the pronoun+operator shape is what routes against the full-NP tails
of FOC-17 ("She's clever, your daughter." has PRP$ + noun), and the "?"
exclude routes against QUE-11/12 operator tags. QIN-03 exclamation tags:
negated-operator inversion ending in "!" not "?" ("Wasn't it brilliant!").
QIN-01/04 are LLM tier -> Phase 4.
"""
from __future__ import annotations
from ...registry import lexicons
from ..llm import LLMStandaloneDetector
from ..spoken import FinalTagDetector, FreestandingUnitDetector
from .common import Scan, ends_with

_QIN01_SYS = (
    "Decide whether the utterance is a DECLARATIVE QUESTION (QIN-01): full "
    "statement word order (subject before finite verb, no operator "
    "inversion, no wh-fronting) used as a question — the question mark / "
    "rising intonation is the only interrogative signal.\n"
    "Positive examples: \"You're leaving?\" | \"He actually said that?\"\n\n"
    "NOT QIN-01 (return NONE):\n"
    "- QUE-01 operator-inverted yes/no question: \"Are you leaving?\"\n"
    "- Wh-questions: \"What do you want?\"\n"
    "- Tag questions: \"You're coming, aren't you?\" (the host is a "
    "statement but the tag carries the question)\n"
    "- Elliptical fragments without a subject+verb statement: \"Ready?\" | "
    "\"Which drawer?\"\n"
    'Return only JSON: {"construct_id": "QIN-01"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')

_QIN04_SYS = (
    "You see a dialogue: previous utterance(s), then the current utterance. "
    "Decide whether the current utterance is a FOLLOW-UP / two-step question "
    "(QIN-04): a question that queries or narrows something in the PREVIOUS "
    "speaker's utterance, typically an elliptical wh-fragment echoing its "
    "material.\n"
    "Positive examples: [previous] \"In the drawer.\" [current] \"Which "
    "drawer?\" | [previous] \"We could meet at the pub.\" [current] "
    "\"Which pub?\"\n\n"
    "NOT QIN-04 (return NONE):\n"
    "- QUE-04 initiating wh-questions that do not build on the previous "
    "utterance: \"What do you want?\"\n"
    "- Non-questions in response position: \"Thanks.\"\n"
    "- If no previous utterance is supplied and the question stands alone.\n"
    'Return only JSON: {"construct_id": "QIN-04"|"NONE", '
    '"confidence": 0.0-1.0, "rationale": "..."}')


def _ends_q(sent):
    return ends_with(sent, "?")

_OPERATOR_TAGS = {"VBZ", "VBP", "VBD", "MD"}


def _qin02(doc):
    for sent in doc.sents:
        if ends_with(sent, "?"):
            continue                        # operator tags with "?" -> QUE-11/12
        commas = [t for t in sent if t.text == ","]
        if not commas:
            # spoken path (no commas in transcripts): utterance-final
            # pronoun+operator echoing an earlier operator with its own
            # subject — "they're beautiful they are".
            toks = [t for t in sent if not t.is_punct and not t.is_space]
            if len(toks) >= 4:
                t1, t2 = toks[-2], toks[-1]
                op_ok = t2.tag_ == "MD" or (t2.tag_ in _OPERATOR_TAGS
                                            and t2.lemma_ in ("be", "do", "have"))
                if t1.tag_ == "PRP" and op_ok \
                        and any(t.lemma_ == t2.lemma_ and t.i < t1.i
                                for t in toks[:-2]) \
                        and any(t.dep_ in ("nsubj", "nsubjpass") and t.i < t1.i
                                for t in toks[:-2]):
                    yield (t1.i, t2.i)
            continue
        last = commas[-1]
        tail = [t for t in doc[last.i + 1:sent.end] if not t.is_punct]
        head_part = [t for t in doc[sent.start:last.i] if not t.is_punct]
        if len(head_part) < 2 or not tail or len(tail) > 2:
            continue
        if tail[0].tag_ != "PRP":
            continue                        # full-NP tail -> FOC-17
        if len(tail) == 2 and tail[1].tag_ not in _OPERATOR_TAGS:
            continue                        # pronoun + operator echo only
        # the host clause needs its own subject (copy, not dislocation)
        if not any(t.dep_ in ("nsubj", "nsubjpass") for t in head_part):
            continue
        yield (tail[0].i, tail[-1].i)


def _qin03(doc):
    for sent in doc.sents:
        if not ends_with(sent, "!"):
            continue
        first = next((t for t in sent if not t.is_punct), None)
        if first is None or first.pos_ not in ("AUX", "VERB") \
                or first.tag_ not in _OPERATOR_TAGS:
            continue
        if first.i + 2 >= sent.end:
            continue
        if doc[first.i + 1].lower_ not in ("n't", "not"):
            continue
        if doc[first.i + 2].pos_ not in ("PRON", "PROPN", "NOUN", "DET"):
            continue
        yield (first.i, first.i + 2)


def build(nlp, client=None):
    lx = lexicons()
    return [
        FreestandingUnitDetector(nlp, "QIN-05",
                                 lx["preface_questions"]["entries"],
                                 question=True, version="qin05-preface@0.1"),
        FinalTagDetector(nlp, "QIN-06", lx["invariant_tags"]["entries"],
                         version="qin06-invtag@0.1"),
        Scan("QIN-02", _qin02, version="qin02-copytag@0.1"),
        Scan("QIN-03", _qin03, version="qin03-excltag@0.1"),
        LLMStandaloneDetector("QIN-01", _QIN01_SYS, client=client,
                              gate=_ends_q, version="qin01-declq@0.1"),
        LLMStandaloneDetector("QIN-04", _QIN04_SYS, client=client,
                              gate=_ends_q, wants_context=True,
                              version="qin04-followup@0.1"),
    ]
