"""Category VOC — vocatives, Part VI.

Phase 2 implements the two closed-class subtypes: VOC-02 kinship/endearment/
familiarizer and VOC-03 impersonal/plural/honorific vocatives, both as
comma-peripheral windows over their lexicons. Referential uses never match
because a determiner/possessive ("My mum", "The folks") breaks the comma
adjacency, and subject uses ("You guys are late") lack the following comma.
VOC-01 (open-class names/titles) is rule tier -> Phase 3.
"""
from __future__ import annotations
from ...registry import lexicons
from ..spoken import VocativeDetector
from .common import Scan


_ARG_DEPS = {"nsubj", "nsubjpass", "dobj", "pobj", "attr", "poss", "dative",
             "conj", "compound", "appos", "oprd"}


_GREETINGS = {"thanks", "thank", "hi", "hello", "hey", "bye", "cheers",
              "sorry", "please", "morning", "night", "welcome"}


def _address_context(doc, sent, i, e):
    """Something outside the name window that shows someone is being ADDRESSED
    (a clause, a second person, a greeting). A bare-name utterance with only
    fillers ("no no Milltown", "er Christmas Eve") is a mention/answer."""
    for t in sent:
        if i <= t.i < e or t.is_punct or t.is_space:
            continue
        if t.pos_ in ("VERB", "AUX") or t.tag_ in ("PRP", "PRP$") \
                or t.lower_ in _GREETINGS:
            return True
    return False


def _voc01(doc):
    """VOC-01 (rule tier): comma-peripheral proper-name/title NP. Same
    positional grammar as the lexicon vocatives, but over PROPN windows
    ("John, are you coming?", "Thanks, Dr Smith."); subject/object names
    ("John is coming", "I thanked Dr Smith") are never comma-delimited."""
    for sent in doc.sents:
        start = next((t.i for t in sent if not t.is_punct), sent.start)
        i = sent.start
        while i < sent.end:
            if doc[i].tag_ not in ("NNP", "NNPS"):
                i += 1
                continue
            j = i
            while j + 1 < sent.end and doc[j + 1].tag_ in ("NNP", "NNPS"):
                j += 1
            e = j + 1                      # exclusive end of PROPN window
            initial = i == start and e < sent.end and doc[e].text == ","
            after_comma = i > sent.start and doc[i - 1].text == ","
            final = after_comma and all(t.is_punct for t in doc[e:sent.end])
            medial = after_comma and e < sent.end and doc[e].text == ","
            # Spoken paths — transcripts have no commas. A name at an
            # utterance edge is vocative when it fills no argument slot:
            # "thanks Sam" / "Sam are you coming" fire, "Sam is coming"
            # (nsubj) and "I thanked Sam" (dobj) do not.
            # A "subject" name is still vocative when its verb has ANOTHER
            # subject of its own ("Sam are you coming" attaches both Sam
            # and you to "coming"); "Sam is coming" (sole subject) is not.
            other_subj = any(t.dep_ in ("nsubj", "nsubjpass")
                             and t.head == doc[j].head
                             for t in doc[e:sent.end])
            non_arg = (doc[j].dep_ not in _ARG_DEPS
                       or (doc[j].dep_ == "nsubj" and other_subj))
            tail_ok = all(t.is_punct or t.is_space for t in doc[e:sent.end])
            addressed = _address_context(doc, sent, i, e)
            final_nc = i > start and tail_ok and non_arg and addressed
            initial_nc = i == start and e < sent.end and non_arg and other_subj
            # summons: the whole utterance is just the name plus "?"
            lead_ok = all(t.is_punct or t.is_space for t in doc[sent.start:i])
            summons = (lead_ok and tail_ok and e < sent.end
                       and any(t.text == "?" for t in doc[e:sent.end]))
            if initial or final or medial or final_nc or initial_nc or summons:
                yield (i, j)
            i = e


def build(nlp, client=None):
    lx = lexicons()
    return [
        VocativeDetector(nlp, "VOC-02",
                         lx["vocatives_kinship_endearment"]["entries"],
                         version="voc02-kinship@0.1"),
        VocativeDetector(nlp, "VOC-03",
                         lx["vocatives_impersonal_honorific"]["entries"],
                         version="voc03-impersonal@0.1"),
        Scan("VOC-01", _voc01, version="voc01-name@0.1"),
    ]
