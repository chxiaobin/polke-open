"""Category PAS - the passive across the paradigm and special passives.

PAS-01..10 share one form (a be-passive participle) and differ only by the
tense/aspect/modality of the aux chain -> one deterministic RuleRoutingDetector.
PAS-11/13 are rule sub-types (by-agent, ditransitive). PAS-14/16/17/20 are
lexicon-gated. PAS-12/15/18 (reading/meaning) and PAS-19 stay in the LLM tiers.
"""
from __future__ import annotations
from ..rules import DependencyRuleDetector
from ..routing import RuleRoutingDetector, LexiconRuleDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector
from .common import (PASSIVE_FORM, aux_chain, has_be_auxpass,
                     pair_set, lemma_set)

_FUTURE_MODALS = {"will", "shall"}


def classify_passive(doc, token_ids):
    """Route a be-passive participle to its PAS-01..10 sibling by aux chain."""
    verb = doc[token_ids[0]]
    if verb.tag_ != "VBN" or not has_be_auxpass(verb):
        return None  # get-passive / non-be -> handled elsewhere
    if verb.lemma_ == "get" and any(c.dep_ == "dobj" for c in verb.children):
        return None  # possession "('s) got X" misparsed as passive
    chain = aux_chain(verb)
    modal = [k for k in chain if k.tag_ == "MD"]
    to = [k for k in chain if k.tag_ == "TO"]
    have = [k for k in chain if k.lemma_ == "have"]
    being = [k for k in chain if k.lemma_ == "be" and k.tag_ == "VBG"]
    been = [k for k in chain if k.lemma_ == "be" and k.tag_ == "VBN"]
    be_pres = [k for k in chain if k.lemma_ == "be" and k.tag_ in ("VBZ", "VBP")]
    be_past = [k for k in chain if k.lemma_ == "be" and k.tag_ == "VBD"]

    # non-finite passives: to-infinitive or a bare -ing participle
    if to:
        return "PAS-10"
    if being and not (be_pres or be_past):
        return "PAS-10"
    if modal:
        if have and been:
            return "PAS-09"                      # perfect modal passive
        if any(m.lemma_ in _FUTURE_MODALS for m in modal):
            return "PAS-07"                      # future passive
        return "PAS-08"                          # other modal passive
    if have and been:
        if any(h.tag_ == "VBD" for h in have):
            return "PAS-06"                      # past perfect passive
        return "PAS-05"                          # present perfect passive
    if being:
        if be_past:
            return "PAS-04"                      # past continuous passive
        return "PAS-03"                          # present continuous passive
    if be_past:
        return "PAS-02"                          # past simple passive
    if be_pres:
        return "PAS-01"                          # present simple passive
    return None


def _by_agent_exclude(doc, token_ids):
    return not has_be_auxpass(doc[token_ids[0]])


def _ditransitive_ok(doc, token_ids):
    verb = doc[token_ids[0]]
    if verb.lemma_ == "get" and any(c.dep_ == "dobj" for c in verb.children):
        return None  # possession have-got, not a retained-object passive
    return "PAS-13" if has_be_auxpass(verb) else None


def _phrasal_key(doc, token_ids):
    verb = doc[token_ids[0]]
    if not has_be_auxpass(verb):
        return None
    for k in verb.children:
        if k.dep_ in ("prt", "prep", "advmod", "agent") and k.i != verb.i:
            if k.tag_ in ("RP", "IN"):
                return (verb.lemma_.lower(), k.lower_)
    return None


def _reporting_key_ccomp(doc, token_ids):
    """It is said (that) ... -> anticipatory reporting passive (PAS-16)."""
    verb = doc[token_ids[0]]
    if not has_be_auxpass(verb):
        return None
    has_expl_it = any(c.dep_ in ("nsubjpass", "nsubj") and c.lower_ == "it"
                      for c in verb.children)
    has_clause = any(c.dep_ in ("ccomp", "acomp", "xcomp") for c in verb.children)
    if has_expl_it and has_clause:
        return verb.lemma_.lower()
    return None


def _reporting_key_raised(doc, token_ids):
    """S is said to VERB ... -> subject-to-subject raising passive (PAS-17)."""
    verb = doc[token_ids[0]]
    if not has_be_auxpass(verb):
        return None
    subj = [c for c in verb.children if c.dep_ == "nsubjpass"]
    if not subj or subj[0].lower_ == "it":
        return None
    xc = [c for c in verb.children if c.dep_ == "xcomp"]
    if xc and any(g.tag_ == "TO" for g in xc[0].children):
        return verb.lemma_.lower()
    return None


def _need_ing_key(doc, token_ids):
    verb = doc[token_ids[0]]
    for k in verb.children:
        if k.dep_ == "dobj" and (k.text.lower().endswith("ing") or k.tag_ == "VBG"):
            return verb.lemma_.lower()
    return None


def build(nlp, client=None):
    dets = []

    # PAS-01..10: the tense/aspect/modality paradigm of the be-passive.
    dets.append(RuleRoutingDetector(
        nlp,
        ["PAS-01", "PAS-02", "PAS-03", "PAS-04", "PAS-05",
         "PAS-06", "PAS-07", "PAS-08", "PAS-09", "PAS-10"],
        PASSIVE_FORM, classify_passive, version="pas-paradigm@0.1"))

    # PAS-11 by-agent phrase.
    by_agent = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ap",
         "RIGHT_ATTRS": {"DEP": "auxpass"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "by",
         "RIGHT_ATTRS": {"DEP": "agent"}},
    ]
    dets.append(DependencyRuleDetector(nlp, "PAS-11", by_agent,
                                       exclude=_by_agent_exclude,
                                       version="pas11-byagent@0.1"))

    # PAS-13 ditransitive passive: retained object (dobj) or dative to-phrase.
    ditr_dobj = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ap",
         "RIGHT_ATTRS": {"DEP": "auxpass"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "obj",
         "RIGHT_ATTRS": {"DEP": "dobj"}},
    ]
    ditr_dative = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ap",
         "RIGHT_ATTRS": {"DEP": "auxpass"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "dat",
         "RIGHT_ATTRS": {"DEP": "dative"}},
    ]
    dets.append(RuleRoutingDetector(nlp, ["PAS-13"], [ditr_dobj, ditr_dative],
                                    _ditransitive_ok, version="pas13-ditr@0.1"))

    # PAS-14 passive of phrasal / prepositional verbs (lexicon-gated).
    phrasal_forms = [
        [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ap",
             "RIGHT_ATTRS": {"DEP": "auxpass"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prt",
             "RIGHT_ATTRS": {"DEP": "prt"}},
        ],
        [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ap",
             "RIGHT_ATTRS": {"DEP": "auxpass"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "adv",
             "RIGHT_ATTRS": {"DEP": {"IN": ["advmod", "prep"]}, "TAG": "IN"}},
        ],
    ]
    dets.append(LexiconRuleDetector(
        nlp, "PAS-14", phrasal_forms, pair_set("phrasal_passive_verbs"),
        _phrasal_key, version="pas14-phrasal@0.1"))

    # PAS-16 anticipatory reporting passive: It is said that ...
    report_form = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ap",
         "RIGHT_ATTRS": {"DEP": "auxpass"}},
    ]
    dets.append(LexiconRuleDetector(
        nlp, "PAS-16", report_form, lemma_set("reporting_verbs"),
        _reporting_key_ccomp, version="pas16-report@0.1"))

    # PAS-17 subject-raised reporting passive: S is said to VERB ...
    dets.append(LexiconRuleDetector(
        nlp, "PAS-17", report_form, lemma_set("reporting_verbs"),
        _reporting_key_raised, version="pas17-raised@0.1"))

    # PAS-20 need/want/require + -ing (passive meaning).
    need_form = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "obj",
         "RIGHT_ATTRS": {"DEP": "dobj"}},
    ]
    dets.append(LexiconRuleDetector(
        nlp, "PAS-20", need_form, lemma_set("need_ing_verbs"),
        _need_ing_key, version="pas20-need-ing@0.1"))

    # --- LLM tier (structural; skipped offline) ------------------------------
    # PAS-12 with-instrument / of-material: passive participle + with/of PP.
    pas12_form = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ap",
         "RIGHT_ATTRS": {"DEP": "auxpass", "LEMMA": "be"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep",
         "RIGHT_ATTRS": {"DEP": "prep", "LOWER": {"IN": ["with", "of"]}}},
    ]
    dets.append(LLMReadingDetector(
        nlp, ["PAS-12"], pas12_form,
        'A be-passive is followed by a with/of phrase. Decide whether it is the '
        'with-instrument / of-material use (PAS-12) rather than an agent phrase. '
        'Return JSON {"construct_id":"PAS-12"|"NONE","confidence":0..1,"rationale":"..."}.',
        client=client, version="pas12-with@0.1"))

    # PAS-15 get-passive: participle with a *get* passive auxiliary.
    pas15_form = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TAG": "VBN"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ap",
         "RIGHT_ATTRS": {"DEP": "auxpass", "LEMMA": "get"}},
    ]
    dets.append(LLMReadingDetector(
        nlp, ["PAS-15"], pas15_form,
        'A get-passive (get + past participle). Confirm the dynamic/adversative/'
        'fortuitous get-passive reading (PAS-15). Return JSON '
        '{"construct_id":"PAS-15"|"NONE","confidence":0..1,"rationale":"..."}.',
        client=client, version="pas15-get@0.1"))

    # PAS-18 have/get something done (causative): have/get + object + VBN.
    pas18_form = [
        {"RIGHT_ID": "matrix", "RIGHT_ATTRS": {"LEMMA": {"IN": ["have", "get"]}, "POS": "VERB"}},
        {"LEFT_ID": "matrix", "REL_OP": ">", "RIGHT_ID": "obj",
         "RIGHT_ATTRS": {"DEP": "dobj"}},
        {"LEFT_ID": "obj", "REL_OP": ">", "RIGHT_ID": "done",
         "RIGHT_ATTRS": {"TAG": "VBN"}},
    ]
    dets.append(LLMReadingDetector(
        nlp, ["PAS-18"], pas18_form,
        'A have/get + object + past participle causative ("I had my visa '
        'renewed"). Confirm the service/causative reading (PAS-18) vs the '
        'adversative "have something happen to one" (PAS-19). Return JSON '
        '{"construct_id":"PAS-18"|"NONE","confidence":0..1,"rationale":"..."}.',
        client=client, version="pas18-causative@0.1"))

    # PAS-19 adversative causative ("He had his car stolen"): no clean form
    # trigger separating it from PAS-18 -> standalone LLM over the sentence.
    dets.append(LLMStandaloneDetector(
        "PAS-19",
        'Decide whether the sentence expresses an adversative causative: the '
        'subject experiences an unfortunate event affecting their possession '
        '("He had his car stolen"), as opposed to arranging a service. Return '
        'JSON {"construct_id":"PAS-19"|"NONE","confidence":0..1,"rationale":"..."}.',
        client=client, version="pas19-adversative@0.1"))

    return dets
