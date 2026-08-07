"""Category ELS - ellipsis & substitution (ELS-01..11).

Lexicon tier (tested offline):
  ELS-06 clausal so / not after a mental/reporting predicate (I think so; I hope
         not; I'm afraid so).
LLM tier (registered, skipped offline) - identifying an elided/substituted
element is a reading task:
  ELS-01 VP-ellipsis; ELS-02 infinitive ellipsis (to); ELS-03 gapping/stripping;
  ELS-04 nominal ellipsis; ELS-05 do/do so/do it/do that substitution;
  ELS-07 the same/such substitution; ELS-08 response/answer ellipsis;
  ELS-09 one/ones nominal substitution; ELS-10 comparative-clause ellipsis;
  ELS-11 situational ellipsis (informal).

spaCy en_core_web_sm labels verified: clausal "so" is ``advmod`` and "not" is
``neg``, attached to the reporting predicate (or, for "I'm afraid so", to the
copula whose ``acomp`` is afraid).
"""
from __future__ import annotations
from ..routing import LexiconRuleDetector
from ..llm import LLMReadingDetector, LLMStandaloneDetector

# Predicates that license clausal so/not.
_MENTAL = {"think", "hope", "believe", "suppose", "guess", "expect", "imagine",
           "say", "tell", "reckon", "assume", "presume", "fear", "trust",
           "suspect", "gather", "doubt", "hear", "seem", "appear", "afraid"}
_AFRAID = {"afraid", "sure", "certain", "hopeful", "glad", "sorry"}


_SONOT_FORM = [{"RIGHT_ID": "x",
                "RIGHT_ATTRS": {"LOWER": {"IN": ["so", "not"]}}}]


def _els06_key(doc, token_ids):
    x = doc[token_ids[0]]
    if x.dep_ not in ("advmod", "neg", "oprd", "acomp", "dep"):
        return None
    h = x.head
    if h.lemma_ in _MENTAL:
        return h.lemma_
    if h.lemma_ == "be":
        for c in h.children:
            if c.dep_ == "acomp" and c.lower_ in _AFRAID:
                return "afraid"
    return None


# --------------------------------------------------------------------------- #
# LLM tier.
# --------------------------------------------------------------------------- #
_POSTV_SYS = (
    "Post-verbal ellipsis/substitution after an operator. Choose:\n"
    "ELS-01 VP-ellipsis - stranded operator with the VP omitted (she has Ø; I "
    "will) | ELS-02 infinitive ellipsis - stranded 'to' (I'd love to Ø) | "
    "ELS-05 do/do so/do it/do that substitution (he did (so)).")
_GAP_SYS = ("Return ELS-03 for gapping/stripping in coordination - the verb is "
            "omitted in the second conjunct with NO auxiliary left behind (I "
            "ordered tea and she Ø coffee; ..., and Bob too). Post-auxiliary "
            "ellipsis ('... but she has Ø') is a different construct - return "
            "NONE.")
_NOM_SYS = ("Return ELS-04 for nominal ellipsis - a determiner/numeral/adjective "
            "with the head noun omitted (I'll take two Ø; the rich Ø; the first "
            "Ø to arrive).")
_SAME_SYS = ("Return ELS-07 for substitution with the literal words 'the same' "
             "or 'such' (I'll have the same; such was his anger). Other "
             "ellipsis or omitted material that could merely be paraphrased "
             "with 'the same' returns NONE.")
_RESP_SYS = ("Return ELS-08 for response/answer ellipsis - a short reply or "
             "reaction standing for a full clause (Yes, I do; Me too; Not me; "
             "So do I). Ellipsis inside one speaker's own coordination ('and "
             "she coffee', '... but she has') is a different construct - "
             "return NONE.")
_ONE_SYS = ("Return ELS-09 for one/ones nominal substitution (the blue one; the "
            "ones I bought).")
_CMP_SYS = ("Return ELS-10 for comparative-clause ellipsis (She's taller than me "
            "Ø; faster than expected Ø).")
_SIT_SYS = ("Return ELS-11 for situational ellipsis - informal omission of "
            "subject/auxiliary at the start of an utterance (Want a coffee? Seen "
            "John?).")

_ROOT_FORM = [{"RIGHT_ID": "r", "RIGHT_ATTRS": {"DEP": "ROOT"}}]
_TO_FORM = [{"RIGHT_ID": "t", "RIGHT_ATTRS": {"TAG": "TO"}}]
_CONJ_FORM = [{"RIGHT_ID": "c", "RIGHT_ATTRS": {"DEP": "conj"}}]
_ONE_FORM = [{"RIGHT_ID": "o", "RIGHT_ATTRS": {"LOWER": {"IN": ["one", "ones"]}}}]
_THAN_FORM = [{"RIGHT_ID": "t", "RIGHT_ATTRS": {"LOWER": "than"}}]


def build(nlp, client=None):
    return [
        LexiconRuleDetector(nlp, "ELS-06", _SONOT_FORM, _MENTAL | {"afraid"},
                            _els06_key, span="match", version="els06-sonot@0.1"),
        # LLM tier
        LLMReadingDetector(nlp, ["ELS-01", "ELS-02", "ELS-05"], _ROOT_FORM,
                           _POSTV_SYS, client=client, version="els-postv@0.1"),
        LLMReadingDetector(nlp, ["ELS-03"], _CONJ_FORM, _GAP_SYS,
                           client=client, version="els03-gapping@0.1"),
        LLMReadingDetector(nlp, ["ELS-04"], _ROOT_FORM, _NOM_SYS,
                           client=client, version="els04-nominal@0.1"),
        LLMReadingDetector(nlp, ["ELS-07"], _ROOT_FORM, _SAME_SYS,
                           client=client, version="els07-same@0.1"),
        LLMReadingDetector(nlp, ["ELS-08"], _ROOT_FORM, _RESP_SYS,
                           client=client, version="els08-response@0.1"),
        LLMReadingDetector(nlp, ["ELS-09"], _ONE_FORM, _ONE_SYS,
                           client=client, version="els09-one@0.1"),
        LLMReadingDetector(nlp, ["ELS-10"], _THAN_FORM, _CMP_SYS,
                           client=client, version="els10-comp@0.1"),
        LLMStandaloneDetector("ELS-11", _SIT_SYS, client=client,
                              version="els11-situational@0.1"),
    ]
