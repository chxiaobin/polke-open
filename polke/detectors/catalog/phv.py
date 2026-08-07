"""Category PHV - phrasal & prepositional verbs (by syntactic type).

The distinction is structural: particle (prt) vs preposition (prep), presence and
position of the object. PHV-01/02/03 (particle verbs) are routed structurally.
PHV-04/05 (preposition-based) are lexicon-gated so free adjunct PPs
("walk in the park") don't fire. PHV-07 is the aspectual-particle set. PHV-06
(literal vs idiomatic) is the hybrid_lexicon_llm tier.
"""
from __future__ import annotations
from ..routing import RuleRoutingDetector, LexiconRuleDetector
from ..llm import LLMStandaloneDetector

# Inseparable prepositional verbs (verb + dependent preposition, transitive).
_PREP_VERBS = {
    ("look", "after"), ("look", "at"), ("look", "for"), ("look", "into"),
    ("wait", "for"), ("listen", "to"), ("depend", "on"), ("rely", "on"),
    ("deal", "with"), ("care", "for"), ("account", "for"), ("apply", "for"),
    ("belong", "to"), ("refer", "to"), ("consist", "of"), ("cope", "with"),
    ("insist", "on"), ("object", "to"), ("approve", "of"), ("cater", "for"),
    ("call", "on"), ("go", "through"), ("run", "into"), ("come", "across"),
}
# Three-part phrasal-prepositional verbs: verb lemma -> allowed particle chains.
_PHRASAL_PREP_VERBS = {
    "look": {("forward", "to")}, "run": {("out", "of")},
    "put": {("up", "with")}, "get": {("on", "with"), ("out", "of"), ("away", "with")},
    "catch": {("up", "with")}, "come": {("up", "with")}, "keep": {("up", "with")},
}
_ASPECTUAL_PARTICLES = {"up", "on", "away", "out", "through", "along", "over"}


def _particles(verb):
    return [c for c in verb.children if c.dep_ == "prt"]


def _preps(verb):
    return [c for c in verb.children if c.dep_ == "prep"]


def _advmod_particle(verb):
    return [c for c in verb.children if c.dep_ == "advmod" and c.tag_ == "RB"
            and c.lower_ in ("forward", "back", "ahead", "around", "about")]


def _classify_particle_pv(doc, tids):
    """PHV-01/02/03: verb + prt, by object presence/type/position."""
    verb = doc[tids[0]]
    prts = _particles(verb)
    if not prts:
        return None
    if _preps(verb) or _advmod_particle(verb):
        return None  # three-part -> PHV-05, handled separately
    prt = prts[0]
    dobj = [c for c in verb.children if c.dep_ == "dobj"]
    if not dobj:
        return "PHV-01"                                   # intransitive
    obj = dobj[0]
    if obj.pos_ == "PRON" and obj.i < prt.i:
        return "PHV-03"                                   # obligatory separation
    return "PHV-02"                                       # transitive separable


def _prep_verb_key(doc, tids):
    verb = doc[tids[0]]
    if _particles(verb):
        return None
    preps = _preps(verb)
    for p in preps:
        if any(gc.dep_ == "prep" for gc in p.children):
            return None  # three-part (run out of) -> PHV-05
        if (verb.lemma_.lower(), p.lower_) in _PREP_VERBS:
            return (verb.lemma_.lower(), p.lower_)
    return None


def _classify_phrasal_prep(doc, tids):
    """PHV-05: three-part verb (prt + prep, advmod-particle + prep, or prep->prep)."""
    verb = doc[tids[0]]
    lemma = verb.lemma_.lower()
    parts = [c.lower_ for c in _particles(verb)] + [c.lower_ for c in _advmod_particle(verb)]
    preps = _preps(verb)
    prep_words = [p.lower_ for p in preps]
    # prep -> prep chain (run out of milk)
    nested = [(p.lower_, gc.lower_) for p in preps for gc in p.children if gc.dep_ == "prep"]
    combos = set()
    for a in parts:
        for b in prep_words:
            combos.add((a, b))
    combos |= set(nested)
    allowed = _PHRASAL_PREP_VERBS.get(lemma)
    if allowed and isinstance(allowed, set):
        if combos & allowed:
            return "PHV-05"
    # generic: any verb with both a particle and a preposition is three-part
    if parts and prep_words:
        return "PHV-05"
    if nested:
        return "PHV-05"
    return None


def _classify_aspectual(doc, tids):
    verb = doc[tids[0]]
    for prt in _particles(verb):
        if prt.lower_ in _ASPECTUAL_PARTICLES:
            return "PHV-07"
    return None


def build(nlp, client=None):
    dets = []
    VERB_PRT = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prt",
         "RIGHT_ATTRS": {"DEP": "prt"}},
    ]
    VERB_PREP = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep",
         "RIGHT_ATTRS": {"DEP": "prep"}},
    ]

    dets.append(RuleRoutingDetector(
        nlp, ["PHV-01", "PHV-02", "PHV-03"], VERB_PRT,
        _classify_particle_pv, version="phv-particle@0.1"))

    # PHV-04 prepositional verb (lexicon-gated).
    dets.append(LexiconRuleDetector(
        nlp, "PHV-04", VERB_PREP, _PREP_VERBS, _prep_verb_key,
        version="phv04-prep@0.1"))

    # PHV-05 phrasal-prepositional (three-part).
    dets.append(RuleRoutingDetector(
        nlp, ["PHV-05"], [VERB_PRT, VERB_PREP], _classify_phrasal_prep,
        version="phv05-3part@0.1"))

    # PHV-07 aspectual particles.
    dets.append(RuleRoutingDetector(
        nlp, ["PHV-07"], VERB_PRT, _classify_aspectual,
        version="phv07-aspect@0.1"))

    # PHV-06 literal vs idiomatic meaning (hybrid_lexicon_llm; structural).
    dets.append(LLMStandaloneDetector(
        "PHV-06",
        'A phrasal verb can be literal (spatial/compositional) or idiomatic. '
        'Decide whether the sentence uses, in its IDIOMATIC sense, a phrasal '
        'verb that ALSO has a literal spatial sense (take off = succeed/depart '
        'vs take off = remove). Combinations with only one established sense '
        '(look after, depend on) are not this contrast - return NONE. '
        'Return JSON {"construct_id":"PHV-06"|"NONE","confidence":0..1,"rationale":"..."}.',
        client=client, version="phv06-llm@0.1"))
    return dets
