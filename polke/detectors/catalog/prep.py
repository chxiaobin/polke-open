"""Category PREP - prepositions.

Meaning sets (place / time / movement / other) are closed classes -> surface
PhraseLexiconDetector, which also fires on the inventory's bare-list fragment
examples. Dependent prepositions (verb/adj/noun + prep) reuse the shared
lexicons via DependencyMatcher (PREP-18 is the reference; 19/20/21 mirror it).
Syntax-of-prepositions (stranding, pied-piping, omission, PP postmodifier) are
rules. The multi-sense prepositions (by/with/for/of/about/as, for/since,
by/until) are the LLM tier.

Flagged in NOTES: PREP-01 vs PREP-04 share {in,on,at}; separated here by a
DATE/TIME object heuristic, not perfectly. PREP-08's "in an hour" fragment
overlaps PREP-04. PREP-24 (omission) detects a proxy, not true absence.
"""
from __future__ import annotations
from ..rules import DependencyRuleDetector
from ..lexicon import LexiconDetector
from ..lexical import PhraseLexiconDetector
from ..routing import RuleRoutingDetector, CompositeDetector
from ..llm import LLMStandaloneDetector
from .common import pair_set

_TIME_ENT = {"DATE", "TIME"}
# First noun of a complex preposition ("in front of", "in spite of", ...): a
# bare in/on/at here is NOT the simple place/time preposition.
_COMPLEX_NEXT = {"front", "spite", "addition", "case", "order", "favour",
                 "favor", "charge", "terms", "respect", "view", "accordance",
                 "behalf", "lieu", "place", "light", "face", "search", "aid",
                 "memory", "honour", "honor", "return", "response", "contrast"}


def _has_temporal_obj(prep):
    for c in prep.children:
        if c.dep_ in ("pobj", "pcomp"):
            if c.ent_type_ in _TIME_ENT or c.like_num:
                return True
    return False


def _simple_pp(doc, start):
    prep = doc[start]
    if not any(c.dep_ == "pobj" for c in prep.children):
        return False
    if start + 1 < len(doc) and doc[start + 1].lower_ in _COMPLEX_NEXT:
        return False
    return True


def _place_gate(doc, start, end):
    return _simple_pp(doc, start) and not _has_temporal_obj(doc[start])


def _transitive_prep_gate(doc, start, end):
    """True only when the matched phrase is a real transitive preposition
    (has a pobj/pcomp), not a verb particle or bare adverb ("tidy up",
    "hand it over", "sneak off")."""
    for i in range(start, end):
        t = doc[i]
        if t.dep_ == "prt":
            return False
        if any(c.dep_ in ("pobj", "pcomp") for c in t.children):
            return True
    return False


def _time_gate(doc, start, end):
    return _simple_pp(doc, start) and _has_temporal_obj(doc[start])


# --- dependent-preposition key functions (mirror the PREP-18 reference) ------
def _head_prep_key(pos_tag):
    def key_fn(doc, token_ids):
        head = prep = None
        for t in token_ids:
            tok = doc[t]
            if tok.dep_ == "prep":
                prep = tok
            elif tok.pos_ == pos_tag:
                head = tok
        if head is None or prep is None:
            return None
        return (head.lemma_.lower(), prep.lower_)
    return key_fn


def build(nlp, client=None):
    dets = []

    # PREP-01 place in/on/at  vs  PREP-04 time at/on/in (shared prepositions,
    # split by whether the object is a date/time/number).
    dets.append(PhraseLexiconDetector(nlp, "PREP-01", ["in", "on", "at"],
                                      gate=_place_gate, version="prep01@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "PREP-04", ["at", "on", "in"],
                                      gate=_time_gate, version="prep04@0.1"))

    # PREP-02 spatial relations.
    dets.append(PhraseLexiconDetector(
        nlp, "PREP-02",
        ["above", "over", "below", "under", "beneath", "between", "among",
         "behind", "beside", "opposite", "in front of", "next to"],
        gate=_transitive_prep_gate, version="prep02@0.1"))

    # PREP-03 inside/outside, against, around, along, beyond.
    dets.append(PhraseLexiconDetector(
        nlp, "PREP-03",
        ["inside", "outside", "against", "around", "round", "along", "beyond"],
        gate=_transitive_prep_gate, version="prep03@0.1"))

    # PREP-06 during/throughout/over/within.
    dets.append(PhraseLexiconDetector(
        nlp, "PREP-06", ["during", "throughout", "over", "within"],
        gate=_transitive_prep_gate, version="prep06@0.1"))

    # PREP-08 from / ago / in + period (time span onset).
    dets.append(PhraseLexiconDetector(
        nlp, "PREP-08",
        ["from", "ago", "in an hour", "in a minute", "in a moment",
         "in a week", "in a year", "in a day", "in a month"],
        version="prep08@0.1"))

    # PREP-09 movement: to/into/onto/out of/off/from (goal/source).
    dets.append(PhraseLexiconDetector(
        nlp, "PREP-09",
        ["into", "onto", "out of", "off", "off of"],
        gate=_transitive_prep_gate, version="prep09@0.1"))

    # PREP-10 path/route prepositions.
    dets.append(PhraseLexiconDetector(
        nlp, "PREP-10",
        ["towards", "toward", "away from", "up", "down", "across", "through",
         "along", "past", "via"],
        gate=_transitive_prep_gate, version="prep10@0.1"))

    # PREP-17 complex prepositions (concession/cause/exception/reference).
    dets.append(PhraseLexiconDetector(
        nlp, "PREP-17",
        ["despite", "in spite of", "because of", "due to", "instead of",
         "apart from", "except for", "according to", "thanks to", "regarding",
         "owing to", "in addition to", "as for", "on behalf of",
         "in accordance with", "with regard to"],
        version="prep17@0.1"))

    # PREP-19 adjective + preposition (shared adj_prep lexicon).
    adj_prep_form = [
        {"RIGHT_ID": "adj", "RIGHT_ATTRS": {"POS": "ADJ"}},
        {"LEFT_ID": "adj", "REL_OP": ">", "RIGHT_ID": "prep",
         "RIGHT_ATTRS": {"DEP": "prep"}},
    ]
    dets.append(LexiconDetector(nlp, "PREP-19", adj_prep_form,
                                pair_set("adj_prep"), _head_prep_key("ADJ"),
                                version="prep19@0.1"))

    # PREP-20 noun + preposition (shared noun_prep lexicon).
    noun_prep_form = [
        {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
        {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "prep",
         "RIGHT_ATTRS": {"DEP": "prep"}},
    ]
    dets.append(LexiconDetector(nlp, "PREP-20", noun_prep_form,
                                pair_set("noun_prep"), _head_prep_key("NOUN"),
                                version="prep20@0.1"))

    # PREP-21 preposition + -ing (governs a gerund via prep) OR the fixed
    # phrase "with a view to" (no overt gerund).
    prep_ing_form = [
        {"RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "prt"]}}},
        {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "ger",
         "RIGHT_ATTRS": {"DEP": {"IN": ["pcomp", "pobj"]}, "TAG": "VBG"}},
    ]
    dets.append(CompositeDetector("PREP-21", [
        RuleRoutingDetector(nlp, ["PREP-21"], prep_ing_form,
                            lambda doc, tids: "PREP-21", span="match"),
        PhraseLexiconDetector(nlp, "PREP-21",
                              ["with a view to", "with an eye to"]),
    ], detector_type="lexicon", version="prep21@0.1"))

    # PREP-22 preposition stranding: a preposition with no overt object, or one
    # whose object is a wh-word fronted to its left (What ... looking at?).
    def _stranded(doc, tids):
        p = doc[tids[0]]
        if p.dep_ not in ("prep", "agent", "dative"):
            return None                   # dep=prt is a verb particle, never stranding
        pobj = [c for c in p.children if c.dep_ in ("pobj", "pcomp")]
        if any(o.i < p.i for o in pobj):
            return "PREP-22"              # fronted object = extraction
        if pobj:
            return None
        # objectless prep counts only with extraction evidence: a wh-word
        # earlier in the sentence, a relative/infinitival clause host, or a
        # passive host ("was talked about").
        head = p.head
        sent = p.sent
        if any(t.tag_ in ("WP", "WDT", "WP$") and t.i < p.i for t in sent):
            return "PREP-22"
        if head.dep_ in ("relcl", "acl"):
            return "PREP-22"
        if any(c.dep_ in ("auxpass", "nsubjpass") for c in head.children):
            return "PREP-22"
        return None
    dets.append(RuleRoutingDetector(
        nlp, ["PREP-22"], [{"RIGHT_ID": "p", "RIGHT_ATTRS": {"POS": "ADP"}}],
        _stranded, version="prep22-strand@0.1", span="match"))

    # PREP-23 pied-piping: preposition + wh-object to its right (to whom / in which).
    pied = [
        {"RIGHT_ID": "p", "RIGHT_ATTRS": {"POS": {"IN": ["ADP", "PART"]}}},
        {"LEFT_ID": "p", "REL_OP": ">", "RIGHT_ID": "wh",
         "RIGHT_ATTRS": {"TAG": {"IN": ["WP", "WDT", "WP$"]}}},
    ]

    def _pied_ok(doc, tids):
        p, wh = doc[tids[0]], doc[tids[1]]
        return "PREP-23" if wh.i > p.i else None
    dets.append(RuleRoutingDetector(nlp, ["PREP-23"], pied, _pied_ok,
                                    version="prep23-pied@0.1", span="match"))

    # PREP-24 preposition omission: bare temporal NP adjunct, or a transitive
    # verb that (unlike its learner-error variant) takes a bare object.
    _NOPREP_VERBS = {"discuss", "enter", "reach", "mention", "marry",
                     "approach", "answer", "resemble", "lack", "attend",
                     "contact", "phone", "email"}
    _TIME_NOUNS = {"week", "morning", "afternoon", "evening", "night", "day",
                   "month", "year", "monday", "tuesday", "wednesday",
                   "thursday", "friday", "saturday", "sunday", "today",
                   "tomorrow", "yesterday", "weekend"}

    class _OmissionDetector(DependencyRuleDetector):
        detector_type = "lexicon"

        def __init__(self, nlp):
            self.construct_ids = ["PREP-24"]
            self.version = "prep24-omission@0.1"
            self._nlp = nlp

        def match(self, doc, text_id="doc"):
            from ...schema import Annotation, Span
            out, seen = [], set()
            for t in doc:
                hit = None
                if t.pos_ == "VERB" and t.lemma_.lower() in _NOPREP_VERBS:
                    hit = t
                elif (t.dep_ == "npadvmod" and t.lemma_.lower() in _TIME_NOUNS) \
                        or (t.lemma_.lower() in _TIME_NOUNS
                            and any(c.lower_ in ("next", "last", "this", "every")
                                    for c in t.children)):
                    hit = t
                if hit is None or hit.i in seen:
                    continue
                seen.add(hit.i)
                out.append(Annotation(
                    text_id=text_id, construct_id="PREP-24",
                    span=Span(hit.idx, hit.idx + len(hit.text), hit.i, hit.i),
                    detector_type="lexicon", detector_version=self.version,
                    confidence=1.0, evidence={"tokens": [hit.i], "matched": hit.text}))
            return out
    dets.append(_OmissionDetector(nlp))

    # PREP-25 prepositional phrase as postmodifier/complement/adjunct: a noun
    # postmodified by a prep phrase (the key to success / a man of his word).
    pp_postmod = [
        {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
        {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "prep",
         "RIGHT_ATTRS": {"DEP": "prep"}},
        {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "obj",
         "RIGHT_ATTRS": {"DEP": "pobj"}},
    ]

    def _pp_postmod_ok(doc, tids):
        # exclude the dependent-noun-prep collocations (those are PREP-20)
        noun = doc[tids[0]]
        prep = next((doc[t] for t in tids if doc[t].dep_ == "prep"), None)
        if prep is None:
            return None
        if (noun.lemma_.lower(), prep.lower_) in pair_set("noun_prep"):
            return None
        return "PREP-25"
    dets.append(RuleRoutingDetector(nlp, ["PREP-25"], pp_postmod,
                                    _pp_postmod_ok, version="prep25-postmod@0.1"))

    # --- LLM tier: multi-sense prepositions (structural; skipped offline) -----
    _llm = {
        "PREP-05": "for/since expressing duration vs starting point of a time span",
        "PREP-07": "by/until(till) expressing a deadline vs up-to-a-point in time",
        "PREP-11": "the preposition OF expressing possession (the leg of the "
                  "table), partitive (a cup of tea), material (made of wood) "
                  "or topic (tales of adventure); other prepositions such as "
                  "about/on do not count",
        "PREP-12": "with/without expressing accompaniment/instrument/manner",
        "PREP-13": "by expressing agent/means/measure",
        "PREP-14": "for expressing purpose/recipient/duration/exchange",
        "PREP-15": "about/on expressing topic",
        "PREP-16": "as expressing role/function",
    }
    for cid, desc in _llm.items():
        sys = (f"Decide whether the sentence contains the preposition use: {desc}. "
               f'Return JSON {{"construct_id":"{cid}"|"NONE","confidence":0..1,'
               f'"rationale":"..."}}.')
        dets.append(LLMStandaloneDetector(cid, sys, client=client,
                                          version=f"{cid.lower()}-llm@0.1"))

    return dets
