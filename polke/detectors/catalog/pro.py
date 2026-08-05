"""Category PRO - pronouns.

Personal, possessive, reflexive, reciprocal, demonstrative and quantifying
pronouns, plus indefinite + adjective/to-infinitive, are CLOSED-class FORM
triggers -> small rules and surface matchers with word lists kept here. The
readings that need meaning (generic/impersonal you/they/one; dummy vs
anticipatory it; emphatic reflexive; indefinite compounds; singular they;
one/ones substitution; formal one) are the LLM tier and are registered but
skipped offline.
"""
from __future__ import annotations
from ..lexical import PhraseLexiconDetector
from ..llm import LLMStandaloneDetector
from ...schema import Annotation, Span
from ..base import Detector

# personal pronouns (nominative/accusative). 'you' and 'it' are excluded: they
# carry the generic (PRO-02) and dummy/anticipatory (PRO-03/04) readings.
PERSONAL = {"i", "me", "he", "him", "she", "her", "we", "us", "they", "them"}
POSS_PRON = ["mine", "yours", "hers", "ours", "theirs"]
REFLEXIVE = {"myself", "yourself", "himself", "herself", "itself", "oneself",
             "ourselves", "yourselves", "themselves"}
INDEFINITE = {
    "something", "anything", "nothing", "everything", "somebody", "anybody",
    "nobody", "everybody", "someone", "anyone", "everyone", "somewhere",
    "anywhere", "nowhere", "everywhere",
}
DEMONSTRATIVES = {"this", "that", "these", "those"}
QUANT = ["all", "none", "some", "both", "many", "each", "one", "most", "few",
         "several", "either", "neither", "half", "any", "enough", "much",
         "plenty", "lots", "a lot", "a couple", "a number"]
PRON_OBJ = ["them", "us", "you", "it", "these", "those", "this", "that",
            "me", "him", "her", "which", "mine", "ours", "yours", "theirs"]


def _ann(text_id, cid, doc, lo, hi, ver, dtype="rule", matched=None):
    span = doc[lo:hi + 1]
    return Annotation(
        text_id=text_id, construct_id=cid,
        span=Span(span.start_char, span.end_char, lo, hi),
        detector_type=dtype, detector_version=ver, confidence=1.0,
        evidence={"tokens": list(range(lo, hi + 1)),
                  "matched": matched or span.text})


class _Personal(Detector):
    """PRO-01: personal subject/object pronouns (I/me, he/him, they/them)."""
    construct_ids = ["PRO-01"]
    detector_type = "rule"
    version = "pro01-personal@0.1"

    def match(self, doc, text_id="doc"):
        return [_ann(text_id, "PRO-01", doc, t.i, t.i, self.version)
                for t in doc if t.tag_ == "PRP" and t.lower_ in PERSONAL]


class _Reflexive(Detector):
    """PRO-06: true (argument) reflexive - object of a verb/preposition and
    coreferential, excluding the 'by oneself' emphatic use (PRO-07)."""
    construct_ids = ["PRO-06"]
    detector_type = "rule"
    version = "pro06-reflexive@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ not in REFLEXIVE:
                continue
            if t.dep_ in ("dobj", "dative", "attr", "oprd"):
                out.append(_ann(text_id, "PRO-06", doc, t.i, t.i, self.version))
            elif t.dep_ == "pobj" and t.head.lower_ != "by":
                out.append(_ann(text_id, "PRO-06", doc, t.i, t.i, self.version))
        return out


class _IndefiniteMod(Detector):
    """PRO-10: indefinite pronoun + postmodifying adjective or to-infinitive
    (something new; nothing to do)."""
    construct_ids = ["PRO-10"]
    detector_type = "rule"
    version = "pro10-indef-mod@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ not in INDEFINITE:
                continue
            has_adj = any(c.dep_ == "amod" for c in t.children)
            has_inf = any(c.dep_ in ("relcl", "acl") and
                          any(g.tag_ == "TO" for g in c.children)
                          for c in t.children)
            # also catch a following "to + VB" not attached as a child
            if not has_inf and t.i + 1 < len(doc) and doc[t.i + 1].tag_ == "TO":
                has_inf = True
            if has_adj or has_inf:
                out.append(_ann(text_id, "PRO-10", doc, t.i, t.i, self.version))
        return out


class _DemonstrativePron(Detector):
    """PRO-14: standalone demonstrative pronoun (This is mine; Those are old) -
    a this/that/these/those NOT used as a determiner."""
    construct_ids = ["PRO-14"]
    detector_type = "rule"
    version = "pro14-demon-pron@0.1"

    def match(self, doc, text_id="doc"):
        out = []
        for t in doc:
            if t.lower_ in DEMONSTRATIVES and t.dep_ != "det" \
                    and t.pos_ in ("PRON", "DET"):
                # a genuine pronoun: not modifying a following noun
                if t.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "attr",
                              "conj", "ROOT", "dative"):
                    out.append(_ann(text_id, "PRO-14", doc, t.i, t.i,
                                    self.version))
        return out


class _QuantOfPron(Detector):
    """PRO-15: quantifying pronoun + of + pronoun/demonstrative (all of them;
    none of us; some of these)."""
    construct_ids = ["PRO-15"]
    detector_type = "rule"
    version = "pro15-quant-of@0.1"

    def __init__(self, nlp):
        from spacy.matcher import Matcher
        self._m = Matcher(nlp.vocab)
        self._m.add("q", [
            [{"LOWER": {"IN": QUANT}}, {"LOWER": "of"},
             {"LOWER": {"IN": PRON_OBJ}}],
        ])

    def match(self, doc, text_id="doc"):
        out, seen = [], set()
        for _mid, start, end in self._m(doc):
            if start in seen:
                continue
            seen.add(start)
            out.append(_ann(text_id, "PRO-15", doc, start, end - 1,
                            self.version))
        return out


def build(nlp, client=None):
    dets = []

    # --- form / rule tiers ------------------------------------------------- #
    dets.append(_Personal())              # PRO-01
    dets.append(_Reflexive())             # PRO-06
    dets.append(_IndefiniteMod())         # PRO-10
    dets.append(_DemonstrativePron())     # PRO-14
    dets.append(_QuantOfPron(nlp))        # PRO-15

    # --- surface lexicon tiers -------------------------------------------- #
    dets.append(PhraseLexiconDetector(nlp, "PRO-05", POSS_PRON,
                                      version="pro05-poss-pron@0.1"))
    dets.append(PhraseLexiconDetector(nlp, "PRO-08", ["each other", "one another"],
                                      version="pro08-reciprocal@0.1"))

    # --- LLM tier (registered structurally; skipped offline) --------------- #
    _llm = {
        "PRO-02": "generic / impersonal you, they, one or we (you never know; "
                  "they say it'll rain)",
        "PRO-03": "dummy it for weather, time or distance (it's raining; it's "
                  "five o'clock; it's far)",
        "PRO-04": "anticipatory it in extraposition (it's hard to say; it seems "
                  "that...)",
        "PRO-07": "emphatic reflexive or by + oneself (I'll do it myself; he "
                  "lives by himself)",
        "PRO-09": "an indefinite compound pronoun (someone/anyone/nobody/"
                  "everything...)",
        "PRO-11": "singular they with a singular indefinite antecedent (someone "
                  "left their bag)",
        "PRO-12": "one / ones as a substitute noun (the red one; the ones on the "
                  "left)",
        "PRO-13": "generic / formal one (one should be careful)",
    }
    for cid, desc in _llm.items():
        dets.append(LLMStandaloneDetector(
            cid, f"Decide whether the sentence contains: {desc}. Return JSON "
            f'{{"construct_id":"{cid}"|"NONE","confidence":0..1,'
            f'"rationale":"..."}}.', client=client,
            version=f"{cid.lower()}-llm@0.1"))
    return dets
