"""The abstract lexicon: which construction each predicate head is.

A frame is the answer to one question — *when the curriculum writes*
``(requires compiler parser)``, *what is being said?* The answer is: a relational
predication, with the compiler as agent and the parser as patient. Not "argument
0 goes before the words and argument 1 after", which is what a format string
says and which is true only of English.

This is the file that makes language N+1 cheap. Once a head is known to be a
relational predication with those two roles, every grammar in the package
already knows how to say it — SVO or SOV, case-marked or positional, agreeing or
not — without anyone writing a Turkish or Swahili line for that head. The
473-line English template table shrinks to this, and this is written once.

Coverage and the long tail
--------------------------

The curriculum uses **467 distinct predicate heads across 714 sites**, and the
distribution is brutally long-tailed: the commonest head appears 26 times
and the average head 1.5. Hand-writing 399 frames would be the same mistake as
hand-writing 399 templates, one level up.

So frames come from two places. :data:`FRAMES` names the heads that carry real
weight or that would be misanalysed by default. Everything else is **inferred
from arity** by :func:`frame_for`, which is what the old realizer's generic path
did — but role-labelled, so the inference feeds a grammar rather than a
concatenation. An unrecognised binary predicate is a relational predication with
an agent and a patient, and that is right far more often than it is wrong.

Where the inference *is* wrong, the failure is visible rather than silent: the
sentence reads as a labelled data row, which is what the previous system
produced for these heads anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .category import (
    AGENT, ATTRIBUTE, GOAL, INDEX, LOCATION, PATIENT, RECIPIENT, SOURCE,
    THEME, VALUE,
)

__all__ = ["Frame", "FRAMES", "frame_for", "coverage", "REPORTING", "SPATIAL_FRAME"]

#: Predicates whose complement the speaker did not witness. This is not a guess
#: about the world — it is read straight off the structure: a proposition inside
#: ``(says alice P)`` is, by construction, reported rather than observed.
#:
#: It matters because Turkish, Quechua, Tariana and a few hundred other
#: languages **oblige** a speaker to mark how they know what they assert. A
#: grammar for one of them cannot render an unmarked clause at all, and picking
#: a marker at random would make the Turkish episode assert something the
#: English one does not — a benchmark-integrity failure, not a wording one.
REPORTING = frozenset({
    "says", "said", "claims", "claim", "reports", "told", "asserted",
    "denied", "testifies", "alleges", "kb_fact", "kb_rule", "observed",
})

#: Predicates that locate one thing relative to another **from the speaker's
#: point of view**. Guugu Yimithirr and Tzeltal have no such frame: they locate
#: things by compass bearing, and "to the left of the cube" is not expressible
#: without knowing which way the scene faces. The curriculum does not record
#: that, so a grammar for an absolute-frame language must refuse rather than
#: invent a bearing. Marking the frame is what lets it refuse.
SPATIAL_FRAME = frozenset({"left_of", "right_of", "front_of", "behind"})


@dataclass(frozen=True)
class Frame:
    """What construction a predicate head realizes, and what its arguments are."""

    construction: str
    #: one role per argument, in the order the structure supplies them
    roles: tuple[str, ...] = ()
    #: for Indexed, what to call the ordinal ("step", "round", "trial")
    kind: str = ""
    #: a fixed lexical head, where the construction needs one
    lemma: str = ""

    @property
    def arity(self) -> int:
        return len(self.roles)


def _f(construction: str, *roles: str, kind: str = "", lemma: str = "") -> Frame:
    return Frame(construction, tuple(roles), kind, lemma)


# ======================================================================
# the heads that carry real weight, or that default inference gets wrong
# ======================================================================
FRAMES: Mapping[str, Frame] = {
    # ---- scene description ------------------------------------------
    "obj/5": _f("ObjFull", AGENT, ATTRIBUTE, VALUE, "x", "y"),
    "obj/4": _f("ObjFull", AGENT, ATTRIBUTE, "x", "y"),
    "obj/3": _f("ObjKind", AGENT, ATTRIBUTE, VALUE),
    "color/2": _f("PredAttr", AGENT, ATTRIBUTE),
    "color/1": _f("Labelled", ATTRIBUTE),
    "shape/2": _f("PredIdent", AGENT, VALUE),
    "shape/1": _f("Labelled", VALUE),
    "prop/2": _f("PredAttr", AGENT, ATTRIBUTE),
    "at/1": _f("Labelled", LOCATION),

    # ---- classification and identity ---------------------------------
    "isa/2": _f("PredIdent", AGENT, VALUE),
    "is_a/2": _f("PredIdent", AGENT, VALUE),
    "inst/2": _f("PredIdent", AGENT, VALUE),
    "entity/2": _f("PredIdent", AGENT, VALUE),
    "type/1": _f("Labelled", VALUE),
    #: note the reversed roles: (fact KIND WHO) means "who is a kind"
    "fact/2": _f("PredIdentRev", VALUE, AGENT),

    # ---- relational predication --------------------------------------
    "requires/2": _f("PredRel", AGENT, PATIENT),
    "provides/2": _f("PredRel", AGENT, PATIENT),
    "causes/2": _f("PredRel", AGENT, PATIENT),
    "precedes/2": _f("PredRel", AGENT, PATIENT),
    "feeds/2": _f("PredRel", AGENT, PATIENT),
    "supports/2": _f("PredRel", AGENT, PATIENT),
    "attacks/2": _f("PredRel", AGENT, PATIENT),
    "contradicts/2": _f("PredRel", AGENT, PATIENT),
    "entails/2": _f("PredRel", AGENT, PATIENT),
    "implies/2": _f("PredRel", AGENT, PATIENT),
    "means/2": _f("PredRel", AGENT, PATIENT),
    "has/2": _f("PredRel", AGENT, PATIENT),
    "holds/2": _f("PredRel", AGENT, PATIENT),
    "parent/2": _f("PredRel", AGENT, PATIENT),
    "predicts/2": _f("PredRel", AGENT, PATIENT),
    "observed/2": _f("PredRel", AGENT, PATIENT),
    "says/2": _f("PredRel", AGENT, PATIENT),
    "claims/2": _f("PredRel", AGENT, PATIENT),
    "bind/2": _f("PredRel", AGENT, PATIENT),
    "set/2": _f("PredRel", AGENT, VALUE),
    "value/2": _f("PredRel", AGENT, VALUE),
    "cost/2": _f("PredRel", AGENT, VALUE),
    "bits/2": _f("PredRel", AGENT, VALUE),
    "weight/1": _f("Labelled", VALUE),
    "macro/2": _f("PredRel", AGENT, PATIENT),

    # ---- three-place --------------------------------------------------
    "give/3": _f("PredRel3", AGENT, RECIPIENT, THEME),
    "predicts/3": _f("PredRel3", AGENT, PATIENT, VALUE),
    "kb_fact/3": _f("Reports", SOURCE, AGENT, VALUE),
    "kb_rule/3": _f("Reports", SOURCE, AGENT, PATIENT),
    "needs/3": _f("PredRel3", AGENT, PATIENT, VALUE),

    # ---- spatial: relative frame, which not every language has ---------
    "left_of/2": _f("Spatial", ATTRIBUTE, VALUE, lemma="left_of"),
    "right_of/2": _f("Spatial", ATTRIBUTE, VALUE, lemma="right_of"),
    "above/2": _f("Spatial", ATTRIBUTE, VALUE, lemma="above"),
    "below/2": _f("Spatial", ATTRIBUTE, VALUE, lemma="below"),

    # ---- packaging ------------------------------------------------------
    "ex/2": _f("Mapping", SOURCE, GOAL),
    "example/2": _f("Mapping", SOURCE, GOAL),
    "rule/2": _f("PredRel", AGENT, PATIENT),
    "rule/3": _f("IndexedCond", INDEX, "consequent", "antecedent", kind="rule"),
    "candidate/2": _f("Labelled", "label", VALUE),
    "candidate/1": _f("Bare", VALUE),
    "claim/1": _f("Bare", VALUE),
    "argument/1": _f("Bare", VALUE),
    "task/1": _f("Bare", VALUE),
    "formula/2": _f("Labelled", "label", VALUE),
    "theory/2": _f("Labelled", "label", VALUE),
    "schema/2": _f("Labelled", "label", VALUE),
    "equation/2": _f("Labelled", "label", VALUE),
    "leaf/1": _f("Labelled", VALUE),

    # ---- indexed rows ---------------------------------------------------
    # An unlabelled step -- the program display in program_explanation, and
    # every description step. Without it the binary inference read `step` as a
    # relational predication and rendered its head as a verb: Polish
    # "0 kroczyć ylittää 2", the verb *to walk*.
    "step/2": _f("Indexed", INDEX, VALUE, kind="step"),
    "step/3": _f("Indexed", INDEX, AGENT, VALUE, kind="step"),
    "step/4": _f("Indexed", INDEX, AGENT, "rel", VALUE, kind="step"),
    "round/1": _f("Labelled", INDEX, kind="round"),
    "round/3": _f("Indexed", INDEX, AGENT, VALUE, kind="round"),
    "turn/3": _f("Indexed", INDEX, AGENT, VALUE, kind="turn"),
    "vote/3": _f("Indexed", INDEX, AGENT, VALUE, kind="round"),
    "trial/4": _f("Indexed", INDEX, AGENT, "rel", VALUE, kind="trial"),
    "obs/3": _f("Indexed", INDEX, AGENT, VALUE, kind="block"),
    "do/3": _f("Indexed", INDEX, AGENT, VALUE, kind="block"),

    # ---- quantification and comparison ----------------------------------
    "quant/2": _f("Quant", "q", ATTRIBUTE),
    #: a quantified claim with all four of its parts: quantifier, polarity,
    #: restriction and scope. Generators used to render this to an English
    #: string themselves, which no other language could then translate.
    "nl_claim/4": _f("NLClaim", "q", "pol", "restriction", "scope"),
    #: a quantified *transitive* clause — "every agent read a book" — whose two
    #: quantifiers are the whole point of the lesson that uses it
    "nl_transitive/5": _f("NLTransitive", "q", AGENT, "rel", "q2", PATIENT),
    "lt/2": _f("Compare", AGENT, PATIENT, lemma="lt"),
    "gt/2": _f("Compare", AGENT, PATIENT, lemma="gt"),
    "le/2": _f("Compare", AGENT, PATIENT, lemma="le"),
    "ge/2": _f("Compare", AGENT, PATIENT, lemma="ge"),
    "eq/2": _f("Compare", AGENT, PATIENT, lemma="eq"),
    "neq/2": _f("Compare", AGENT, PATIENT, lemma="neq"),

    # ---- logical --------------------------------------------------------
    # The eight list operations, described rather than named. A description
    # was an English sentence inside a `Str` and reached every language
    # unchanged; as a construction the grammar builds it, so Finnish and
    # Turkish put the verb last and German agrees its plural.
    "desc_take/1": _f("Operation"),
    "desc_drop_first/1": _f("Operation"),
    "desc_keep_greater/1": _f("Operation"),
    "desc_keep_even/1": _f("Operation"),
    "desc_sort/1": _f("Operation"),
    "desc_reverse/1": _f("Operation"),
    "desc_add/1": _f("Operation"),
    "desc_mul/1": _f("Operation"),
    "and": _f("Coord", lemma="and"),
    "or": _f("Coord", lemma="or"),
    "not/1": _f("Neg", "inner"),
    "neg/1": _f("Neg", "inner"),
}

#: roles assigned to an unrecognised head, by arity. The inference the old
#: generic path made implicitly, now written down and role-labelled.
_BY_ARITY: Mapping[int, Frame] = {
    0: _f("Bare"),
    1: _f("Labelled", VALUE),
    2: _f("PredRel", AGENT, PATIENT),
    3: _f("Enumerated", VALUE, VALUE, VALUE),
    4: _f("Enumerated", VALUE, VALUE, VALUE, VALUE),
}


def frame_for(head: str, arity: int) -> Frame:
    """The frame for a head, looked up by ``head/arity`` then ``head``, then inferred.

    The inference is not a failure mode. It is the design: a long tail of heads
    used once or twice each does not repay hand-analysis, and a role-labelled
    default feeds every grammar correctly even when nobody has thought about
    that particular predicate.
    """
    for key in (f"{head}/{arity}", head):
        frame = FRAMES.get(key)
        if frame is not None and (not frame.roles or frame.arity == arity
                                  or frame.construction in ("Coord", "Bare")):
            return frame
    if arity in _BY_ARITY:
        return _BY_ARITY[arity]
    return Frame("Enumerated", tuple([VALUE] * arity))


def coverage(heads: Sequence[tuple[str, int]]) -> dict[str, float]:
    """How much of a head inventory is named rather than inferred.

    Reported honestly rather than assumed: a curriculum that grows new lessons
    grows new heads, and the number that matters is what fraction of *sites*
    hit an authored frame, not what fraction of the table is filled in.
    """
    named = sum(1 for h, a in heads
                if f"{h}/{a}" in FRAMES or h in FRAMES)
    total = len(heads) or 1
    return {"named": named, "inferred": total - named,
            "fraction_named": round(named / total, 4)}
