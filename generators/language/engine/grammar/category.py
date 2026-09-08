"""The category inventory and the feature vocabulary every grammar draws on.

Two decisions in this module do most of the typological work, and both are
choices about *generality*, made once here so that no individual grammar has to
re-make them.

**Noun class, not gender.** The feature is :data:`CLS`, and Spanish masculine is
just the value ``"m"``. Swahili's eighteen classes are values ``"1"`` … ``"18"``
of the same feature, agreed by the same unification, consumed by the same
concord rules. A grammar with no noun class simply never mentions it. Had this
been called ``gender`` and given the values ``m``/``f``, Bantu would have needed
a parallel mechanism — which is exactly the mistake the previous realizer made
and the reason it could not be extended.

**Role, not position.** An argument arrives labelled :data:`AGENT` or
:data:`PATIENT`, never "argument 0". English realizes an agent by putting it
before the verb; Turkish by leaving it unmarked and putting it first; Japanese
by suffixing ``が``; Hindi by suffixing ``ne`` in the perfective and nothing
otherwise; Basque by marking it ergative when the clause is transitive and
absolutive when it is not. None of those are expressible over a positional slot,
and all of them are expressible over a role.

The categories themselves follow the resource-grammar tradition — ``NP``, ``CN``,
``AP``, ``VP``, ``Cl``, ``Utt`` — because that inventory is the one with forty
languages of evidence behind it. ``CN`` (a common noun with its modifiers but no
determiner) earns its place separately from ``NP``: it is the constituent that
adjectives attach to and that a determiner or a numeral-plus-classifier scopes
over, and languages differ on all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .features import EMPTY, FS

__all__ = [
    "Cat", "UTT", "S", "CL", "QCL", "NP", "CN", "N", "AP", "A", "VP", "V",
    "PP", "ADV", "DET", "CARD", "CONJ", "TEXT", "SYM",
    "NUM", "PERS", "CLS", "CASE", "DEF", "POL", "TENSE", "ASPECT", "MOOD",
    "EVID", "HONOR", "CLF", "ROLE", "HUMAN", "DEGREE",
    "SG", "PL", "DUAL", "AGENT", "PATIENT", "RECIPIENT", "THEME", "LOCATION",
    "ATTRIBUTE", "VALUE", "INDEX", "SOURCE", "GOAL", "ROLES",
]


# ======================================================================
# feature names
# ======================================================================
#: grammatical number. ``sg``/``pl``, plus ``dual`` for Arabic and Slovene.
NUM = "num"
#: person: 1, 2, 3.
PERS = "pers"
#: **noun class**. ``m``/``f``/``n`` in Indo-European; ``1``…``18`` in Bantu;
#: absent in Turkish and Chinese. One feature, because it is one phenomenon.
CLS = "cls"
#: case. Open-valued: a grammar declares the inventory it actually has.
CASE = "case"
#: definiteness: ``def``, ``indef``, ``bare``.
DEF = "def"
#: polarity: ``pos``, ``neg``.
POL = "pol"
TENSE, ASPECT, MOOD = "tense", "aspect", "mood"
#: evidentiality — how the speaker came to know. Obligatory in Turkish and
#: Quechua, which means a generator that does not record it cannot be
#: translated into them without inventing evidence.
EVID = "evid"
#: addressee honorification: ``plain``, ``polite``, ``humble``.
HONOR = "honor"
#: the measure word a noun requires to be counted or pointed at.
CLF = "clf"
#: the semantic role an argument fills. See the module docstring.
ROLE = "role"
#: animacy, which several case and agreement systems condition on.
HUMAN = "human"
#: degree of comparison. Unmapped, the comparative looked exactly like the
#: plain form and Swedish answered *gulare* "yellower" for *gul*.
DEGREE = "degree"

SG, PL, DUAL = "sg", "pl", "dual"

# ---- semantic roles ---------------------------------------------------
AGENT = "agent"
PATIENT = "patient"
RECIPIENT = "recipient"
THEME = "theme"
LOCATION = "location"
#: the property predicated of something — the ``red`` of "the cube is red"
ATTRIBUTE = "attribute"
#: the filler of a labelled row — the ``3`` of "weight: 3"
VALUE = "value"
#: an ordinal that situates the clause — the ``4`` of "step 4: …"
INDEX = "index"
SOURCE, GOAL = "source", "goal"

#: every role the frame lexicon may assign, for validation
ROLES = frozenset({AGENT, PATIENT, RECIPIENT, THEME, LOCATION, ATTRIBUTE,
                   VALUE, INDEX, SOURCE, GOAL})


# ======================================================================
# categories
# ======================================================================
@dataclass(frozen=True)
class Cat:
    """A syntactic category, optionally constrained by features.

    ``NP`` on its own is any noun phrase; ``NP.with_(case="acc")`` is an
    accusative one. A construction states what it needs and the linearizer
    unifies, so a grammar that has no cases never notices the constraint.
    """

    name: str
    features: FS = EMPTY

    def with_(self, **kw: Any) -> "Cat":
        return Cat(self.name, self.features.but(**kw))

    def __repr__(self) -> str:
        if not self.features:
            return self.name
        inner = ", ".join(f"{k}={v!r}" for k, v in self.features.items())
        return f"{self.name}[{inner}]"

    def matches(self, other: "Cat") -> bool:
        """Same category, and features that can be made consistent."""
        from .features import unify
        return self.name == other.name and unify(self.features, other.features) is not None


def _cat(name: str) -> Cat:
    return Cat(name)


#: a complete utterance — what a learner reads as one line
UTT = _cat("Utt")
#: a declarative sentence
S = _cat("S")
#: a clause, tense and polarity not yet fixed
CL = _cat("Cl")
#: an interrogative clause
QCL = _cat("QCl")
#: a noun phrase: determiner scope resolved, ready to fill an argument slot
NP = _cat("NP")
#: a common noun with its modifiers and no determiner — what ``Det`` scopes over
CN = _cat("CN")
N = _cat("N")
#: an adjective phrase
AP = _cat("AP")
A = _cat("A")
VP = _cat("VP")
V = _cat("V")
#: an adposition phrase — prepositional or postpositional, the grammar decides
PP = _cat("PP")
ADV = _cat("Adv")
DET = _cat("Det")
CARD = _cat("Card")
CONJ = _cat("Conj")
#: several utterances in sequence
TEXT = _cat("Text")
#: an opaque symbol — an object id, a nonce form, a number. Never inflected,
#: never translated; it passes through every grammar unchanged.
SYM = _cat("Sym")
