"""Turkish: the grammar the previous architecture could not have expressed.

Turkish was chosen as the forcing function because it breaks the old realizer in
five independent places, and each break is a design assumption rather than a
missing word.

**Stored forms die.** ``ev`` is *evler, evi, evde, evden, evin, evlerimde,
evlerimizden*, and a noun's paradigm runs to hundreds of cells. The vocabulary
file for this language stores one form per noun — the stem — and the rest is
derived by :class:`~langcurriculum.grammar.morphology.ConcatenativeMorphology`
over four ordered slots, with vowel harmony resolving cyclically at each seam.

**Positional templates die.** Turkish is verb-final, so ``"{0} requires {1}"``
has no analogue: the object precedes the verb, and it is *case-marked* rather
than positioned. ``clause="SOV"`` plus a nominative–accusative
:class:`~langcurriculum.grammar.linearize.Alignment` produces that from the same
abstract tree English reads.

**The gender enum dies** — by being unused. Turkish has no grammatical gender
and no adjective agreement, so :class:`~langcurriculum.grammar.linearize.Concord`
is empty and adjectives are invariant. A design that hard-codes agreement cannot
express *absence* of agreement without a special case.

**The copula dies.** Turkish has no overt present-tense third-person copula:
*küp kırmızı* is "the cube is red" with no word for "is". ``copula_overt=False``.

**Question formation is clitic, not syntactic.** ``mI`` is written as a separate
word but harmonizes with what precedes it — *büyük mü*, *kırmızı mı*, *doğru mu*
— so it needs the phonology, not a lookup table.

Differential object marking
---------------------------

The one genuinely subtle thing implemented here: Turkish marks the accusative
only on *specific* objects. *Kitap okudum* is "I read a book"; *kitabı okudum*
is "I read the book". The case is a function of definiteness, not of position,
which :meth:`Turkish.case_for_object` decides.

Evidentiality
-------------

Turkish obliges a speaker to say how they know what they assert. *Küp kırmızı*
claims the cube is red on the speaker's own authority; *küp kırmızıymış* claims
it on someone else's. There is no neutral form, so a grammar has to choose, and
choosing at random would make the Turkish episode assert something the English
one does not.

The choice is not guessed. A proposition embedded under *says*, *claims* or
*reports* is second-hand **by construction**, so the compiler reads the evidence
source off the structure and marks the complement; see
:data:`~langcurriculum.grammar.frames.REPORTING`. An unembedded scene
description is presented to the learner as given, and takes the direct form.

Not attempted
-------------

The inferential and mirative uses of ``-mIş``, aspect, and the past-tense
evidential contrast between ``-DI`` and ``-mIş``. Those need a temporal
structure the curriculum does not carry.
"""

from __future__ import annotations

from ..category import CASE, DEF, EVID, N, NUM, PL, SG
from ..features import EMPTY, FS, unify
from ..linearize import (
    NOM_ACC, Alignment, Concord, Grammar, Typography, WordOrder,
)
from ..morphology import (
    TURKISH_PHONOLOGY, Affix, ConcatenativeMorphology, Slot,
)
from ..syntax import Node
from .vocab import VocabularyGrammar

__all__ = ["Turkish"]


# ======================================================================
# nominal morphology: stem - plural - possessive - case
# ======================================================================
def _nominal_morphology(spec: dict) -> ConcatenativeMorphology:
    """Build the noun's affix slots from the data file.

    The *inventory* is data — ``lAr``, ``(y)I``, ``DA`` are a list of morphemes
    somebody looked up — while the *phonology* that resolves them stays in
    :mod:`~langcurriculum.grammar.morphology`, because harmony is an algorithm.
    Slots stack outward from the stem in the order the file gives, which is not
    a convention but a fact about the language: *ev-ler-im-de* is well formed
    and no permutation of those suffixes is.
    """
    return ConcatenativeMorphology(
        phonology=TURKISH_PHONOLOGY,
        slots=[Slot(name=slot["name"], order=slot["order"],
                    prefix=bool(slot.get("prefix")),
                    affixes=tuple(Affix(a["form"], FS(a.get("when") or {}),
                                        a.get("priority", 0))
                                  for a in slot["affixes"]))
               for slot in spec.get("slots", ())])


class Turkish(VocabularyGrammar):
    """Turkish. Agglutinative, verb-final, case-marking, harmonizing."""

    code = "turkish"
    name = "Turkish (Türkçe)"
    pack = "turkish"
    iso = "tur"

    order = WordOrder(
        clause="SOV",              # object before verb
        adj="AN",                  # kırmızı küp
        det="DN",                  # bir küp
        numeral="NumN",            # üç kitap
        adposition="post",         # postpositions, not prepositions
        possessive="GN",           # evin değeri — possessor first, genitive
        label="LV",
        conditional="AC",          # eğer B ise, A
        wh_fronting=False,         # wh stays in situ, preverbally
        copula_overt=False,        # no present-tense 3sg copula at all
        numeral_forces_plural=False,   # üç kitap, never üç kitaplar
        negation="post",           # küp kırmızı değil
    )
    typography = Typography(question_mark="?", label_separator=":")
    alignment = Alignment(case_of=NOM_ACC)
    #: nothing agrees with anything inside the Turkish noun phrase
    concord = Concord()

    notes = (
        "SOV, postpositional, possessor-first",
        "agglutinative: plural, possessive and case derived, not stored",
        "four-way vowel harmony resolved cyclically at each morpheme seam",
        "consonant assimilation and intervocalic softening (kitap → kitabı)",
        "no grammatical gender and no adjective agreement",
        "zero copula in the present third person (küp kırmızı)",
        "polar questions by the harmonizing clitic mI",
        "differential object marking: accusative only on specific objects",
        "evidentiality: -(y)mIş on a reported proposition, from the structure "
        "rather than guessed",
        "NOT attempted: the inferential/mirative uses of -mIş, aspect, and "
        "the past-tense evidential contrast (-DI vs -mIş)",
    )

    def __init__(self) -> None:
        super().__init__()
        self.morphology[N.name] = _nominal_morphology(
            self.raw.get("morphology") or {})

    # ---- differential object marking -------------------------------------
    def case_for_object(self, definite: bool) -> str:
        """Accusative if the object is specific, bare nominative otherwise.

        *Kitap okudum* "I read a book" / *kitabı okudum* "I read the book". The
        contrast is carried entirely by the case suffix, so a grammar that always
        marked the object would make every indefinite reading impossible.
        """
        return "acc" if definite else "nom"

    def _arg(self, node: Node, role: str, ctx: FS, *, transitive: bool) -> str:
        if role == "patient" and transitive:
            case = self.case_for_object(node.feats.get_atom("det", "bare") != "indef")
            return self.lin(node, ctx.but(**{CASE: case}, role=role))
        return super()._arg(node, role, ctx, transitive=transitive)

    # ---- evidentiality ----------------------------------------------------
    def lin_PredAttr(self, node: Node, ctx: FS) -> str:
        """A reported proposition takes ``-(y)mIş``; a witnessed one takes nothing.

        Turkish obliges the speaker to say how they know. *Küp kırmızı* asserts
        the cube is red on the speaker's own authority; *küp kırmızıymış* says
        so on someone else's. There is no neutral form, so a grammar that
        ignored the distinction would have to pick one and would be asserting
        something the English episode does not.

        The feature is not invented here: the compiler sets it from the
        structure, because a proposition embedded under *says* is reported by
        construction. Where the structure does not say — a bare scene
        description — the direct form is correct, since the episode presents it
        as given.
        """
        return self._evidential(super().lin_PredAttr(node, ctx), node, ctx)

    def lin_PredIdent(self, node: Node, ctx: FS) -> str:
        return self._evidential(super().lin_PredIdent(node, ctx), node, ctx)

    def _evidential(self, text: str, node: Node, ctx: FS) -> str:
        merged = unify(node.feats, ctx)
        feats = merged[0] if merged else node.feats
        if feats.get_atom(EVID) != "reported" or not text:
            return text
        harmony = TURKISH_PHONOLOGY.harmony
        assert harmony is not None
        # -(y)mIş attaches directly: büyükmüş, kırmızıymış. The buffer y appears
        # only to break hiatus after a vowel; a space would make it a separate
        # word, which it is not.
        suffix = harmony.resolve("mIş", text)
        return text + ("y" if text[-1] in "aeıioöuü" else "") + suffix

    # ---- the harmonizing question clitic ---------------------------------
    def lin_YNQ(self, node: Node, ctx: FS) -> str:
        """``mI`` is a separate orthographic word that harmonizes with the last
        one: *büyük mü*, *kırmızı mı*, *doğru mu*, *evde mi*.

        It cannot be a lookup, because what it harmonizes to is whatever the rest
        of the sentence happened to end with — which is only known here.
        """
        body = node.arg("body")
        assert body is not None
        inner = self.lin(body, ctx)
        harmony = TURKISH_PHONOLOGY.harmony
        assert harmony is not None
        clitic = harmony.resolve(self.cw("q_particle", "mI"), inner)
        return f"{inner} {clitic}"

    # ---- possession is marked on both ends -------------------------------
    def lin_Possess(self, node: Node, ctx: FS) -> str:
        """*evin değeri* — genitive on the possessor, 3sg possessive on the
        possessum. Marking only one end, as English and Spanish do, is
        ungrammatical here.
        """
        er, ed = node.arg("possessor"), node.arg("possessed")
        assert er is not None and ed is not None
        possessor = self.lin(er, ctx.but(**{CASE: "gen"}))
        possessed = self.lin(ed, ctx.but(poss="3sg"))
        return self.join([possessor, possessed])

    # ---- location is a case, not a preposition ---------------------------
    def lin_PredLoc(self, node: Node, ctx: FS) -> str:
        """*küp masada* — the locative is a suffix and the copula is silent."""
        subj, loc = node.arg("agent"), node.arg("location")
        assert subj is not None and loc is not None
        s = self.lin(subj, ctx)
        place = self.lin(loc, ctx.but(**{CASE: "loc"}))
        return self.join([s, place])
