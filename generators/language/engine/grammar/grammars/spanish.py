"""Spanish: gender and number concord, post-nominal adjectives, ¿inverted?

Ported from the template pack it replaces, with every behaviour that pack's
tests asserted kept intact — including the two that are easy to lose.

**Concord is now unification, not a special case.** The template pack agreed
adjectives by calling ``Adjective.agree(gender, plural)`` at each of a hundred
call sites. Here the noun's gender is an inherent feature, the linearizer's
:class:`~langcurriculum.grammar.linearize.Concord` declares that adjectives and
determiners share it, and agreement happens once in the walk. The same
declaration, over an inventory nine times larger, is what makes Swahili work.

**el agua.** A feminine noun beginning with a stressed /a/ takes the *masculine*
article while its adjectives still agree as feminine: *el agua fría*, *las aguas
frías*. It is the one place in Spanish where the article and the adjective
disagree about gender, and a concord system that propagates one feature to both
gets it wrong by construction. The determiner therefore reads the noun's
phonology, not just its class.

**ser and estar.** Identity takes *ser*, location takes *estar*. The engine asks
:meth:`~langcurriculum.grammar.linearize.Grammar.copula` which kind of predication
it is building, so this is three lines rather than a parallel template table.

Not attempted
-------------

Subjunctive, clitic pronouns, agreement across a relative clause. Nothing in the
curriculum's structures needs them and guessing produces exactly the
confident-sounding errors this grammar exists to avoid.
"""

from __future__ import annotations

import re
from typing import Any

from ..category import A, CLS, N, NUM, PL
from ..features import EMPTY, FS
from ..linearize import (
    NO_CASE, Alignment, Concord, Typography, WordOrder,
)
from ..morphology import Morphology
from ..syntax import Node
from .vocab import VocabularyGrammar

__all__ = ["Spanish"]


_I_INITIAL = re.compile(r"^[iíhH]?[ií]", re.IGNORECASE)
_O_INITIAL = re.compile(r"^(h?o)", re.IGNORECASE)



def _pluralize(word: str) -> str:
    """Regular Spanish plural, for a word the vocabulary does not carry."""
    if not word:
        return word
    if word[-1] in "aeiouáéíóú":
        return word + "s"
    if word.endswith("z"):
        return word[:-1] + "ces"
    return word + "es"


class SpanishNoun(Morphology):
    """Plural from the vocabulary, with the regular rule as the fallback."""

    def __init__(self, vocabulary):
        self.by_lemma = {n.lemma: n for n in vocabulary.nouns.values()}

    def inflect(self, lemma: str, feats: FS = EMPTY) -> str:
        if feats.get_atom(NUM) != PL:
            return lemma
        noun = self.by_lemma.get(lemma)
        if noun is not None and noun.plural:
            return noun.plural
        return _pluralize(lemma)

    def forms(self, lemma: str) -> set[str]:
        return {lemma, self.inflect(lemma, FS({NUM: PL}))}


class SpanishAdjective(Morphology):
    """Four agreement forms, selected by the class and number that reach it."""

    def __init__(self, vocabulary):
        self.by_base = {}
        for a in vocabulary.adjectives.values():
            for form in (a.base, a.ms, a.fs, a.mp, a.fp):
                if form:
                    self.by_base.setdefault(form, a)

    def inflect(self, lemma: str, feats: FS = EMPTY) -> str:
        adjective = self.by_base.get(lemma)
        if adjective is None:
            return lemma
        return adjective.agree(feats.get_atom(CLS, "m") or "m",
                               plural=feats.get_atom(NUM) == PL)

    def forms(self, lemma: str) -> set[str]:
        a = self.by_base.get(lemma)
        return {lemma} if a is None else {f for f in (a.base, a.ms, a.fs, a.mp, a.fp) if f}


class Spanish(VocabularyGrammar):
    """Spanish prose, on the grammar engine."""

    code = "spanish"
    name = "Spanish"
    pack = "spanish"
    iso = "spa"

    order = WordOrder(
        clause="SVO", adj="NA", det="DN", numeral="NumN",
        adposition="pre", possessive="NG", label="LV", conditional="CA",
        wh_fronting=True, copula_overt=True, numeral_forces_plural=True,
        negation="pre",
    )
    typography = Typography(question_open="¿", question_mark="?",
                            label_separator=":")
    alignment = Alignment(case_of=NO_CASE)
    #: the declaration that replaces a hundred calls to ``agree()``
    concord = Concord(adjective=(CLS, NUM), determiner=(CLS, NUM),
                      predicative=(CLS, NUM))

    notes = (
        "gender and number concord on articles and adjectives",
        "adjectives follow the noun",
        "ser for identity, estar for location",
        "the el agua exception: masculine article, feminine adjective",
        "inverted opening question mark",
        "y/e before i-/hi- and o/u before o-/ho-",
        "plural from the vocabulary, regular -s/-es/-ces otherwise",
        "NOT attempted: subjunctive, clitic pronouns, agreement across a "
        "relative clause",
    )

    def __init__(self) -> None:
        super().__init__()
        # the article paradigm, the stressed-/a/ list and the prepositions a
        # label may trail are all *lists*, and live in data/spanish.json
        self._articles = {tuple(k.split("|")): v
                          for k, v in (self.raw.get("articles") or {}).items()}
        self._el_agua = frozenset(self.raw.get("el_agua") or ())
        self._trailing = tuple(self.raw.get("trailing_prepositions") or ())
        self.morphology[N.name] = SpanishNoun(self.vocabulary)
        self.morphology[A.name] = SpanishAdjective(self.vocabulary)

    # ---- the article, including the one exception -------------------------
    def determiner(self, kind: str, head: Node | None, feats: FS) -> str:
        """``el``/``la``/``los``/``las`` and ``un``/``una``, agreeing with the noun.

        The *el agua* rule lives here rather than in the concord declaration
        because it is a fact about the article alone: the adjective in the same
        phrase stays feminine. Propagating a single gender feature to both, as
        concord does everywhere else, would produce *el agua frío*.
        """
        if kind not in ("def", "indef"):
            return ""
        gender = feats.get_atom(CLS, "m") or "m"
        plural = feats.get_atom(NUM) == PL
        if gender == "f" and not plural and head is not None:
            if self.word(head.lemma, pos="N").lower() in self._el_agua:
                gender = "m"
        return self._articles.get((kind, gender, "pl" if plural else "sg"), "")

    # ---- coordination -----------------------------------------------------
    def join_list(self, items):
        """``a, b y c`` — with ``y`` becoming ``e`` before *i-* or *hi-*."""
        items = [i for i in items if i]
        if len(items) <= 1:
            return items[0] if items else ""
        conj = "e" if _I_INITIAL.match(items[-1]) else "y"
        return f"{self.typography.list_separator.join(items[:-1])} {conj} {items[-1]}"

    def disjoin(self, items):
        """``siete u ocho`` — ``o`` becomes ``u`` before *o-* or *ho-*."""
        items = [i for i in items if i]
        if len(items) <= 1:
            return items[0] if items else ""
        conj = "u" if _O_INITIAL.match(items[-1]) else "o"
        return f"{self.typography.list_separator.join(items[:-1])} {conj} {items[-1]}"

    # ---- ser and estar ----------------------------------------------------
    def copula(self, kind: str, feats: FS) -> str:
        """*ser* for what a thing is, *estar* for where it is."""
        plural = feats.get_atom(NUM) == PL
        if kind == "loc":
            return "están" if plural else "está"
        return "son" if plural else "es"

    # ---- labels -----------------------------------------------------------
    def clean_label(self, label: str) -> str:
        """Drop a trailing preposition: it needs a complement that is not there."""
        for tail in self._trailing:
            if label.endswith(tail):
                return label[: -len(tail)]
        return label

