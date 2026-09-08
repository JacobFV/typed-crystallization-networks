"""A generative grammar engine for the curriculum's languages.

The curriculum presents every episode in several languages, and until now it did
so by template: a format string per predicate per language, with a small
implicit grammar of four constructions carrying whatever the templates did not.
That works for three languages and stops working at four, because the templates
encode English word order in their slot positions and the vocabulary encodes
morphology as stored forms — neither of which survives contact with a
verb-final, agglutinative, or noun-class language.

This package replaces that with the architecture the problem actually has:

:mod:`~langcurriculum.grammar.features`
    feature structures and unification — the single mechanism that carries
    Spanish gender, Chinese classifiers, Turkish harmony and Bantu concord.

:mod:`~langcurriculum.grammar.category`
    the category inventory, and the feature vocabulary. Noun **class** rather
    than gender, semantic **role** rather than argument position.

:mod:`~langcurriculum.grammar.morphology`
    word formation: paradigm slots, an ordered phonological layer, and the four
    kinds of morphology a language can have — isolating, stored, concatenative,
    templatic.

:mod:`~langcurriculum.grammar.syntax`
    the abstract syntax: eighteen constructions, language-neutral, which is what
    an episode is compiled into before any language is chosen.

:mod:`~langcurriculum.grammar.linearize`
    the walk that turns an abstract tree into a sentence, parameterized by word
    order, alignment and concord.

:mod:`~langcurriculum.grammar.grammars`
    the concrete grammars.

The test of the design is :mod:`~langcurriculum.grammar.grammars.turkish`: a
grammar for a language typologically remote from all three originals, written
as parameters plus four overrides.
"""

from __future__ import annotations

from .category import Cat
from .features import FS, EMPTY, Var, unify
from .linearize import (
    ERG_ABS, NOM_ACC, NO_CASE, Alignment, Concord, Grammar, Typography,
    WordOrder,
)
from .morphology import (
    Affix, ConcatenativeMorphology, Harmony, IsolatingMorphology, Morphology,
    PhonRule, Phonology, Slot, StoredMorphology, TemplaticMorphology,
)
from .syntax import CONSTRUCTIONS, Node, sym

__all__ = [
    "FS", "EMPTY", "Var", "unify", "Cat", "Node", "sym", "CONSTRUCTIONS",
    "Grammar", "WordOrder", "Typography", "Alignment", "Concord",
    "NOM_ACC", "ERG_ABS", "NO_CASE",
    "Morphology", "IsolatingMorphology", "StoredMorphology",
    "ConcatenativeMorphology", "TemplaticMorphology",
    "Harmony", "PhonRule", "Phonology", "Affix", "Slot",
]
