"""What a language is, from the outside.

This module is now only an **interface**. The realizer that used to live here —
a template walker with a strategy method per point of cross-linguistic variation
— has been replaced by :mod:`langcurriculum.grammar`, which does the same job
with a real grammar: feature structures and unification for agreement, induced
paradigms for morphology, and word order read off typological parameters rather
than baked into format strings.

What survives is the seam every caller talks to. A lesson, the CLI, the dataset
exporter and the test suite all want the same three things from a language —
:meth:`Language.render` an episode, :meth:`Language.token` an answer option, and
:meth:`Language.prompt` the two together — and none of them should know or care
whether a grammar, a notation, or something not yet written is behind it.

:class:`Lexicon` likewise survives as a *view*: the flat bundle of closed-class
words and typography that the invariant tests read. Grammars populate it through
:class:`~langcurriculum.grammar.adapter.GrammarLanguage`; nothing writes one by
hand any more.
"""

from __future__ import annotations

import re
import string as _string
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .._structure import PRIMITIVE_TYPES, Term
from .lexicon import Vocabulary

__all__ = ["Lexicon", "Language", "FormalLanguage"]

@dataclass(frozen=True)
class Lexicon:
    """The closed class and the typography, separated from the walk that uses them."""

    # ---- closed class -------------------------------------------------
    definite: str = "the"
    definite_f: str = ""
    definite_pl: str = ""
    definite_fpl: str = ""
    indefinite: str = "a"
    indefinite_f: str = ""
    indefinite_before_vowel: str = ""
    copula_sg: str = "is"
    copula_pl: str = "are"
    negation: str = "not"
    conjunction: str = "and"
    disjunction: str = "or"
    yes: str = "yes"
    no: str = "no"
    of: str = "of"

    # ---- typography ---------------------------------------------------
    #: what separates words. Empty for scripts written without spacing.
    word_joiner: str = " "
    #: whether a sentence starts with a capital letter
    capitalizes: bool = True
    full_stop: str = "."
    question_mark: str = "?"
    #: opening question mark, for languages that use one
    question_open: str = ""
    list_separator: str = ", "
    clause_separator: str = "; "
    #: separator inside a Latin function application, which stays half-width
    #: even in a script that is otherwise full-width: ``t(a, b, c)``
    arg_separator: str = ", "
    colon: str = ":"
    bullet: str = "  - "

    # ---- relational and operator words --------------------------------
    #: Kept because the cross-language invariant tests read them when checking
    #: that a question did not drop an argument. Grammars fill them from their
    #: own relational lexicon; nothing authors one by hand.
    operators: Mapping[str, str] = field(default_factory=dict)
    prepositions: Mapping[str, str] = field(default_factory=dict)
    quantifiers: Mapping[str, str] = field(default_factory=dict)
    prefix_operators: Mapping[str, str] = field(default_factory=dict)

    # ---- framing ------------------------------------------------------
    field_intros: Mapping[str, str] = field(default_factory=dict)
    #: predicate head -> the words that realize it between two arguments
    predicate_words: Mapping[str, str] = field(default_factory=dict)
    instruction: str = "Answer with exactly one of: {choices}\nReply with the answer only."
    instruction_many: str = ("Answer with exactly one of the {n} options listed above.\n"
                             "Reply with the answer only.")
    options_heading: str = "Options:"

    # ---- open class, for the lessons that are about morphology --------
    verbs: Sequence[str] = ()
    intransitive_verbs: Sequence[str] = ()
    adverbs: Sequence[str] = ()
    noun_forms: Sequence[tuple[str, str]] = ()
    agreement_forms: Sequence[tuple[str, str]] = ()
    pronouns: Mapping[str, str] = field(default_factory=dict)
    name_gender: Mapping[str, str] = field(default_factory=dict)
    preposition_words: Sequence[str] = ()
    #: *then* — written into a token list by the coreference lesson, so it has
    #: to come from the pack like everything else in that sentence
    then: str = "then"
    #: the article these lessons put in front of :attr:`noun_forms`, where the
    #: pack keeps it as a separate token rather than inside the noun
    article: str = ""

    # ---- the open-class vocabulary ------------------------------------
    vocabulary: Vocabulary = field(default_factory=Vocabulary)

    # ---- held-out vocabulary -----------------------------------------
    synonyms: Mapping[str, str] = field(default_factory=dict)

    # ---- helpers a pack may override ----------------------------------
    def copula(self, plural: bool = False) -> str:
        return self.copula_pl if plural else self.copula_sg

    def word_for(self, head: str) -> str:
        """A head symbol as words: ``left_of`` -> ``to the left of``."""
        return (self.predicate_words.get(head)
                or self.operators.get(head)
                or self.prepositions.get(head)
                or head.replace("_", " "))


class Language:
    """Base class. A language turns a structure into the text a learner reads."""

    code: str = ""
    name: str = ""
    kind: str = "natural"
    description: str = ""
    #: what the pack claims to realize, and what it leaves as a labelled row
    grammar_notes: tuple[str, ...] = ()
    lexicon: Lexicon = Lexicon()

    def render(self, term: Term) -> str:            # pragma: no cover - abstract
        raise NotImplementedError

    def token(self, value: str) -> str:
        """Render one answer token — an option, or the gold answer.

        The options have to be in the same language as the prompt. If the scene
        says ``rojo`` and the option list says ``red``, the task has quietly
        become "translate, then answer", which is a different and much harder
        lesson than the one being scored. Words the pack does not know — object
        ids, nonce forms, numbers — pass through, which is most of them.
        """
        return value

    def knows(self, value: str) -> bool:
        """Whether this pack has a word of its own for ``value``.

        Asked before an answer set is translated, since a set is rendered whole
        or not at all. The default reads the declared vocabulary; a pack that
        keeps its words somewhere else — a database, say — overrides this.
        Putting the question to one particular kind of lexicon is how four
        hundred languages came to answer "no" to every option they knew.
        """
        return self.lexicon.vocabulary.knows(value)

    def prompt(self, observation: str, choices: Sequence[str], *, max_inline: int = 40) -> str:
        """Assemble what an agent is handed: the episode, then the answer set."""
        lex = self.lexicon
        opts = [str(c) for c in choices]
        if len(opts) <= max_inline:
            tail = lex.instruction.format(choices=" | ".join(opts))
        else:
            listed = "\n".join(f"{lex.bullet}{o}" for o in opts)
            tail = f"{lex.options_heading}\n{listed}\n" + lex.instruction_many.format(n=len(opts))
        return f"{observation}\n\n{tail}"

    def coverage(self) -> dict[str, int]:
        """How much of the curriculum this pack actually has words for.

        Reported rather than the size of a declared word list, because the two
        are not the same question and only one of them is comparable. A pack
        that keeps its words in a database declares nothing at all, and was
        therefore reporting a total of zero while covering two hundred and
        thirty-six of the four hundred and five words the lessons can coin.
        """
        return self.lexicon.vocabulary.counts()

    def info(self) -> dict[str, Any]:
        return {"code": self.code, "name": self.name, "kind": self.kind,
                "description": self.description,
                "vocabulary": self.coverage(),
                "grammar": list(self.grammar_notes)}

    def __repr__(self) -> str:
        return f"<Language {self.code} ({self.kind})>"


class FormalLanguage(Language):
    """A notation rather than a prose language."""

    kind = "formal"


_SENTENCE_END = re.compile(r"[.!?。？！]$")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([?.!,;:])")
