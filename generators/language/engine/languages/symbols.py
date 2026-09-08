"""The formal notation: the same episode as an s-expression.

This is the compact, fully-parenthesized form the structure is built in. It is
not prose and does not pretend to be — it is here because for some work it is
the better interface:

* **it is exact.** Nothing is smoothed over by a realizer, so a structure that
  renders ambiguously in prose is unambiguous here;
* **it is short.** For a lesson whose episode carries forty facts, the notation
  is a fraction of the tokens of the English;
* **it is what invented vocabulary looks like raw.** Many lessons coin words
  per episode — three-letter nonce forms, four-letter aliases, freshly-named
  predicates — and this view shows them without the English scaffolding that
  would otherwise carry some of the meaning for free.

That last point is why the pack is also registered under the alias ``invented``.
The invented vocabulary itself is a property of the *lessons*, not of this pack:
``lexicon_induction`` coins three new words per episode in English too. What the
notation adds is that the frame around those words is a notation the learner
also has to read, rather than English it may already know.
"""

from __future__ import annotations

from .._structure import Term, sexpr
from .base import FormalLanguage, Lexicon

__all__ = ["Symbols", "SYMBOL_LEXICON"]

SYMBOL_LEXICON = Lexicon(
    instruction="Answer with exactly one of: {choices}\nReply with the answer only.",
    options_heading="Options:",
)


class Symbols(FormalLanguage):
    """S-expression notation."""

    code = "symbols"
    name = "Symbolic notation"
    kind = "formal"
    description = ("The structure as an s-expression: exact, compact, and the raw "
                   "view of per-episode invented vocabulary.")
    lexicon = SYMBOL_LEXICON

    def render(self, term: Term) -> str:
        return sexpr(term)
