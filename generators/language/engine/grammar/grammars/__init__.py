"""The concrete grammars, and the registry that finds them.

Each module here is one language expressed as parameters plus the overrides its
typology genuinely needs, and the count of overrides is the honest measure of
whether the parameterization is doing its job. Counted from the source rather
than remembered:

======================  ==========  ===========================================
grammar                 overrides   what they are
======================  ==========  ===========================================
:mod:`.swahili`         1           class concord prefixes
:mod:`.english`         4           a/an phonology, and negation, polar
                                    questions and wh-questions all attaching to
                                    a finite auxiliary
:mod:`.spanish`         5           gendered articles with the el-agua
                                    exception, y/e and o/u, ser vs estar, a
                                    label's trailing preposition
:mod:`.turkish`         6           the mI clitic, evidentiality, double-marked
                                    possession, locative case, differential
                                    object marking
:mod:`.chinese`         7           measure words, per-adjective 的,
                                    topic-comment framing, 吗 and 还是 questions
======================  ==========  ===========================================

The counts came down by nine when they were last checked against the source,
and the reason is worth keeping: most of what looked like idiosyncrasy was
**vestigial**. Chinese carried its own ``join_list``, ``join_clauses``,
``sentence`` and ``block_heading``, all four written before
:class:`~langcurriculum.grammar.linearize.Typography` grew ``item_separator``
and ``label_separator``, and all four producing exactly what the base already
produced. Spanish and Chinese each carried a ``lin_Labelled`` whose only
content was a call to ``clean_label`` — a strategy the base declared and then
never invoked, so a grammar could only benefit from it by overriding the whole
method. The base calls it now.

An override that is never needed is worse than a missing parameter: it reads as
evidence that the language is unusual when it is evidence that nobody rechecked.
If a grammar starts needing a dozen, count them from the source before believing
it.
"""

from __future__ import annotations

from ..linearize import Grammar
from ..registry import REGISTRY
from .chinese import Chinese
from .english import English, EnglishSynonym
from .spanish import Spanish
from .swahili import Swahili
from .turkish import Turkish

__all__ = ["GRAMMARS", "get_grammar", "register_grammar", "grammar_codes",
           "English", "EnglishSynonym", "Spanish", "Chinese", "Turkish",
           "Swahili", "HANDWRITTEN"]

#: The hand-written grammars, by the code the curriculum has always used, and
#: by ISO 639-3 so that they shadow the derived grammar for the same language.
#: A verified grammar must always win over a generated one for its own language.
HANDWRITTEN: dict[str, type[Grammar]] = {
    "english": English, "spanish": Spanish, "chinese": Chinese,
    "turkish": Turkish, "swahili": Swahili,
    "english_synonym": EnglishSynonym,
}
_ISO = {"english": "eng", "spanish": "spa", "chinese": "cmn",
        "turkish": "tur", "swahili": "swh"}

GRAMMARS: dict[str, Grammar] = {}


def register_grammar(grammar: Grammar) -> Grammar:
    if not grammar.code:
        raise ValueError("a grammar needs a code")
    GRAMMARS[grammar.code] = grammar
    REGISTRY.register(grammar)
    iso = _ISO.get(grammar.code)
    if iso and iso != grammar.code:
        REGISTRY._handwritten[iso] = grammar
        REGISTRY._available = None
    return grammar


def get_grammar(code: str) -> Grammar:
    key = (code or "").strip().lower()
    if key in GRAMMARS:
        return GRAMMARS[key]
    return REGISTRY.get(key)


def grammar_codes() -> list[str]:
    return sorted(GRAMMARS)


for _cls in (English, Spanish, Chinese, Turkish, Swahili, EnglishSynonym):
    register_grammar(_cls())
