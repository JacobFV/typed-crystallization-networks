"""The language registry: which languages an episode can be read in.

The curriculum's lessons are about language, so the language an episode is
presented in is a **parameter of the resource**, not a formatting detail. It
defaults to English:

    >>> import langcurriculum as lc
    >>> lc.DEFAULT_LANGUAGE
    'english'
    >>> lc.get("symbol_grounding").example(0).prompt.splitlines()[0]
    'In the scene: o0 is a yellow cube at (4, 8); o1 is a yellow prism at (4, 7);
    o2 is a green disc at (3, 8) and o3 is a blue cone at (2, 1).'

Three packs ship:

===================  ========  =====================================================
code                 kind      what it is
===================  ========  =====================================================
``english``          natural   prose; the default
``spanish``          natural   gender agreement, post-nominal adjectives, ¿…?
``chinese``          natural   measure words, 的 modification, 吗 questions
``english_synonym``  natural   English whose question uses held-out near-synonyms
``symbols``          formal    s-expression notation (alias: ``invented``)
===================  ========  =====================================================

Each natural pack ships its own vocabulary — 321 typed open-class entries with
the morphology that language needs — and states, in ``grammar_notes``, both what
it implements and what it deliberately leaves as a labelled row.

Adding one is a matter of supplying words. See
:mod:`langcurriculum.languages.base` for the two objects a pack consists of and
a worked sketch; the realizer's generic rules are stated in shape-neutral terms
so a new pack overrides word order and morphology rather than the walk itself.
"""

from __future__ import annotations

from .base import FormalLanguage, Language, Lexicon
from .lexicon import Adjective, Noun, Verb, Vocabulary
from .symbols import Symbols

__all__ = ["Language", "Lexicon", "FormalLanguage",
           "Vocabulary", "Noun", "Adjective", "Verb",
           "LANGUAGES", "LANGUAGE_ALIASES", "DEFAULT_LANGUAGE",
           "get_language", "register_language", "language_codes", "languages"]

#: the language every entry point uses when none is named
DEFAULT_LANGUAGE = "english"

LANGUAGES: dict[str, Language] = {}

#: Languages built on demand from the database. Kept apart from
#: :data:`LANGUAGES` on purpose: that dict is what the package *declares* it
#: has, and a cache must not change a declaration. Caching into it made
#: ``language_codes()`` grow with whatever a caller happened to ask for, so
#: the CLI and the site listed a different set depending on what had been
#: touched first, and a test iterating it covered a different set each run.
_DERIVED: dict[str, Language] = {}

#: other names the same pack answers to
LANGUAGE_ALIASES: dict[str, str] = {
    # the notation is where per-episode invented vocabulary is seen unadorned
    "invented": "symbols",
    "sexpr": "symbols",
}


def register_language(language: Language, *, aliases: tuple[str, ...] = ()) -> Language:
    """Add a language pack to the registry. Returns it, so it can be assigned."""
    if not language.code:
        raise ValueError("a language pack needs a code")
    LANGUAGES[language.code] = language
    for a in aliases:
        LANGUAGE_ALIASES[a] = language.code
    return language


def get_language(code: str | Language | None = None) -> Language:
    """The pack for a code, resolving aliases. ``None`` gives the default.

    Falls through to the grammar registry, which can build a language for any
    of the thousand-odd codes the language database covers. Those are
    constructed on demand and cached: instantiating them all at import time to
    serve a caller who wants one would be indefensible, and most callers want
    one.
    """
    if isinstance(code, Language):
        return code
    name = (code or DEFAULT_LANGUAGE).strip().lower()
    name = LANGUAGE_ALIASES.get(name, name)
    if name in LANGUAGES:
        return LANGUAGES[name]
    if name in _DERIVED:
        return _DERIVED[name]
    derived = _derived_language(name)
    if derived is not None:
        return derived
    raise ValueError(
        f"unknown language {code!r}; {len(LANGUAGES)} are registered eagerly "
        f"({sorted(LANGUAGES)}) and the language database covers "
        f"{_n_derived()} more by ISO 639-3 code")


def _derived_language(code: str) -> Language | None:
    """Build a grammar-backed language from the database, once, and cache it."""
    from ..grammar.adapter import GrammarLanguage
    from ..grammar.registry import REGISTRY

    if code not in REGISTRY.available:
        return None
    language = GrammarLanguage(REGISTRY.get(code))
    _DERIVED[code] = language
    return language


def _n_derived() -> int:
    from ..grammar.registry import REGISTRY
    try:
        return len(REGISTRY.available)
    except Exception:                                   # pragma: no cover
        return 0


def language_codes() -> list[str]:
    """Every registered code, in registration order with the default first.

    Registration order is a reading order — the natural languages, then the
    held-out-vocabulary variant, then the notation — which is more use than
    alphabetical when the list is printed.
    """
    rest = [c for c in LANGUAGES if c != DEFAULT_LANGUAGE]
    return [DEFAULT_LANGUAGE, *rest]


def languages() -> list[dict[str, str]]:
    """One plain-data entry per registered language."""
    return [LANGUAGES[c].info() for c in language_codes()]


def _register_grammar_languages() -> None:
    """Register every hand-written grammar as a language.

    All five go through the grammar engine. There is no template realizer left
    to fall back to and no code path that treats English differently from
    Swahili — which was the point of the rewrite, and is the only way to know
    the engine really carries the languages it claims to.
    """
    from ..grammar.adapter import GrammarLanguage
    from ..grammar.grammars import GRAMMARS

    for code in ("english", "spanish", "chinese", "turkish", "swahili",
                 "english_synonym"):
        grammar = GRAMMARS.get(code)
        if grammar is not None:
            register_language(GrammarLanguage(grammar))


_register_grammar_languages()
register_language(Symbols())
