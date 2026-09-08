"""Support for the supplementary syntax and semantics lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from contextvars import ContextVar
from typing import Any, Mapping, Sequence

from .._structure import Ident, Lst, Num, Pred, Term
from ..binding import SlotVocabulary, bind_scene
from ..languages import DEFAULT_LANGUAGE, get_language
from .base import COLORS, SHAPES


#: The language an episode is being generated for. Set by ``Lesson.example``
#: around the call to a generator, because these lessons are *about*
#: morphology and their material has to be the material of the language the
#: episode will be read in.
ACTIVE_LANGUAGE: ContextVar[str] = ContextVar("ACTIVE_LANGUAGE",
                                              default=DEFAULT_LANGUAGE)

#: The English material, and the shape every other language has to match.
#: Read once because English is a fixed pack, and used as the fallback.
_ENGLISH = get_language(DEFAULT_LANGUAGE).lexicon

#: Which lexicon fields are drawn on here, and how many entries each must have.
#: The count is not a style rule. ``rng.choice`` consumes a variable number of
#: bits depending on the length of what it is choosing from, so a pack offering
#: five nouns where English offers six would shift the whole random stream and
#: with it the position of the correct option -- and the curriculum's central
#: cross-language invariant is that the correct option does not move. Parallel
#: tables of equal length keep the stream identical and change only the words.
PARALLEL_FIELDS = {
    "verbs": 6, "intransitive_verbs": 6, "adverbs": 4, "preposition_words": 5,
    "noun_forms": 6, "agreement_forms": 6, "pronouns": 2, "name_gender": 6,
}

#: Not length-checked: a scalar, and legitimately empty. Whether the article
#: stands as its own token is a fact about the pack, not something to guess at.
#: English writes *the farmer* as two tokens; Spanish writes *el granjero* as
#: one because a single article cannot agree with both *la llave* and *el
#: libro*; a derived grammar writes none, having no reliable way to agree.


def _material(field: str):
    """One paradigm table, from the active language if it has a full one.

    All or nothing, and per field. A half-supplied table is worse than none:
    the lesson would present a sentence half in one language and half in
    another, and the learner could not tell which half the question was about.

    This used to be read once at import from the default language and bound to
    module constants, so the portability the docstring above promises never
    happened -- ``example(language="rus")`` got English verbs and English
    nouns, and the whole sentence of an agreement lesson came out in English
    with only its heading translated.
    """
    lexicon = get_language(ACTIVE_LANGUAGE.get()).lexicon
    supplied = getattr(lexicon, field, None) or ()
    if len(supplied) == PARALLEL_FIELDS[field]:
        return supplied
    return getattr(_ENGLISH, field)


def verbs() -> list[str]:
    return list(_material("verbs"))


def intransitive() -> list[str]:
    return list(_material("intransitive_verbs"))


def adverbs() -> list[str]:
    return list(_material("adverbs"))


def prepositions() -> list[str]:
    return list(_material("preposition_words"))


def noun_forms() -> list[tuple[str, str]]:
    """``(singular, plural)`` pairs."""
    return [tuple(x) for x in _material("noun_forms")]


def agree_forms() -> list[tuple[str, str]]:
    """``(third singular, plural)`` verb pairs."""
    return [tuple(x) for x in _material("agreement_forms")]


def gender() -> dict[str, str]:
    return dict(_material("name_gender"))


def pronoun() -> dict[str, str]:
    return dict(_material("pronouns"))


def supplies(field: str) -> bool:
    """Whether the active pack has a full table of its own for ``field``."""
    lexicon = get_language(ACTIVE_LANGUAGE.get()).lexicon
    return len(getattr(lexicon, field, None) or ()) == PARALLEL_FIELDS[field]


def determiner() -> str:
    """The definite article that goes with :func:`noun_forms`.

    Spelled ``"the"`` into the lessons, so even a pack that supplied every
    paradigm would have produced *the granjero por the llaves*.

    It belongs to the same all-or-nothing set as the nouns it stands in front
    of: a pack whose nouns fall back to English gets English's article too,
    because *farmer* preceded by *el* is not a sentence of either language.

    A pack that does supply its own nouns has already written whatever article
    belongs with each of them -- Spanish ships *la llave* and *el libro*,
    because one scalar cannot agree with both -- so nothing is added here. The
    alternative was a single article in front of every noun, which produced
    *der Buch* in German and *ο κλειδιά* in Greek. A wrong article is worse
    than none, and the number contrast the lesson turns on is carried by the
    noun either way.
    """
    lexicon = get_language(ACTIVE_LANGUAGE.get()).lexicon
    if not supplies("noun_forms"):
        return getattr(_ENGLISH, "article", "") or _ENGLISH.definite
    return getattr(lexicon, "article", "") or ''


def then_word() -> str:
    """*then*, the discourse connective the coreference lesson sequences with.

    Gated like the rest. It was not, so a Russian episode read "carol -
    avoided - dave - again - зате́м - he - waited": one Russian word in an
    English sentence, which is the half-and-half the all-or-nothing rule
    exists to prevent.
    """
    if not supplies("verbs"):
        return _ENGLISH.then
    lexicon = get_language(ACTIVE_LANGUAGE.get()).lexicon
    return lexicon.then or _ENGLISH.then


SIZES = SlotVocabulary("size", ["small", "big"])


NONCE_LETTERS = "kmtszlpvr"


def _shuffled(rng: random.Random, xs: Sequence[Any]) -> list[Any]:
    """A shuffled copy. Every answer vocabulary in this module goes through it."""
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _nonce(rng: random.Random, n: int = 3) -> str:
    return "".join(rng.choice(NONCE_LETTERS) for _ in range(n))


def _objects(rng: random.Random, n: int, *, distinct_colors: bool = False,
             distinct_shapes: bool = False) -> list[dict[str, Any]]:
    """``n`` objects with *shuffled* id labels, so an id never predicts a role."""
    ids = _shuffled(rng, [f"o{i}" for i in range(n)])
    colors = rng.sample(COLORS, n) if distinct_colors else [rng.choice(COLORS) for _ in range(n)]
    shapes = rng.sample(SHAPES, n) if distinct_shapes else [rng.choice(SHAPES) for _ in range(n)]
    objs = [{"id": ids[i], "color": colors[i], "shape": shapes[i]} for i in range(n)]
    return bind_scene(objs, rng) or objs


def _obj_list(objs: Sequence[Mapping[str, Any]]) -> Term:
    return Lst([Pred("obj", Ident(o["id"]), Ident(o["color"]), Ident(o["shape"])) for o in objs])


def _yesno(rng: random.Random, truth: bool) -> tuple[list[str], str]:
    return _shuffled(rng, ["yes", "no"]), ("yes" if truth else "no")


def _nest(rng: random.Random, depth: int) -> str:
    """A bracket string whose maximum nesting depth is exactly ``depth``."""
    if depth <= 0:
        return ""
    s = "(" + _nest(rng, depth - 1) + ")"
    if rng.random() < 0.4:
        s = s + "()" if rng.random() < 0.5 else "()" + s
    return s


def _max_depth(s: str) -> int:
    d = best = 0
    for c in s:
        d += 1 if c == "(" else -1
        best = max(best, d)
    return best


_OPS = ("+", "-", "*")


def _expr(rng: random.Random, depth: int) -> tuple[Term, int]:
    if depth <= 0 or rng.random() < 0.25:
        v = rng.randint(1, 5)
        return Num(v), v
    op = rng.choice(_OPS)
    left, lv = _expr(rng, depth - 1)
    right, rv = _expr(rng, depth - 1)
    val = {"+": lv + rv, "-": lv - rv, "*": lv * rv}[op]
    return Pred(op, left, right), val


def _items(objs: Sequence[Mapping[str, Any]]) -> Term:
    return Lst([Pred("item", Ident(o["id"]), Ident(o["color"]), Ident(o["shape"]),
                     Num(o["size"]), Num(o["x"])) for o in objs])


def _formula(rng: random.Random, objs: Sequence[Mapping[str, Any]]) -> tuple[Term, list[str], str]:
    """A random boolean formula over colour/shape, plus the ids it denotes."""
    def prop() -> tuple[Term, Any]:
        if rng.random() < 0.5:
            c = rng.choice(COLORS)
            return Pred("color", Ident(c)), lambda o: o["color"] == c
        s = rng.choice(SHAPES)
        return Pred("shape", Ident(s)), lambda o: o["shape"] == s

    form = rng.choice(["and", "or", "and_not", "not_and", "not"])
    p, fp = prop()
    q, fq = prop()
    if form == "and":
        sym, test = Pred("and", p, q), lambda o: fp(o) and fq(o)
    elif form == "or":
        sym, test = Pred("or", p, q), lambda o: fp(o) or fq(o)
    elif form == "and_not":
        sym, test = Pred("and", p, Pred("not", q)), lambda o: fp(o) and not fq(o)
    elif form == "not_and":
        sym, test = Pred("and", Pred("not", p), q), lambda o: (not fp(o)) and fq(o)
    else:
        sym, test = Pred("not", p), lambda o: not fp(o)
    return sym, [o["id"] for o in objs if test(o)], form
