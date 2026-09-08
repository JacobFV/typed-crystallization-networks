"""Support for compositional semantics and language-as-action lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from ..binding import SlotVocabulary
from .._structure import Ident, Lst, Num, Pred, Term
from .base import COLORS

SECTION = "II/III"


PLACES = SlotVocabulary("place", ["kitchen", "garden", "cellar", "attic", "porch", "study"])


LOCATIONS = SlotVocabulary("surface", ["table", "shelf", "box", "floor", "cart", "crate"])


OBJECTS = SlotVocabulary("thing", ["tea", "book", "lamp", "clock", "apple", "coin"])


ACTIONS = SlotVocabulary("intransitive", ["slept", "sang", "cooked", "read", "painted", "waited"])


TRANS_VERBS = SlotVocabulary("transitive", ["likes", "reads", "sells", "paints", "carries", "builds"])


EVENT_NAMES = SlotVocabulary("event", ["storm", "parade", "recital", "eclipse", "auction", "rehearsal",
               "blackout", "harvest"])


SIZES = SlotVocabulary("size", ["tiny", "small", "large", "huge"])


POSITIONS = ["left", "right", "front", "back"]


BASE_KINDS = SlotVocabulary("kind", ["bird", "fish", "robot", "insect", "mammal", "fungus"])


DERIVED_KINDS = ["warm", "flies", "swims", "glows", "hums", "floats"]


ACTIVITIES = SlotVocabulary("activity", ["smoking", "running", "singing", "writing", "fishing", "cycling"])


THINGS = [("hamster", "animal"), ("otter", "animal"), ("magpie", "animal"),
          ("kettle", "tool"), ("spade", "tool"), ("chisel", "tool"),
          ("barge", "vehicle"), ("truck", "vehicle"), ("rose", "plant"),
          ("fern", "plant")]


DIRS = ["north", "east", "south", "west"]


EGO = ["front", "right", "behind", "left"]


DIRVEC = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}


NONCE_LETTERS = "kmtszlpvrn"


def _shuffled(rng: random.Random, xs: Sequence[Any]) -> list[Any]:
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _nonce(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(NONCE_LETTERS) for _ in range(n))


def _nonce_words(rng: random.Random, k: int, length: int = 3,
                 avoid: Sequence[str] = ()) -> list[str]:
    """``k`` distinct nonce words; the whole point of the induction lessons is
    that these have never been seen before and never will be again."""
    out: list[str] = []
    seen = set(avoid)
    while len(out) < k:
        w = _nonce(rng, length)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _world(rng: random.Random) -> tuple[list[str], dict[str, dict[str, str]]]:
    ids = _shuffled(rng, ["a", "b", "c"])
    st = {o: {"color": rng.choice(COLORS), "loc": rng.choice(LOCATIONS)} for o in ids}
    return ids, st


def _world_facts(ids: Sequence[str], st: Mapping[str, Mapping[str, str]]) -> Term:
    return Lst([Pred("obj", Ident(o), Ident(st[o]["color"]), Ident(st[o]["loc"])) for o in ids])


def _rand_action(rng: random.Random, ids: Sequence[str]) -> tuple[str, str, str]:
    verb = rng.choice(["paint", "move"])
    return (verb, rng.choice(list(ids)),
            rng.choice(COLORS if verb == "paint" else LOCATIONS))


def _apply(st: dict[str, dict[str, str]], act: tuple[str, str, str]) -> None:
    verb, obj, val = act
    st[obj]["color" if verb == "paint" else "loc"] = val


def _act_sym(name: str, i: int, act: tuple[str, str, str], *extra: Term) -> Term:
    verb, obj, val = act
    return Pred(name, Num(i), *extra, Ident(verb), Ident(obj), Ident(val))


def _instruction_query(rng: random.Random, ids: Sequence[str],
                       st: Mapping[str, Mapping[str, str]]):
    slot = rng.choice(["color", "loc"])
    q = rng.choice(list(ids))
    vocab = COLORS if slot == "color" else LOCATIONS
    name = "color_of" if slot == "color" else "location_of"
    return Pred(name, Ident(q)), _shuffled(rng, vocab), st[q][slot], (slot, q)

