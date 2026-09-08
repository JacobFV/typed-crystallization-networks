"""Support for the dialogue, convention and channel lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from ..binding import SlotVocabulary
from .._structure import Ident, Lst, Pred, Term

SIZES = SlotVocabulary("size", ["small", "large"])


ITEMS = SlotVocabulary("material", ["wood", "ore", "grain", "clay", "gold", "salt"])


PLACES = SlotVocabulary("place", ["attic", "cellar", "garden", "kitchen", "study", "barn"])


ACTIONS = SlotVocabulary("act", ["lift", "turn", "push", "wait", "open", "close"])


COMPANIES = ["acme", "zenith", "orbis", "helix", "vertex"]


CITIES = SlotVocabulary("city", ["paris", "lisbon", "osaka", "denver", "cairo", "oslo"])


PROPS = SlotVocabulary("property", ["metallic", "fragile", "heavy", "hollow", "striped", "warm"])


DETECT_CLASSES = SlotVocabulary("thing", ["cup", "book", "laptop", "plant", "bottle", "phone", "clock", "bowl"])


SURFACES = SlotVocabulary("surface", ["tray", "table", "shelf", "mat"])


CONCEPTS = SlotVocabulary("substance", ["fire", "water", "stone", "cloud", "seed", "wind"])


WORDS = ["tama", "luko", "reshi", "vune", "peki", "soga"]


TAGS = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]


NONCE_LETTERS = "kmtszlpvr"


def _shuffled(rng: random.Random, xs: Sequence[Any]) -> list[Any]:
    """A shuffled copy. Every answer vocabulary in this module goes through it."""
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _nonce(rng: random.Random, n: int = 3) -> str:
    return "".join(rng.choice(NONCE_LETTERS) for _ in range(n))


def _obj_facts(objs: Sequence[Mapping[str, Any]]) -> Term:
    return Lst([Pred("obj", Ident(o["id"]), Ident(o["color"]), Ident(o["shape"]), Ident(o["size"]))
                for o in objs])


def _prefix_free(codewords: Sequence[str]) -> bool:
    """No codeword is a prefix of another — duplicates fail this too."""
    return all(not a.startswith(b)
               for i, a in enumerate(codewords) for j, b in enumerate(codewords) if i != j)


def _encoded_length(code: Mapping[str, str], message: Sequence[str]) -> int:
    return sum(len(code[w]) for w in message)


def _code_ok(code: Mapping[str, str], message: Sequence[str], budget: int) -> bool:
    return _prefix_free(list(code.values())) and _encoded_length(code, message) <= budget


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _spread_code(rng: random.Random, n: int, length: int, alphabet: str, dmin: int) -> list[str] | None:
    """Greedily draw ``n`` codewords pairwise at Hamming distance ``>= dmin``."""
    code: list[str] = []
    for _ in range(4000):
        cand = "".join(rng.choice(alphabet) for _ in range(length))
        if all(sum(x != y for x, y in zip(cand, c)) >= dmin for c in code):
            code.append(cand)
            if len(code) == n:
                return code
    return None


def _inside(det: Mapping[str, float], box: Mapping[str, float]) -> bool:
    return (abs(det["cx"] - box["cx"]) <= box["w"] / 2 and
            abs(det["cy"] - box["cy"]) <= box["h"] / 2)
