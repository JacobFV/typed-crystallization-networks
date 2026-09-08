"""The shared vocabulary and the scene helpers the whole curriculum draws on.

Six colours, six shapes, six names. The smallness is deliberate: a lesson that
matters is never about how many colour words exist, and a small closed
vocabulary keeps the answer sets small enough that a uniform-guessing floor is
meaningful. Where a lesson needs *new* words it invents them per episode
instead of drawing on a bigger fixed list — see the nonce-word helpers in the
section support modules.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..binding import SlotVocabulary, bind_scene

__all__ = ["COLORS", "SHAPES", "NAMES", "_scene", "_scene_term",
           "_balanced", "_is_balanced", "_max_depth", "_mini_tree"]

COLORS = SlotVocabulary("color", ["red", "blue", "green", "yellow", "purple", "orange"])
SHAPES = SlotVocabulary("shape", ["cube", "sphere", "cone", "prism", "disc", "rod"])
NAMES = SlotVocabulary("name", ["alice", "bob", "carol", "dave", "erin", "frank"])


def _scene(rng: random.Random, n: int) -> list[dict[str, Any]]:
    objs = []
    for i in range(n):
        objs.append({"id": f"o{i}", "color": rng.choice(COLORS), "shape": rng.choice(SHAPES),
                     "x": rng.randint(0, 9), "y": rng.randint(0, 9)})
    return bind_scene(objs, rng) or objs


def _scene_term(objs: Sequence[Mapping[str, Any]], query: Term) -> Term:
    facts = [Pred("obj", Ident(o["id"]), Ident(o["color"]), Ident(o["shape"]), Num(o["x"]), Num(o["y"]))
             for o in objs]
    return Rec(scene=Lst(facts), query=query)


def _balanced(rng: random.Random, depth: int) -> str:
    if depth <= 0:
        return ""
    inner = _balanced(rng, depth - 1)
    return "(" + inner + ")" + ("()" if rng.random() < 0.4 else "")


def _is_balanced(s: str) -> bool:
    d = 0
    for c in s:
        d += 1 if c == "(" else -1
        if d < 0:
            return False
    return d == 0


def _max_depth(s: str) -> int:
    d = best = 0
    for c in s:
        d += 1 if c == "(" else -1
        best = max(best, d)
    return best


def _mini_tree(rng: random.Random, depth: int) -> tuple[Term, list[str]]:
    import string
    if depth <= 0 or rng.random() < 0.3:
        leaf = rng.choice(string.ascii_lowercase[:6])
        return Pred("leaf", Ident(leaf)), [leaf]
    left, ly = _mini_tree(rng, depth - 1)
    right, ry = _mini_tree(rng, depth - 1)
    return Pred("node", left, right), ly + ry
