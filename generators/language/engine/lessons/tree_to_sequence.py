"""``tree_to_sequence`` — realization from structure.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random
import string

from .._structure import Ident, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import _mini_tree


def gen_tree_to_sequence(rng: random.Random, ctx):
    """Realize a symbolic tree as its surface form (here: its yield)."""
    depth = rng.randint(*ctx.span((1, 3), (2, 5)))
    tree, yield_ = _mini_tree(rng, depth)
    # The first leaf of an in-order rendering is the first symbol printed, so
    # "the token after the word *tree*" answers the lesson without ever
    # computing a yield.  Asking for a leaf drawn at random still requires the
    # yield -- the k-th leaf is not at any fixed offset once the tree shape
    # varies -- and no fixed position holds the answer.
    if not ctx.hardens("tree_to_sequence"):
        obs = Rec(tree=tree, query=Ident("first_leaf"))
        return obs, list(string.ascii_lowercase[:6]), yield_[0], {"yield": yield_}
    k = rng.randrange(len(yield_))
    obs = Rec(tree=tree, query=Pred("leaf_at", Num(k)))
    return (obs, list(string.ascii_lowercase[:6]), yield_[k],
            {"yield": yield_, "index": k})


class TreeToSequence(Lesson):
    """Realization from structure."""

    id = "tree_to_sequence"
    level = 9
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "realization from structure"
    capabilities = ()
    axes = {'recursion_depth': 2, 'compositional_depth': 2}

    generate = staticmethod(gen_tree_to_sequence)
