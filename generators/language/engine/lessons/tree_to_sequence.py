"""``tree_to_sequence`` — realization from structure.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random
import string

from .._structure import Ident, Rec
from ..lesson import Lesson
from ..generators.base import _mini_tree


def gen_tree_to_sequence(rng: random.Random, ctx):
    """Realize a symbolic tree as its surface form (here: its yield)."""
    depth = rng.randint(*ctx.span((1, 3), (2, 5)))
    tree, yield_ = _mini_tree(rng, depth)
    obs = Rec(tree=tree, query=Ident("first_leaf"))
    return obs, list(string.ascii_lowercase[:6]), yield_[0], {"yield": yield_}


class TreeToSequence(Lesson):
    """Realization from structure."""

    id = "tree_to_sequence"
    level = 9
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "realization from structure"
    capabilities = ()
    axes = {'recursion_depth': 2, 'compositional_depth': 2}

    generate = staticmethod(gen_tree_to_sequence)
