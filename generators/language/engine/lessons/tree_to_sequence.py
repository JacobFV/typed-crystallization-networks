"""``tree_to_sequence`` — realization from structure.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random
import string

from .._structure import Ident, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import _mini_tree
from .._structure import Pred as _Pred


def _wide_tree(rng: random.Random, depth: int, leaves: int, alpha: str):
    """A binary tree with exactly ``leaves`` leaves and depth at most ``depth``.

    ``_mini_tree`` stops early with probability 0.3 at every node, so at the
    depths this lesson draws it emits a handful of shapes over six symbols and
    two thirds of its prompts recur across seeds.  Splitting a randomly chosen
    leaf until the count is reached keeps the same shape family and makes the
    space grow with the size.
    """
    def leaf():
        return _Pred("leaf", Ident(rng.choice(alpha)))

    tree = leaf()
    yield_ = [tree.value[1].value]
    for _ in range(leaves - 1):
        spots = _open_leaves(tree, depth)
        if not spots:
            break
        target = rng.choice(spots)
        tree = _split(tree, target, leaf(), leaf())
        yield_ = _yield(tree)
    return tree, yield_


def _yield(t):
    if t.value[0] == "leaf":
        return [t.value[1].value]
    return _yield(t.value[1]) + _yield(t.value[2])


def _open_leaves(t, budget, k=0):
    """Indices, in yield order, of leaves that may still be split."""
    if t.value[0] == "leaf":
        return [k] if budget > 0 else []
    left = _open_leaves(t.value[1], budget - 1, k)
    nl = len(_yield(t.value[1]))
    return left + [i + nl for i in _open_leaves(t.value[2], budget - 1, k)]


def _split(t, index, a, b, k=0):
    if t.value[0] == "leaf":
        return _Pred("node", a, b) if index == k else t
    nl = len(_yield(t.value[1]))
    if index < nl:
        return _Pred("node", _split(t.value[1], index, a, b, k), t.value[2])
    return _Pred("node", t.value[1], _split(t.value[2], index - nl, a, b, k))


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
    alpha = string.ascii_lowercase[:10]
    tree, yield_ = _wide_tree(rng, depth + 2, rng.randint(3, 8), alpha)
    k = rng.randrange(len(yield_))
    obs = Rec(tree=tree, query=Pred("leaf_at", Num(k)))
    return (obs, list(alpha), yield_[k], {"yield": yield_, "index": k})


class TreeToSequence(Lesson):
    """Realization from structure."""

    id = "tree_to_sequence"
    level = 9
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "realization from structure"
    capabilities = ()
    axes = {'recursion_depth': 2, 'compositional_depth': 2}

    generate = staticmethod(gen_tree_to_sequence)
