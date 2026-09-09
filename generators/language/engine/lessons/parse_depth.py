"""``parse_depth`` — structural decomposition.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Rec, Str
from ..lesson import Lesson
from ..generators.base import _balanced, _dyck, _max_depth


def gen_parse_depth(rng: random.Random, ctx):
    """Recover a structural property of the hidden parse: nesting depth."""
    depth = rng.randint(*ctx.span((1, 5), (2, 6)))
    if not ctx.hardens("parse_depth"):
        s = _balanced(rng, depth)
        obs = Rec(string=Str(s), query=Ident("max_depth"))
        return obs, list(range(0, 7)), _max_depth(s), {"string": s}
    # ``_balanced`` emits so few distinct strings that 85% of prompts recur
    # across seeds and a lookup table over training prompts scores 0.998.  It
    # also makes the string longer when it is deeper, so length alone carries
    # the answer.  Drawing the number of pairs independently of the depth fixes
    # both: the length distribution is the same at every depth.
    s = _dyck(rng, depth, rng.randint(max(depth, 6), 18))
    obs = Rec(string=Str(s), query=Ident("max_depth"))
    return obs, list(range(0, 7)), _max_depth(s), {"string": s, "pairs": len(s) // 2}


class ParseDepth(Lesson):
    """Structural decomposition."""

    id = "parse_depth"
    level = 8
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "structural decomposition"
    capabilities = ()
    axes = {'recursion_depth': 3, 'grammar_complexity': 3}

    generate = staticmethod(gen_parse_depth)
