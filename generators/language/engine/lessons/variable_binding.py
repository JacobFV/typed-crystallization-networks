"""``variable_binding`` — variable identity vs token identity.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random
import string

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES


def gen_variable_binding(rng: random.Random, ctx):
    """Variable identity independent of token identity."""
    n_bind = ctx.at(2, 5, default=2)
    vs = rng.sample(list(string.ascii_uppercase[:5]), n_bind)
    vals = rng.sample(NAMES, n_bind)
    subst = [Pred("bind", Ident(vs[i]), Ident(vals[i])) for i in range(n_bind)]
    rng.shuffle(subst)
    which = rng.randrange(n_bind)
    obs = Rec(substitution=Lst(subst), query=Pred("value_of", Ident(vs[which])))
    return obs, NAMES, vals[which], {"bindings": dict(zip(vs, vals))}


class VariableBinding(Lesson):
    """Variable identity vs token identity."""

    id = "variable_binding"
    level = 10
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "variable identity vs token identity"
    capabilities = ()
    axes = {'compositional_depth': 2, 'reasoning_depth': 2}
    answers = ['alice', 'bob', 'carol', 'dave', 'erin', 'frank']

    generate = staticmethod(gen_variable_binding)
