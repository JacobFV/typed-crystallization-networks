"""``ellipsis`` — recovering elided structure.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.semantics import OBJECTS, TRANS_VERBS, _shuffled


def gen_ellipsis(rng: random.Random, ctx):
    """Recover elided structure: gapping and VP-ellipsis.

    ``alice likes tea and bob coffee`` (the verb is gapped: what does bob do?)
    and ``alice likes tea and bob does too`` (the VP is elided: what does bob
    like?). Distractor clauses precede the antecedent so the answer is never the
    only verb or the only object in the discourse.
    """
    d = ctx.at(2, 4, default=2)                  # clauses between the reader and the antecedent
    subs = rng.sample(NAMES, d + 2)
    verbs = rng.sample(TRANS_VERBS, d + 1)
    objs = rng.sample(OBJECTS, d + 2)
    mode = rng.choice(["gapping", "vp_ellipsis"])

    clauses = [(subs[2 + i], verbs[1 + i], objs[1 + i]) for i in range(d)]
    rng.shuffle(clauses)
    antecedent = (subs[0], verbs[0], objs[0])
    clauses.append(antecedent)

    lines = [Pred("clause", Num(i), Ident(s), Ident(v), Ident(o))
             for i, (s, v, o) in enumerate(clauses)]
    i = len(clauses)
    if mode == "gapping":
        lines.append(Pred("gap", Num(i), Ident(subs[1]), Ident(objs[d + 1])))
        query = Pred("verb_of", Ident(subs[1]))
        vocab, answer = _shuffled(rng, TRANS_VERBS), antecedent[1]
    else:
        lines.append(Pred("vp_gap", Num(i), Ident(subs[1])))
        query = Pred("object_of", Ident(subs[1]))
        vocab, answer = _shuffled(rng, OBJECTS), antecedent[2]

    obs = Rec(discourse=Lst(lines), query=query)
    return obs, vocab, answer, {"mode": mode, "antecedent": list(antecedent)}


class Ellipsis(Lesson):
    """Recovering elided structure."""

    id = "ellipsis"
    level = 24
    tags = ("pragmatics", "language-as-action")
    teaches = "recovering elided structure"
    capabilities = ('recursive_syntax', 'variable_binding')
    axes = {'grammar_complexity': 3, 'compositional_depth': 3, 'discourse_horizon': 2}

    generate = staticmethod(gen_ellipsis)
