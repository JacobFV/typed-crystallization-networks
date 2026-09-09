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
    # clauses between the reader and the antecedent.  The hardened draw pins
    # this at the maximum the name pool allows: with only two distractors the
    # antecedent is first or second and "the option that occurs earliest" is
    # right half the time.  The knob has no range left under hardening, and the
    # lesson's difficulty axis is that much poorer for it.
    d = 4 if ctx.hardens("ellipsis") else ctx.at(2, 4, default=2)
    subs = rng.sample(NAMES, d + 2)
    verbs = rng.sample(TRANS_VERBS, d + 1)
    objs = rng.sample(OBJECTS, d + 2)
    mode = rng.choice(["gapping", "vp_ellipsis"])

    clauses = [(subs[2 + i], verbs[1 + i], objs[1 + i]) for i in range(d)]
    rng.shuffle(clauses)
    antecedent = (subs[0], verbs[0], objs[0])
    # Appending the antecedent last makes it the most recently mentioned verb
    # and object in the whole discourse, so "take the last option that occurs"
    # answers every episode without resolving anything.  Ellipsis still binds to
    # the immediately preceding clause; the hardened draw moves that *pair* into
    # the discourse instead of pinning it to the end, so what follows the gap is
    # a distractor and recency alone is wrong.
    at_end = not ctx.hardens("ellipsis")
    cut = len(clauses) if at_end else rng.randrange(len(clauses))
    before, after = clauses[:cut], clauses[cut:]
    clauses = before + [antecedent] + after

    lines = [Pred("clause", Num(i), Ident(s), Ident(v), Ident(o))
             for i, (s, v, o) in enumerate(before + [antecedent])]
    i = len(lines)
    if mode == "gapping":
        lines.append(Pred("gap", Num(i), Ident(subs[1]), Ident(objs[d + 1])))
        query = Pred("verb_of", Ident(subs[1]))
        vocab, answer = _shuffled(rng, TRANS_VERBS), antecedent[1]
    else:
        lines.append(Pred("vp_gap", Num(i), Ident(subs[1])))
        query = Pred("object_of", Ident(subs[1]))
        vocab, answer = _shuffled(rng, OBJECTS), antecedent[2]

    lines += [Pred("clause", Num(len(lines) + j), Ident(s), Ident(v), Ident(o))
              for j, (s, v, o) in enumerate(after)]
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
