"""``theorem_proving`` — the next rule of a shortest proof in a generated calculus.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Str
from ..lesson import Lesson
from ..generators.mathematics import _FALLBACK_RULES, _nonces, _reach, _rewrite, _shuffled


def gen_theorem_proving(rng: random.Random, ctx):
    """A generated rewrite calculus, and the *next* step of a shortest proof.

    Breadth-first search over the whole derivation space fixes the distance from
    the start term to the goal; a rule is the answer only if applying it lands on
    a term one step closer, and the episode is rejected unless exactly one rule
    does that while at least two are applicable. So the label is the shortest
    proof's first move, computed, not guessed."""
    names = _nonces(rng, 4, 2)
    alpha = "abcde"
    d_lo, d_hi = ctx.span((2, 4), (4, 7))
    max_depth = ctx.at(5, 8, default=5)
    for _ in range(300):
        rules = []
        for nm in names:
            lhs = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 2)))
            rhs = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 2)))
            if lhs != rhs:
                rules.append((nm, lhs, rhs))
        if len(rules) != 4:
            continue
        start = "".join(rng.choice(alpha) for _ in range(rng.randint(2, 3)))
        dist = _reach(start, rules, max_len=7, max_depth=max_depth)
        goals = [t for t, d in dist.items() if d_lo <= d <= d_hi]
        if not goals:
            continue
        goal = rng.choice(goals)
        d = dist[goal]
        succ = [(nm, _rewrite(start, lhs, rhs)) for nm, lhs, rhs in rules]
        succ = [(nm, t) for nm, t in succ if t is not None and len(t) <= 7]
        if len(succ) < 2:
            continue
        on_path = []
        for nm, t in succ:
            dt = _reach(t, rules, max_len=7, max_depth=max_depth).get(goal)
            if dt is not None and 1 + dt == d:
                on_path.append(nm)
        if len(on_path) == 1:
            answer = on_path[0]
            break
    else:                                                # pragma: no cover - construction
        rules = [(names[i], lhs, rhs) for i, (lhs, rhs) in enumerate(_FALLBACK_RULES)]
        start, goal, d, answer = "ab", "eb", 2, names[0]

    obs = Rec(calculus=Lst([Pred("rule", Ident(nm), Str(lhs), Str(rhs))
                            for nm, lhs, rhs in _shuffled(rng, rules)]),
              state=Str(start), goal=Str(goal),
              query=Ident("next_rule_of_a_shortest_proof"))
    return obs, _shuffled(rng, [nm for nm, _, _ in rules]), answer, {
        "start": start, "goal": goal, "proof_length": d}


class TheoremProving(Lesson):
    """The next rule of a shortest proof in a generated calculus."""

    id = "theorem_proving"
    level = 83
    tags = ("mathematics", "formal-reasoning")
    teaches = "the next rule of a shortest proof in a generated calculus"
    capabilities = ('proof_search', 'rewriting', 'planning')
    axes = {'reasoning_depth': 5, 'recursion_depth': 3, 'grammar_complexity': 3}

    generate = staticmethod(gen_theorem_proving)
