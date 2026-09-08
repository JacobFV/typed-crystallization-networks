"""``symbolic_world_builder`` — which generated world actually demands a named capability.

Civilization-scale symbolic learning.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from .._structure import Ident, Lst, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.capstone import _grammar, _nonce_pool, _productive, _repeated_variable, _self_embedding, _shuffled


def gen_symbolic_world_builder(rng: random.Random, ctx):
    """Four generated worlds, one named capability: which world actually forces
    a learner to acquire it?

    Building a world cannot be graded in one step, so this grades the half of
    that ability that can be: recognizing what a world you did not write
    demands. The label is computed, never assumed — both checkers are run over
    all four candidates, exactly one candidate has the queried property, and at
    least one *other* candidate has the other property, so answering without
    reading the query is punished.
    """
    caps = ["unbounded_nesting", "variable_binding"]
    target = rng.choice(caps)
    varnames = ["_x", "_y"]
    positives: list[Any] = []
    negatives: list[Any] = []
    other_positives: list[Any] = []
    size = ctx.at(3, 5, default=3)               # nonterminals per candidate grammar
    for _ in range(4000):
        nts = [w.upper() for w in _nonce_pool(rng, size, 1)]
        terms = _nonce_pool(rng, size, 1)
        if len({*nts, *terms}) < 2 * size:
            continue
        rules = _grammar(rng, nts, terms, varnames)
        if not _productive(rules, nts):
            continue
        se = _self_embedding(rules, nts)
        rv = _repeated_variable(rules, varnames)
        has = se if target == "unbounded_nesting" else rv
        other = rv if target == "unbounded_nesting" else se
        cand = (nts, terms, rules, se, rv)
        if has and len(positives) < 4:
            positives.append(cand)
        elif (not has) and other and len(other_positives) < 4:
            other_positives.append(cand)
        elif (not has) and len(negatives) < 8:
            negatives.append(cand)
        if len(positives) >= 3 and len(other_positives) >= 3 and len(negatives) >= 6:
            break
    if not positives or not other_positives or len(negatives) < 2:      # pragma: no cover
        return gen_symbolic_world_builder(random.Random(rng.random()), ctx)

    def _bulk(c) -> int:
        return sum(len(b) for _h, b in c[2])

    def _nvars(c) -> int:
        return sum(1 for _h, b in c[2] for s in b if s in varnames)

    # The answer must be neither the extreme-largest nor the extreme-smallest
    # candidate on either statistic. "Pick the longest grammar" and "pick the one
    # with the fewest variables" are both correlates of the real properties, and
    # left alone they buy a third of the episodes for free — in either direction,
    # which is just as exploitable.
    combo = None
    for pos in positives:
        for oth in other_positives:
            for negs in itertools.combinations(negatives, 2):
                rest = [oth, *negs]
                if all(min(f(c) for c in rest) <= f(pos) <= max(f(c) for c in rest)
                       for f in (_bulk, _nvars)):
                    combo = [pos, *rest]
                    break
            if combo:
                break
        if combo:
            break
    if combo is None:
        return gen_symbolic_world_builder(random.Random(rng.random()), ctx)
    cands = combo

    names = _shuffled(rng, [f"w{i}" for i in range(4)])
    order = _shuffled(rng, list(range(len(cands))))
    worlds: list[Term] = []
    answer = names[0]
    for slot, idx in enumerate(order):
        nts, terms, rules, se, rv = cands[idx]
        name = names[slot]
        if idx == 0:
            answer = name
        body = [Pred("rule", Ident(name), Ident(h), Lst([Ident(s) for s in b])) for h, b in rules]
        worlds.append(Rec(world=Ident(name),
                          ontology=Lst([Ident(s) for s in nts] + [Ident(s) for s in terms]),
                          start=Ident(nts[0]),
                          dynamics=Lst(_shuffled(rng, body)),
                          task=Pred("recognize_generated_strings")))
    obs = Rec(candidates=Lst(worlds),
              legend=Lst([Pred("variable", Ident(v)) for v in varnames]),
              query=Pred("which_world_requires", Ident(target)))
    hidden = {"target": target, "answer": answer,
              "properties": {names[s]: {"self_embedding": cands[i][3],
                                        "repeated_variable": cands[i][4]}
                             for s, i in enumerate(order)}}
    return obs, _shuffled(rng, names), answer, hidden


class SymbolicWorldBuilder(Lesson):
    """Which generated world actually demands a named capability."""

    id = "symbolic_world_builder"
    level = 165
    tags = ("civilization-scale",)
    teaches = "which generated world actually demands a named capability"
    capabilities = ('metareasoning', 'recursive_syntax', 'variable_binding', 'abstraction')
    axes = {'recursion_depth': 5, 'grammar_complexity': 5, 'reasoning_depth': 5, 'compositional_depth': 4}

    generate = staticmethod(gen_symbolic_world_builder)
