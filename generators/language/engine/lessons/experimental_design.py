"""``experimental_design`` — choose the experiment that separates hypotheses.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random
from typing import Callable, Mapping

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.science import _add, _labels, _mul, _random_core, _shuffled, _sub


def gen_experimental_design(rng: random.Random, ctx):
    """Three hypotheses survive the data; four experiments are affordable; one.

    The hypotheses agree everywhere except on a hidden switch — a variable that
    only matters away from one particular value — so three of the four candidate
    experiments produce the same outcome under every hypothesis and are worth
    nothing. Which one is informative is decided by *simulating each experiment
    under each hypothesis and counting distinct outcomes*, and a decoy variable
    is planted so that the surface heuristic "pick the experiment that is the
    odd one out" points somewhere else.
    """
    names = ["a", "b", "c"]
    n_flat = ctx.at(3, 8, default=3)
    for _ in range(300):
        switch = rng.choice(names)
        rest = [v for v in names if v != switch]
        rng.shuffle(rest)
        z = rng.randint(0, 3)                              # value where the switch is inert
        core_sym, core_f = _random_core(rng, rest)
        square = rng.random() < 0.4
        ks = rng.sample([-3, -2, -1, 1, 2, 3], 3)
        hyps: list[tuple[Term, Callable[[Mapping[str, int]], int]]] = []
        for k in ks:
            gap = _sub(Ident(switch), Num(z))
            term = _mul(Num(k), Pred("sq", gap) if square else gap)
            def f(e: Mapping[str, int], k: int = k) -> int:
                d = e[switch] - z
                return core_f(e) + k * (d * d if square else d)
            hyps.append((_add(core_sym, term), f))

        zp = rng.choice([v for v in range(0, 4) if v != z])
        decoy_var = rest[0]
        w, wp = rng.sample(range(0, 4), 2)
        free = rest[1]
        split = {switch: zp, decoy_var: w, free: rng.randint(0, 3)}
        flats = [{switch: z, decoy_var: (wp if j == n_flat - 1 else w),
                  free: rng.randint(0, 3)} for j in range(n_flat)]
        trials = [split] + flats
        if any(t == split for t in flats):
            continue
        informative = [i for i, t in enumerate(trials) if len({f(t) for _, f in hyps}) > 1]
        if informative != [0]:                             # recomputed, never assumed
            continue
        order = _shuffled(rng, range(len(trials)))
        labels = _labels("e", len(trials))
        answer = labels[order.index(0)]
        experiments = Lst([Pred("experiment", Ident(labels[j]),
                                *[Pred("set", Ident(v), Num(trials[i][v])) for v in names])
                           for j, i in enumerate(order)])
        hyp_order = _shuffled(rng, range(3))
        obs = Rec(hypotheses=Lst([Pred("hypothesis", Ident(f"h{j + 1}"),
                                       Pred("eq", Ident("y"), hyps[i][0]))
                                  for j, i in enumerate(hyp_order)]),
                  experiments=experiments,
                  query=Ident("discriminating_experiment"))
        hidden = {"switch": switch, "inert_value": z, "coefficients": ks,
                  "square": square, "answer": answer}
        return obs, _shuffled(rng, labels), answer, hidden
    raise RuntimeError("experimental_design: no admissible episode")


class ExperimentalDesign(Lesson):
    """Choose the experiment that separates hypotheses."""

    id = "experimental_design"
    level = 70
    tags = ("science", "induction", "model-discovery")
    teaches = "choose the experiment that separates hypotheses"
    capabilities = ('scientific_induction', 'planning', 'causal_reasoning')
    axes = {'reasoning_depth': 5, 'compositional_depth': 3, 'ambiguity': 2, 'world_complexity': 3}

    generate = staticmethod(gen_experimental_design)
