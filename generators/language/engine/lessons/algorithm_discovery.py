"""``algorithm_discovery`` — identifying a procedure from I/O demonstrations.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import PROCEDURES, _shuffled


def gen_algorithm_discovery(rng: random.Random, ctx):
    """Infer the procedure from its input/output behaviour.

    Five named candidates are offered and all five are executed on all three
    demonstrations; the episode is rejected unless every wrong candidate
    disagrees with the observed outputs somewhere, so no distractor is
    eliminated by luck and none is secretly extensionally equal on this data.
    """
    n_cands = ctx.at(5, len(PROCEDURES), default=5)
    for _ in range(400):
        names = rng.sample(sorted(PROCEDURES), n_cands)
        truth = rng.choice(names)
        inputs = [[rng.randint(1, 9) for _ in range(rng.randint(4, 6))] for _ in range(3)]
        outs = [PROCEDURES[truth](x) for x in inputs]
        if all(any(PROCEDURES[nm](x) != o for x, o in zip(inputs, outs))
               for nm in names if nm != truth):
            break
    else:                                     # pragma: no cover - construction
        pass

    obs = Rec(demonstrations=Lst([Pred("maps", Lst([Num(v) for v in x]),
                                       Lst([Num(v) for v in o]))
                                  for x, o in zip(inputs, outs)]),
              candidates=Lst([Pred("procedure", Ident(nm)) for nm in _shuffled(rng, names)]),
              query=Ident("which_procedure"))
    return (obs, _shuffled(rng, names), truth,
            {"answer": truth, "inputs": inputs, "outputs": outs})


class AlgorithmDiscovery(Lesson):
    """Identifying a procedure from I/O demonstrations."""

    id = "algorithm_discovery"
    level = 113
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "identifying a procedure from I/O demonstrations"
    capabilities = ('program_induction', 'algorithms', 'hypothesis_elimination')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'world_complexity': 3}

    generate = staticmethod(gen_algorithm_discovery)
