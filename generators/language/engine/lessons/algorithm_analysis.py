"""``algorithm_analysis`` — comparing algorithms by exact executed cost.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random
from typing import Sequence

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _backward_scan, _binary_search, _forward_scan, _jump_search, _shuffled


def gen_algorithm_analysis(rng: random.Random, ctx):
    """Compare candidate algorithms by executing them and counting.

    Four search procedures run on the same sorted array and target; the cost is
    the number of comparisons each actually performs, obtained by simulation
    rather than by asymptotic reasoning — which is the point, since on arrays
    this size the linear scans routinely beat the logarithmic one and the
    ranking depends entirely on where the target sits.
    """
    def cost_table(arr: Sequence[int], t: int, b: int) -> dict[str, int]:
        return {"forward_scan": _forward_scan(arr, t),
                "backward_scan": _backward_scan(arr, t),
                "binary_search": _binary_search(arr, t),
                f"jump_search_{b}": _jump_search(arr, t, b)}

    for _ in range(400):
        n = rng.randint(*ctx.span((8, 14), (20, 34)))
        arr = sorted(rng.sample(range(1, 60), n))
        b = rng.randint(2, 4)
        # which algorithm wins depends entirely on where the target sits, so the
        # winner is drawn uniformly first and a target realizing it is chosen
        by_winner: dict[str, list[int]] = {}
        for t in arr:
            c = cost_table(arr, t, b)
            w = min(c, key=lambda nm: c[nm])
            if list(c.values()).count(c[w]) == 1:
                by_winner.setdefault(w, []).append(t)
        if len(by_winner) < 3:
            continue
        best = rng.choice(sorted(by_winner))
        t = rng.choice(by_winner[best])
        costs = cost_table(arr, t, b)
        names = sorted(costs)
        break
    else:                                     # pragma: no cover - construction
        pass

    obs = Rec(array=Lst([Num(v) for v in arr]),
              target=Num(t),
              algorithms=Lst(_shuffled(rng, [
                  Pred("algorithm", Ident("forward_scan"), Pred("scan_from_start")),
                  Pred("algorithm", Ident("backward_scan"), Pred("scan_from_end")),
                  Pred("algorithm", Ident("binary_search"), Pred("halve_the_interval")),
                  Pred("algorithm", Ident(f"jump_search_{b}"), Pred("blocks_of", Num(b)))])),
              cost_model=Pred("count_element_comparisons"),
              query=Ident("fewest_comparisons"))
    return (obs, _shuffled(rng, names), best,
            {"costs": costs, "answer": best, "target_index": arr.index(t), "block": b})


class AlgorithmAnalysis(Lesson):
    """Comparing algorithms by exact executed cost."""

    id = "algorithm_analysis"
    level = 114
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "comparing algorithms by exact executed cost"
    capabilities = ('algorithms', 'computational_cost', 'simulation')
    axes = {'reasoning_depth': 4, 'computational_budget': 3, 'world_complexity': 3}

    generate = staticmethod(gen_algorithm_analysis)
