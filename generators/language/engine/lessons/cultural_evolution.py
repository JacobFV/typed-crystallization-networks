"""``cultural_evolution`` — which symbolic variant survives transmission bottlenecks.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.reflective import _labels, _shuffled


def gen_cultural_evolution(rng: random.Random, ctx):
    """Which variant survives transmission across generations?

    Four variants differ in how regular they are and how long they are; the
    stated dynamics reward regularity, punish length, and impose a bottleneck
    that kills all but the top ``k`` each generation. The population is
    simulated for the stated number of generations and the survivor is whoever
    is on top at the end — routinely not whoever started on top.
    """
    fallback = None
    for _ in range(400):
        ids = _labels(rng, "v", ctx.at(4, 9, default=4))
        info = {i: {"count": rng.randint(6, 24), "length": rng.randint(1, 5),
                    "regular": rng.random() < 0.5} for i in ids}
        g_reg, g_irr, c_len = rng.randint(4, 9), rng.randint(0, 3), rng.randint(1, 3)
        k = rng.randint(2, 3)
        gens = rng.randint(2, 4)
        count = {i: info[i]["count"] for i in ids}
        for _ in range(gens):
            for i in ids:
                if count[i] <= 0:
                    continue
                c = count[i] + (g_reg if info[i]["regular"] else g_irr) - c_len * info[i]["length"]
                count[i] = max(c, 0)
            alive = sorted([i for i in ids if count[i] > 0],
                           key=lambda i: (-count[i], info[i]["length"], i))
            for i in alive[k:]:
                count[i] = 0
        top = max(count.values())
        winners = [i for i in ids if count[i] == top]
        cand = (ids, info, g_reg, g_irr, c_len, k, gens, count, winners[0])
        if fallback is None:
            fallback = cand
        if len(winners) == 1 and top > 0:
            fallback = cand
            break
    ids, info, g_reg, g_irr, c_len, k, gens, count, answer = fallback
    obs = Rec(population=Lst(_shuffled(rng, [
                  Pred("variant", Ident(i), Num(info[i]["count"]), Num(info[i]["length"]),
                       Ident("regular" if info[i]["regular"] else "irregular")) for i in ids])),
              dynamics=Lst([Pred("gain", Ident("regular"), Num(g_reg)),
                            Pred("gain", Ident("irregular"), Num(g_irr)),
                            Pred("cost", Pred("per_unit_length"), Num(c_len)),
                            Pred("bottleneck", Pred("keep_top"), Num(k)),
                            Pred("tie_break", Pred("shorter_then_name"))]),
              generations=Num(gens),
              query=Pred("which_variant_leads_at_the_end"))
    return obs, _shuffled(rng, ids), answer, {"final_counts": count, "generations": gens}


class CulturalEvolution(Lesson):
    """Which symbolic variant survives transmission bottlenecks."""

    id = "cultural_evolution"
    level = 130
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "which symbolic variant survives transmission bottlenecks"
    capabilities = ('scientific_induction', 'abstraction', 'open_ended_discovery')
    axes = {'reasoning_depth': 4, 'discourse_horizon': 4, 'world_complexity': 4}

    generate = staticmethod(gen_cultural_evolution)
