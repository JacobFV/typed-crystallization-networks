"""``anomaly_resolution`` — noise, hidden variable, or boundary condition.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.science import _CAUSES, _lin, _shuffled


def gen_anomaly_resolution(rng: random.Random, ctx):
    """A good theory, and data that disobeys it. Why?

    Three diagnoses are available and the episode generates exactly one of them,
    then checks that the *other two are ruled out by the table itself*: a single
    deviant row that no column explains and that has well-behaved rows above it
    is noise; a deviant set that a recorded binary column predicts perfectly is a
    hidden variable; a deviant set that is exactly a suffix in ``x`` that no
    column predicts is a boundary condition. Nothing about the diagnosis depends
    on knowing which one was drawn — it is recoverable from the rows.
    """
    n = ctx.at(7, 13, default=7)
    for _ in range(400):
        cause = rng.choice(_CAUSES)
        p = rng.choice([-3, -2, -1, 1, 2, 3])
        q = rng.randint(-5, 5)
        xs = sorted(rng.sample(range(1, 16), n))
        cols = {c: [rng.randint(0, 1) for _ in range(n)] for c in ("c1", "c2")}
        dev: set[int] = set()
        shift = [0] * n

        if cause == "noise":
            i = rng.randrange(n - 2)                        # never the top of the x range
            d = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            shift[i], dev = d, {i}
        elif cause == "hidden_variable":
            col = rng.choice(["c1", "c2"])
            marked = rng.sample(range(n), rng.choice([2, 3]))
            if max(marked) == n - 1 and set(marked) == set(range(n - len(marked), n)):
                continue                                    # would also read as a boundary
            for j in range(n):
                cols[col][j] = 1 if j in marked else 0
            d = rng.choice([-5, -4, -3, 3, 4, 5])
            for j in marked:
                shift[j] = d
            dev = set(marked)
        else:
            k = rng.choice([2, 3])
            slope = rng.choice([-3, -2, 2, 3])
            t = xs[n - k]
            for j in range(n - k, n):
                shift[j] = slope * (xs[j] - t) + rng.choice([-2, -1, 1, 2])
            dev = {j for j in range(n) if shift[j] != 0}
            if len(dev) < 2:
                continue

        ys = [p * x + q + s for x, s in zip(xs, shift)]
        if {j for j in range(n) if ys[j] != p * xs[j] + q} != dev:
            continue                                        # a "shift" that shifted nothing
        column_match = any(set(j for j in range(n) if cols[c][j] == v) == dev
                           for c in ("c1", "c2") for v in (0, 1))
        suffix = dev == {j for j in range(n) if xs[j] >= min(xs[j] for j in dev)}
        if cause == "noise" and (column_match or len(dev) != 1):
            continue
        if cause == "hidden_variable" and (not column_match or suffix or len(dev) < 2):
            continue
        if cause == "boundary_condition" and (column_match or not suffix or len(dev) < 2):
            continue

        order = _shuffled(rng, range(n))
        rows = Lst([Pred("row", Ident(f"r{j + 1}"), Num(xs[i]), Num(cols["c1"][i]),
                         Num(cols["c2"][i]), Num(ys[i])) for j, i in enumerate(order)])
        obs = Rec(theory=Pred("eq", Ident("y"), _lin(p, "x", q)),
                  columns=Lst([Ident("x"), Ident("c1"), Ident("c2"), Ident("y")]),
                  measurements=rows, query=Ident("anomaly_cause"))
        hidden = {"cause": cause, "law": [p, q], "deviant_rows": sorted(dev),
                  "n_deviant": len(dev)}
        return obs, _shuffled(rng, _CAUSES), cause, hidden
    raise RuntimeError("anomaly_resolution: no admissible episode")


class AnomalyResolution(Lesson):
    """Noise, hidden variable, or boundary condition."""

    id = "anomaly_resolution"
    level = 73
    tags = ("science", "induction", "model-discovery")
    teaches = "noise, hidden variable, or boundary condition"
    capabilities = ('scientific_induction', 'causal_reasoning', 'ontology_learning')
    axes = {'reasoning_depth': 5, 'world_complexity': 3, 'ambiguity': 3}
    answers = ['noise', 'hidden_variable', 'boundary_condition']

    generate = staticmethod(gen_anomaly_resolution)
