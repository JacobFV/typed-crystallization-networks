"""The search found a better predicate than the one the bound was derived for.

`chroma_bound.py` and `residual.py` analyse `max(cross products) <= T`, the
conjunction of all three collinearity tests, because that is the textbook form
of the invariant.  The searched space also contains every other Boolean
combination of the three, and the program the wide arm returned is

    same = (|r1*g2 - r2*g1| <= T  or  |g1*b2 - g2*b1| <= T)  and  |r1*b2 - r2*b1| <= T

(`m1 = truth_1` is NOR, `same = truth_2` is `not m1 and rb`).  This measures the
exact-threshold interval of both shapes on all three splits, which is the honest
comparison: the bound bounds the shape it was written for, not the family.
"""
from __future__ import annotations
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "discrete-perception"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common2 as C
from common2 import OUT
import common as DP

R = 8
POOL_COARSE = (0, 16, 48, 96, 192, 384, 1024, 4096)


def cross(a, b):
    r1, g1, b1 = a; r2, g2, b2 = b
    return (abs(r1 * g2 - r2 * g1), abs(g1 * b2 - g2 * b1), abs(r1 * b2 - r2 * b1))


def pairs(seeds, split):
    out = []
    for s in seeds:
        px, pr = DP.episode(s, R, split=split, objects=6)
        g = C.rgb_grid(px, R); ids = C.ids_grid(pr, R)
        for i in range(R * R - R):
            out.append((g[i // R][i % R], g[(i + 1) // R][(i + 1) % R],
                        ids[i // R][i % R] == ids[(i + 1) // R][(i + 1) % R]))
    return out


RULES = {
    "AND of all three (the shape the bound was derived for)":
        lambda d, T: all(x <= T for x in d),
    "(rg OR gb) AND rb -- the program the wide search returned":
        lambda d, T: (d[0] <= T or d[1] <= T) and d[2] <= T,
}


def main():
    sets = {"train": pairs(range(8), "train"),
            "validation": pairs(range(50, 58), "train"),
            "test": pairs(range(100, 108), "test")}
    res = {}
    for name, rule in RULES.items():
        row = {}
        for k, ps in sets.items():
            exact = [T for T in range(900)
                     if all(rule(cross(a, b), T) == y for a, b, y in ps)]
            row[k] = {"exact_interval": [min(exact), max(exact)] if exact else None}
        allex = [T for T in range(900)
                 if all(all(rule(cross(a, b), T) == y for a, b, y in ps) for ps in sets.values())]
        row["exact_on_all_three"] = [min(allex), max(allex)] if allex else None
        row["width"] = len(allex)
        row["coarse_pool_hits"] = [T for T in POOL_COARSE if T in allex]
        res[name] = row
        DP.report(name, json.dumps(row))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rule_shapes.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / "rule_shapes.json")


if __name__ == "__main__":
    main()
