"""The bound for the family the algebra can actually express, before searching it.

`same_chroma` in `bounds.py` uses floating ratios, which `role="byte"` pixels
cannot produce.  What the algebra *can* produce, once `pack` strips the byte
role, is the integer cross product: two pixels are collinear iff
`r1*g2 == r2*g1` and `g1*b2 == g2*b1` and `r1*b2 == r2*b1`.  The renderer
truncates `base * shade` to uint8, so exact equality is too strict; the
expressible predicate is `|r1*g2 - r2*g1| <= T` with `T` a searched constant.

This measures that predicate's ceiling over `T` -- the bound on the rung-4
search, computed before it.
"""
from __future__ import annotations
import collections, json, time
import common2 as C
from common2 import OUT
import common as DP

R = 8
OBJECTS = 6
TRAIN = range(0, 12)
HELD = range(100, 112)


def pairs(eps, R, offset="right"):
    out = []
    for pixels, probes in eps:
        g = C.rgb_grid(pixels, R); ids = C.ids_grid(probes, R)
        for r in range(R):
            for c in range(R):
                rr, cc = (r, c + 1) if offset == "right" else (r + 1, c)
                if not (0 <= rr < R and 0 <= cc < R):
                    continue
                out.append((g[r][c], g[rr][cc], int(ids[r][c] == ids[rr][cc]),
                            ids[r][c] >= 0, ids[rr][cc] >= 0))
    return out


def cross(a, b):
    r1, g1, b1 = a; r2, g2, b2 = b
    return (abs(r1 * g2 - r2 * g1), abs(g1 * b2 - g2 * b1), abs(r1 * b2 - r2 * b1))


def sweep(train, held, offset="right"):
    tr, he = pairs(train, R, offset), pairs(held, R, offset)
    rows = []
    grid = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]
    for T in grid:
        def pred(a, b):
            d = cross(a, b)
            return all(x <= T for x in d)
        def acc(rs, restrict=None):
            n = h = 0
            for a, b, y, fa, fb in rs:
                if restrict == "fg" and not (fa and fb):
                    continue
                if restrict == "fgfg_diff" and not (fa and fb):
                    continue
                n += 1; h += int(pred(a, b)) == y
            return h / max(1, n), n
        atr, ntr = acc(tr); ahe, nhe = acc(he)
        atrf, _ = acc(tr, "fg"); ahef, nhef = acc(he, "fg")
        rows.append({"T": T, "train_acc": atr, "held_acc": ahe,
                     "train_acc_fgfg": atrf, "held_acc_fgfg": ahef,
                     "n_held": nhe, "n_held_fgfg": nhef})
    base = collections.Counter(y for _, _, y, _, _ in he).most_common(1)[0][1] / len(he)
    basef = collections.Counter(y for _, _, y, fa, fb in he if fa and fb)
    basef = basef.most_common(1)[0][1] / max(1, sum(basef.values()))
    best = max(rows, key=lambda x: x["train_acc"])
    bestf = max(rows, key=lambda x: x["train_acc_fgfg"])
    return {"offset": offset, "rows": rows, "majority_baseline": base,
            "majority_baseline_fgfg": basef,
            "best_by_train": best, "best_by_train_fgfg": bestf}


def exact_equality_check(train):
    """How often does the *exact* cross product vanish for same-object pairs?"""
    tr = pairs(train, R, "right")
    same = [p for p in tr if p[2] and p[3] and p[4]]
    diff = [p for p in tr if not p[2] and p[3] and p[4]]
    def zero(p):
        return all(x == 0 for x in cross(p[0], p[1]))
    return {"fgfg_same_object_pairs": len(same),
            "exact_cross_zero_fraction_same": sum(map(zero, same)) / max(1, len(same)),
            "fgfg_different_object_pairs": len(diff),
            "exact_cross_zero_fraction_different": sum(map(zero, diff)) / max(1, len(diff)),
            "max_cross_over_same_object_pairs": max((max(cross(p[0], p[1])) for p in same),
                                                    default=0),
            "min_cross_over_different_object_pairs": min((max(cross(p[0], p[1])) for p in diff),
                                                         default=0)}


def main():
    t0 = time.perf_counter()
    train = C.load(TRAIN, R, OBJECTS, "train")
    held = C.load(HELD, R, OBJECTS, "train")
    out = {"resolution": R, "objects": OBJECTS}
    out["exact"] = exact_equality_check(train)
    for k, v in out["exact"].items():
        DP.report(k, v)
    for off in ("right", "down"):
        s = sweep(train, held, off)
        out[off] = s
        DP.report(f"[{off}] majority baseline all / fg-fg",
                  f"{s['majority_baseline']:.4f} / {s['majority_baseline_fgfg']:.4f}")
        for row in s["rows"]:
            print(f"   T={row['T']:6d} train={row['train_acc']:.4f} held={row['held_acc']:.4f}"
                  f"   fg-fg train={row['train_acc_fgfg']:.4f} held={row['held_acc_fgfg']:.4f}",
                  flush=True)
        DP.report(f"[{off}] best T by train accuracy",
                  f"T={s['best_by_train']['T']} held={s['best_by_train']['held_acc']:.4f}")
    out["seconds"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chroma_bound.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", OUT / "chroma_bound.json")


if __name__ == "__main__":
    main()
