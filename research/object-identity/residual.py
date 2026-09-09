"""What is left over after the collinearity rung, and why.

The search returns programs that are exact on training and validation and score
0.9956 on a fresh *test* split.  This says whether that residual is a search
failure or a property of the data: it computes the best accuracy any threshold
can reach on the test split (the ceiling for the searched family) and classifies
every position the best threshold gets wrong.
"""
from __future__ import annotations
import collections, json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "discrete-perception"))
import common2 as C
from common2 import OUT, BACKGROUND
import common as DP

R = 8
OBJECTS = 6


def cross(a, b):
    r1, g1, b1 = a; r2, g2, b2 = b
    return (abs(r1 * g2 - r2 * g1), abs(g1 * b2 - g2 * b1), abs(r1 * b2 - r2 * b1))


def pairs(seeds, split):
    out = []
    for s in seeds:
        pixels, probes = DP.episode(s, R, split=split, objects=OBJECTS)
        g = C.rgb_grid(pixels, R); ids = C.ids_grid(probes, R)
        for r in range(R):
            for c in range(R):
                if r * R + c >= R * R - R:
                    break
                rr, cc = (r, c + 1) if c + 1 < R else (r + 1, 0)   # raster +1, as the module reads it
                out.append({"seed": s, "r": r, "c": c, "a": g[r][c], "b": g[rr][cc],
                            "ia": ids[r][c], "ib": ids[rr][cc],
                            "same": ids[r][c] == ids[rr][cc]})
    return out


def main():
    t0 = time.perf_counter()
    out = {}
    for split, seeds in (("train", range(0, 8)), ("validation", range(50, 58)),
                         ("test", range(100, 108))):
        ps = pairs(seeds, "train" if split != "test" else "test")
        best = None
        for T in list(range(0, 600, 4)) + [800, 1024, 2048]:
            acc = sum((max(cross(p["a"], p["b"])) <= T) == p["same"] for p in ps) / len(ps)
            if best is None or acc > best[1]:
                best = (T, acc)
        base = collections.Counter(p["same"] for p in ps).most_common(1)[0][1] / len(ps)
        # classify the errors of the T the training split selects
        T192 = 192
        errs = [p for p in ps if (max(cross(p["a"], p["b"])) <= T192) != p["same"]]
        kinds = collections.Counter()
        for p in errs:
            fa = p["ia"] >= 0; fb = p["ib"] >= 0
            if p["same"]:
                kinds["same object, not collinear"] += 1
            elif fa and fb:
                kinds["different objects, collinear"] += 1
            elif fa != fb:
                kinds["object vs background, collinear"] += 1
            else:
                kinds["other"] += 1
        # the exact separating interval on this split: every T with
        # max-cross(same) <= T < min-cross(different) is exact here
        lo = max((max(cross(q["a"], q["b"])) for q in ps if q["same"]), default=0)
        hi = min((max(cross(q["a"], q["b"])) for q in ps if not q["same"]), default=10**9)
        out[split] = {"pairs": len(ps), "majority_baseline": base,
                      "exact_interval_lo": lo, "exact_interval_hi": hi - 1,
                      "exact_interval_empty": lo > hi - 1,
                      "best_T_on_this_split": best[0], "best_accuracy": best[1],
                      "accuracy_at_T192": sum((max(cross(p["a"], p["b"])) <= T192) == p["same"]
                                              for p in ps) / len(ps),
                      "errors_at_T192": len(errs), "error_kinds": dict(kinds),
                      "examples": [{k: p[k] for k in ("seed", "r", "c", "a", "b", "ia", "ib",
                                                      "same")} | {"cross": cross(p["a"], p["b"])}
                                   for p in errs[:8]]}
        DP.report(f"[{split}] pairs / base / exact interval / acc@192 / errors@192",
                  f"{len(ps)} / {base:.4f} / [{lo}, {hi-1}] / "
                  f"{out[split]['accuracy_at_T192']:.6f} / {len(errs)}")
        for k, v in kinds.items():
            DP.report(f"   [{split}] {k}", v)
        for e in out[split]["examples"][:4]:
            DP.report("   error", e)
    out["seconds"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "residual.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", OUT / "residual.json")


if __name__ == "__main__":
    main()
