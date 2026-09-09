"""Evaluate the frozen TCN mixed program on exactly the same metric set as the MLP."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tcn.runtime import load_program, benchmark  # noqa: E402
from tcn.types import BOOL, Value, floating  # noqa: E402

F = floating()
TRAIN_X = [-0.7, -0.2, 0.3, 0.8]
AB = [(False, False), (False, True), (True, False), (True, True)]


def errors(p, r, xs):
    worst, sq, n = 0.0, 0.0, 0
    for a, b in AB:
        for x in xs:
            out, _ = p.run({"a": Value.of(BOOL, a), "b": Value.of(BOOL, b), "x": Value.of(F, x)}, registry=r)
            e = abs(out["answer"].decoded - math.sin(float(a != b) + x))
            worst = max(worst, e)
            sq += e * e
            n += 1
    return {"maxabs": worst, "rmse": (sq / n) ** 0.5}


def main(path="research/baselines/out/tcn_mixed/program.json"):
    p, r = load_program(path)
    grid = [v for v in (round(-0.7 + 0.01 * i, 3) for i in range(151)) if v not in TRAIN_X]
    ex = [v for v in (round(-2.0 + 0.05 * i, 3) for i in range(81)) if v < -0.7 or v > 0.8]
    pts = errors(p, r, [0.13, -0.43])
    res = {
        "train": errors(p, r, TRAIN_X),
        "interp": errors(p, r, grid),
        "extrap": errors(p, r, ex),
        "repo_test_points_maxabs": max(
            abs(p.run({"a": Value.of(BOOL, True), "b": Value.of(BOOL, False), "x": Value.of(F, x)}, registry=r)[0]["answer"].decoded - math.sin(1 + x))
            for x in (0.13, -0.43)),
        "all_ab_at_repo_test_points": pts,
        "benchmark": benchmark(p, [{"a": Value.of(BOOL, a), "b": Value.of(BOOL, b), "x": Value.of(F, x)}
                                   for a, b in AB for x in TRAIN_X], registry=r),
    }
    print(json.dumps(res, indent=2))
    Path("research/baselines/out/tcn_mixed_eval.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
