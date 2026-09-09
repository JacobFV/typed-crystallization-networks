"""Task 5: partial credit under noisy targets.

The fair-to-gradient claim is that a relaxation gives partial credit when no
program conforms exactly, whereas a decision procedure only answers yes/no.
That is true of SAT as a *decision* procedure and false of enumeration in
general: enumeration with a scoring objective (minimum Hamming distance to the
corrupted labels) is a MaxSAT-style optimiser and handles noise directly.

Protocol: draw a depth-`d` target from `generators/logic`, flip `f` of the 16
truth-table rows, hand the corrupted table to each method, and ask whether the
method recovers the UNCORRUPTED function.
  * enumeration_best  -- complete search, keep the minimum-Hamming program
  * sat_exact         -- the decision encoding on the corrupted table
  * gradient          -- the TCN path, BCE on the corrupted table
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import write, machine, Timer
from scaling import (FULL, INPUT_MASKS, N_ROWS, apply_table, gradient,
                     sat_solve, target_from_generator, verify)


def corrupt(mask, flips, seed):
    rng = random.Random(seed)
    rows = rng.sample(range(N_ROWS), flips)
    out = mask
    for r in rows:
        out ^= 1 << r
    return out


def enumerate_best(depth, noisy):
    """Complete search keeping the minimum-Hamming-distance program."""
    best = (99, None)
    path = []

    def rec(k, vals):
        nonlocal best
        n = len(vals)
        last = k == depth - 1
        for a in range(n):
            am = vals[a]
            for b in range(n):
                bm = vals[b]
                na, nb = FULL ^ am, FULL ^ bm
                parts = (na & nb, na & bm, am & nb, am & bm)
                for tb in range(16):
                    out = 0
                    if tb & 1:
                        out |= parts[0]
                    if tb & 2:
                        out |= parts[1]
                    if tb & 4:
                        out |= parts[2]
                    if tb & 8:
                        out |= parts[3]
                    if last:
                        h = bin((out ^ noisy) & FULL).count("1")
                        if h < best[0]:
                            best = (h, list(path) + [(tb, a, b)], out)
                    else:
                        path.append((tb, a, b))
                        vals.append(out)
                        rec(k + 1, vals)
                        vals.pop()
                        path.pop()

    with Timer() as t:
        rec(0, list(INPUT_MASKS))
    return {"hamming": best[0], "program": best[1], "mask": best[2], "seconds": t.seconds}


def run(depth=2, flips=(0, 1, 2, 3), trials=4, gradient_seeds=2):
    rows = []
    for f in flips:
        for ti in range(trials):
            _, clean = target_from_generator(depth, 0, 1000 * ti + depth)
            noisy = corrupt(clean, f, seed=ti)
            e = enumerate_best(depth, noisy)
            s = sat_solve(depth, noisy, max_conflicts=200000)
            gs = [gradient(depth, noisy, seed=sd) for sd in range(gradient_seeds)]
            # "recovered" = the returned program computes the UNCORRUPTED function
            grad_recovered = sum(g["final_mask"] == clean for g in gs) / len(gs)
            rows.append({
                "depth": depth, "flips": f, "trial": ti,
                "clean_mask": clean, "noisy_mask": noisy,
                "enumeration": {"seconds": e["seconds"], "hamming_to_noisy": e["hamming"],
                                "recovered_clean": e["mask"] == clean,
                                "hamming_to_clean": bin((e["mask"] ^ clean) & FULL).count("1")},
                "sat_exact": {"solved": s["solved"], "seconds": s["seconds"],
                              "recovered_clean": bool(s["program"]) and verify(depth, s["program"], clean)},
                "gradient": {"fit_noisy_rate": sum(g["solved"] for g in gs) / len(gs),
                             "recovered_clean_rate": grad_recovered,
                             "hamming_to_noisy": [g["hamming"] for g in gs],
                             "hamming_to_clean": [bin((g["final_mask"] ^ clean) & FULL).count("1") for g in gs],
                             "seconds_median": sorted(g["seconds"] for g in gs)[len(gs) // 2]},
            })
            print(f"flips={f} trial={ti} enum_h={e['hamming']} enum_recovered={rows[-1]['enumeration']['recovered_clean']} "
                  f"sat={'SAT' if s['solved'] else 'UNSAT'} grad_fit={rows[-1]['gradient']['fit_noisy_rate']:.2f}",
                  flush=True)
    return rows


if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows = run(depth=depth)
    print(write(f"noise_d{depth}.json", {"machine": machine(), "rows": rows}))
