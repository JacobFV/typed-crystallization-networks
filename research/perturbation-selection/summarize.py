"""Aggregate result files into the tables in RESULTS.md."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import LABELS  # noqa: E402

ORDER = ("entropy", "perturbation", "perturbation-total-guard",
         "argmax@entropy", "argmax@perturbation", "argmax0")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]]."""
    n = a + b + c + d
    def p(x):
        return (math.comb(a + b, x) * math.comb(c + d, a + c - x)) / math.comb(n, a + c)
    observed = p(a)
    lo = max(0, a + c - (c + d))
    hi = min(a + b, a + c)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= observed * (1 + 1e-9)))


def load(paths):
    rows = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["arm"]].append(r)
    return rows


def mixed_table(rows):
    n = max(len(v) for v in rows.values())
    out = [f"| arm | exact conformance | fully frozen | median loss | frozen nodes | optimizer steps | forward passes | sweep evals |",
           "|---|---|---|---|---|---|---|---|"]
    for arm in ORDER:
        rs = rows.get(arm)
        if not rs:
            continue
        out.append("| {} | {}/{} | {}/{} | {:.3g} | {:.3g} | {:.1f} | {:.1f} | {:.1f} |".format(
            LABELS.get(arm, arm),
            sum(bool(r["exact_conformance"]) for r in rs), len(rs),
            sum(bool(r["fully_frozen"]) for r in rs), len(rs),
            median([r["loss"] for r in rs]),
            mean([r["frozen_fraction"] for r in rs]),
            mean([r["total_steps"] for r in rs]),
            mean([r["total_forwards"] for r in rs]),
            mean([r["sweep_evaluations"] for r in rs])))
    return "\n".join(out), n


def joint_table(rows):
    out = ["| arm | fully frozen | det. mean return | frozen-program return | optimizer steps | forward passes | sweep evals | deferrals |",
           "|---|---|---|---|---|---|---|---|"]
    for arm in ORDER:
        rs = rows.get(arm)
        if not rs:
            continue
        returns = [r["post_crystallization"]["mean_return"] if r["post_crystallization"] else None for r in rs]
        frozen = [r["frozen_agent_mean_return"] for r in rs]
        deferrals = [sum(1 for e in r["freeze_events"] if not e["accepted"]) for r in rs]
        out.append("| {} | {}/{} | {:.3f} ± {:.3f} | {:.3f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} |".format(
            LABELS.get(arm, arm),
            sum(bool(r["fully_frozen"]) for r in rs), len(rs),
            mean(returns), sd(returns), mean(frozen),
            mean([r["total_steps"] for r in rs]),
            mean([r["total_forwards"] for r in rs]),
            mean([r["sweep_evaluations"] for r in rs]),
            mean(deferrals)))
    return "\n".join(out)


def reasons(rows):
    out = ["| arm | trials | " + " | ".join(("accepted", "degradation", "disconnected remaining region",
                                             "runtime conformance", "nonfinite loss", "invalid trial")) + " |",
           "|---|---|---|---|---|---|---|---|"]
    keys = ("accepted", "degradation", "disconnected remaining region",
            "runtime conformance", "nonfinite loss", "invalid trial")
    for arm in ORDER:
        rs = rows.get(arm)
        if not rs or arm.startswith("argmax"):
            continue
        counts = defaultdict(int)
        trials = 0
        for r in rs:
            for e in r["freeze_events"]:
                trials += 1
                counts["accepted" if e["accepted"] else e["reason"].split(":")[0]] += 1
        out.append("| {} | {} | ".format(LABELS.get(arm, arm), trials)
                   + " | ".join(str(counts[k]) for k in keys) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for path in args.files:
        rows = load([path])
        task = next(iter(rows.values()))[0]["task"]
        print(f"\n### {path}\n")
        if task == "mixed":
            table, n = mixed_table(rows)
            print(table)
            for a, b in (("perturbation", "entropy"), ("perturbation", "argmax@perturbation"),
                         ("entropy", "argmax@entropy")):
                if a in rows and b in rows:
                    ka = sum(bool(r["exact_conformance"]) for r in rows[a])
                    kb = sum(bool(r["exact_conformance"]) for r in rows[b])
                    na, nb = len(rows[a]), len(rows[b])
                    print(f"\nFisher exact, exact conformance {a} ({ka}/{na}) vs {b} ({kb}/{nb}): "
                          f"p = {fisher_exact(ka, na - ka, kb, nb - kb):.3g}")
        else:
            print(joint_table(rows))
        print("\n" + reasons(rows))


if __name__ == "__main__":
    main()
