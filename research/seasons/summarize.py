"""Aggregate result files into the tables in RESULTS.md."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import LABELS  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]]."""
    n = a + b + c + d
    if n == 0:
        return float("nan")

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


def summarize(rows, task):
    out = []
    for arm, rs in rows.items():
        n = len(rs)
        rec = {"arm": arm, "n": n,
               "conformant": sum(1 for r in rs if r.get("exact_conformance")),
               "fully_frozen": sum(1 for r in rs if r.get("fully_frozen")),
               "steps": mean([r.get("total_steps") for r in rs]),
               "forwards": mean([r.get("total_forwards") for r in rs]),
               "wall": mean([r.get("wall") for r in rs]),
               "bits": mean([r.get("description_bits") for r in rs])}
        if task == "joint":
            rec["episodes"] = mean([r.get("total_episodes")
                                    if r.get("total_episodes") is not None
                                    else (r.get("episodes", 0) + r.get("crystallization_episodes", 0))
                                    for r in rs])
            returns = [r.get("frozen_agent_mean_return") for r in rs]
            rec["return"] = mean([x if x is not None else 0.0 for x in returns])
            rec["return_sd"] = sd([x if x is not None else 0.0 for x in returns])
            rec["best_member_return"] = mean([r["best_member_return"] for r in rs
                                              if r.get("best_member_return") is not None]) \
                if any("best_member_return" in r for r in rs) else None
        rec["any_conformant"] = sum(1 for r in rs if r.get("any_conformant")) \
            if any("any_conformant" in r for r in rs) else None
        rec["seasons_run"] = mean([r.get("seasons_run") for r in rs if r.get("seasons_run") is not None])
        rec["thaw_count"] = mean([r.get("thaw_count") for r in rs if r.get("thaw_count") is not None])
        out.append(rec)
    return sorted(out, key=lambda r: r["arm"])


def fmt(x, spec=".1f"):
    return "" if x is None or x != x else format(x, spec)


def table(recs, task):
    if task == "joint":
        head = ("| arm | frozen return | conformant | opt steps | forwards | episodes | seasons | thaws |\n"
                "|---|---|---|---|---|---|---|---|\n")
        return head + "".join(
            "| `{arm}` | {ret} +- {sd} | {c}/{n} | {steps} | {fwd} | {eps} | {seasons} | {thaws} |\n".format(
                arm=r["arm"], ret=fmt(r["return"], ".3f"), sd=fmt(r["return_sd"], ".3f"),
                c=r["conformant"], n=r["n"], steps=fmt(r["steps"], ".0f"),
                fwd=fmt(r["forwards"], ".0f"), eps=fmt(r["episodes"], ".0f"),
                seasons=fmt(r["seasons_run"]), thaws=fmt(r["thaw_count"]))
            for r in recs)
    head = ("| arm | conformant | fully frozen | opt steps | forwards | pruned bits | seasons | thaws |\n"
            "|---|---|---|---|---|---|---|---|\n")
    return head + "".join(
        "| `{arm}` | {c}/{n} | {f}/{n} | {steps} | {fwd} | {bits} | {seasons} | {thaws} |\n".format(
            arm=r["arm"], c=r["conformant"], n=r["n"], f=r["fully_frozen"],
            steps=fmt(r["steps"], ".0f"), fwd=fmt(r["forwards"], ".0f"),
            bits=fmt(r["bits"], ".0f"), seasons=fmt(r["seasons_run"]), thaws=fmt(r["thaw_count"]))
        for r in recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["mixed", "joint"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = load(args.paths)
    recs = summarize(rows, args.task)
    if args.json:
        print(json.dumps(recs, indent=2))
        return
    print(table(recs, args.task))
    # Every arm against its step-matched argmax partner, and against argmax0.
    by = {r["arm"]: r for r in recs}
    for arm in list(by):
        for partner in (f"argmax@{arm}", "argmax0"):
            if arm.startswith("argmax") or partner not in by:
                continue
            a, b = by[arm], by[partner]
            p = fisher_exact(a["conformant"], a["n"] - a["conformant"],
                             b["conformant"], b["n"] - b["conformant"])
            print(f"{arm} {a['conformant']}/{a['n']} vs {partner} {b['conformant']}/{b['n']}: "
                  f"Fisher p = {p:.4g}")
    for arm in by:
        if arm not in LABELS and not arm.startswith("argmax@"):
            print(f"note: {arm} has no label in arms.LABELS")


if __name__ == "__main__":
    main()
