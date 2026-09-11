"""Measured per-edit decision rate on a real defective scaffold, for the budget.

Times a deterministic sample of a case's edits -- every k-th edit in the
enumerator's own order, so the sample spans the whole space-size distribution
rather than the cheap head -- and reports the mean, the worst, and the projected
cost of the whole case.  No repair label is read into any claim.

    python rate.py <domain> [--sample 24]
"""
from __future__ import annotations

import argparse
import time

import kit
import domains
import edits as E
import run_domain as R
from tcn.search import space_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--defect-index", type=int, default=0,
                    help="which admissible defect to time, in enumeration order")
    a = ap.parse_args()
    d = domains.BUILDERS[a.domain]()
    base, r, sig = d["program"], d["registry"], d["signals"]
    allep = d["train"] + d["admission"]
    defects = R.enumerate_defects(base, r, [nd.name for nd in base.nodes])
    case, seen = None, -1
    for kind, site, arg in defects:
        f = R.APPLY[kind](base, r, site, arg)
        if f is None:
            continue
        tr = R.decide(f, d["train"], sig, r)
        if tr["decided"] and not tr["conforming"] and tr["exhausted"]:
            seen += 1
            if seen == a.defect_index:
                case = (f, f"{kind}:{site}:{arg}")
                break
    if case is None:
        raise SystemExit(f"{a.domain}: no admissible defect")
    failed, cid = case
    es = E.enumerate_edits(failed, r)
    over = sum(1 for _, q in es if space_size(q) > R.MAX_SPACE)
    step = max(1, len(es) // a.sample)
    times, n = [], 0
    for edit, q in es[::step]:
        if space_size(q) > R.MAX_SPACE:
            continue
        t = time.perf_counter()
        on_all = R.decide(q, allep, sig, r)
        if on_all["decided"] and not on_all["conforming"]:
            R.decide(q, d["train"], sig, r)
        times.append(time.perf_counter() - t)
        n += 1
    mean = sum(times) / len(times)
    decided = len(es) - over
    out = {"domain": a.domain, "first_admissible_defect": cid,
           "n_edits": len(es), "over_cap": over, "decided_edits": decided,
           "sampled": n, "mean_seconds_per_edit": round(mean, 3),
           "worst_seconds_per_edit": round(max(times), 3),
           "projected_seconds_per_case": round(mean * decided, 1),
           "projected_minutes_per_case": round(mean * decided / 60, 1),
           "episodes_all": len(allep), "episodes_train": len(d["train"])}
    print(kit.dump(f"rate_{a.domain}_{a.defect_index}", out))
    print(out)


if __name__ == "__main__":
    main()
