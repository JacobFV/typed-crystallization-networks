"""PREREGISTRATION section 5 -- the timing, on the protocol inherited whole.

`research/lazy-latency/latency.py` supplies `median_ci`, `summarize`,
`ratio_ci`, `sweep_ns` and `pin`; `research/compiled-runtime/harness.py`
supplies the gc discipline underneath them.  Neither is re-implemented here.

Two things are timed in the same interleave so they share drift:

* **the reproduction** -- arms A, B and C envelope-matched exactly as
  FINDINGS section 51 timed them, so B / C = 6.208x can be checked on this host
  before anything is attributed to it;
* **the ladder** -- R0..R7 on the bare entry, which is the attribution.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "lazy-latency"),
           str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms                                                  # noqa: E402
import latency as LL                                         # noqa: E402  the protocol
import gate as GATE                                          # noqa: E402
import rungs as RUNGS                                        # noqa: E402

OUT = HERE / "out"
REPEATS = 21


def build_entries(g, R):
    """Every timed entry: the three arms envelope-matched, the rungs bare."""
    okey = g["program"].outputs[0][0]
    ent = {}
    for a in ("A", "B", "C"):
        ent["env:" + a] = {"call": g["arms"][a]["call"],
                           "what": "envelope-matched " + g["arms"][a]["what"]}
    for name in RUNGS.LADDER:
        ent["bare:" + name] = {"call": R[name]["module"].run,
                               "what": R[name]["label"]}
    # bare arm B and bare arm C, for the boundary/envelope decomposition
    ent["bare:B"] = {"call": g["arms"]["B"]["bare"], "what": "bare arm B"}
    ent["bare:C"] = {"call": g["arms"]["C"]["bare"], "what": "bare arm C"}
    return ent, okey


def measure(g, ent, repeats=REPEATS):
    dists = {"deploy": g["deploy"], "uniform": g["uniform"],
             "worst": [g["worst"]] * len(g["uniform"])}
    names = list(ent)
    for d in dists.values():
        for a in names:
            ent[a]["call"](d[0])
            LL.sweep_ns(ent[a]["call"], d[:64])
    samples = {d: {a: [] for a in names} for d in dists}
    for rep in range(repeats):
        for dname, recs in dists.items():
            for a in names:
                ns = LL.sweep_ns(ent[a]["call"], recs)
                samples[dname][a].append(ns / 1e3 / len(recs))
        print("   repeat %d/%d" % (rep + 1, repeats), flush=True)
    return samples


def main(repeats=REPEATS, tag="run1"):
    t0 = time.time()
    print("== gate")
    grep, g, R = GATE.check()
    (OUT / "gate.json").write_text(json.dumps(grep, indent=1))
    if not grep["ladder_gate_passed"]:
        print("GATE FAILED -- no timing recorded")
        return {"gate": grep, "timed": False}

    ent, okey = build_entries(g, R)
    print("== timing %d entries, %d repeats" % (len(ent), repeats))
    rep = {"protocol": "research/residual-gap/PREREGISTRATION.md sections 4-5",
           "inherits": "research/lazy-latency/latency.py",
           "python": sys.version.split()[0], "cpu_pinned_to": LL.CPU,
           "cpu_affinity": LL.pin(), "loadavg_before": LL.load(),
           "repeats": repeats, "tag": tag,
           "gate_passed": True,
           "gate_failed_rungs": [k for k, v in grep["rungs"].items()
                                 if not v["passed"]],
           "entries": {k: v["what"] for k, v in ent.items()},
           "distribution_sizes": {"deploy": len(g["deploy"]),
                                  "uniform": len(g["uniform"]), "worst": 1},
           "worst_note": g["worst_note"]}

    raw = measure(g, ent, repeats)
    rep["raw_us"] = raw
    rep["summary_us"] = {d: {a: LL.summarize(v) for a, v in per.items()}
                         for d, per in raw.items()}

    # every ratio the attribution needs
    ratios = {}
    for d, per in raw.items():
        r = {}
        r["A_over_B_envelope"] = LL.ratio_ci(per["env:B"], per["env:A"])
        r["A_over_C_envelope"] = LL.ratio_ci(per["env:C"], per["env:A"])
        r["B_over_C_envelope"] = LL.ratio_ci(per["env:C"], per["env:B"])
        r["B_over_C_bare"] = LL.ratio_ci(per["bare:C"], per["bare:B"])
        r["R0_over_R7_bare"] = LL.ratio_ci(per["bare:R7"], per["bare:R0"])
        for i in range(len(RUNGS.LADDER) - 1):
            a, b = RUNGS.LADDER[i], RUNGS.LADDER[i + 1]
            r["%s_over_%s" % (a, b)] = LL.ratio_ci(per["bare:" + b],
                                                   per["bare:" + a])
        ratios[d] = r
    rep["ratios"] = ratios
    rep["loadavg_after"] = LL.load()
    rep["seconds"] = round(time.time() - t0, 1)
    p = OUT / ("latency_%s.json" % tag)
    p.write_text(json.dumps(rep, indent=1, default=str))
    print("->", p)

    per = rep["summary_us"]["deploy"]
    print("\ndeployment distribution, median us/record")
    for k in ent:
        print("   %-10s %8.3f" % (k, per[k]["median_us"]))
    print("   A/B env %.3f   A/C env %.3f   B/C env %.3f   B/C bare %.3f"
          % (ratios["deploy"]["A_over_B_envelope"]["ratio"],
             ratios["deploy"]["A_over_C_envelope"]["ratio"],
             ratios["deploy"]["B_over_C_envelope"]["ratio"],
             ratios["deploy"]["B_over_C_bare"]["ratio"]))
    return rep


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "run1"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else REPEATS
    main(n, tag)
