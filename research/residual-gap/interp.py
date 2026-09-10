"""Re-assert the interpreter link myself, on a stratified sample.

`research/lazy-latency/out/equivalence.json` already records that arm A, arm B
and arm C agree with the shipped typed interpreter on a 32-record stratified
sample of the held-out set (the full 2,883 would cost 22 minutes at 0.454 s per
record).  This track inherits that, but the house rule is that a timing is
gated on bit-identical output **against the interpreter**, so the link is
re-run here rather than only cited -- extended to the whole ladder, and
stratified over achieved `w + h` so every control-flow class is covered
including the deployment corners.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "lazy-latency")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms                                                  # noqa: E402
import gate as GATE                                          # noqa: E402
import rungs as RUNGS                                        # noqa: E402

OUT = HERE / "out"
SEED = 20260909


def sample(g, per_stratum=1, extra_corners=6):
    """One record per achieved-`w + h` stratum, plus corner records."""
    by_k = collections.defaultdict(list)
    for r in g["uniform"]:
        by_k[arms.stage3_iterations(r)].append(r)
    rng = random.Random(SEED)
    picks = []
    for k in sorted(by_k):
        picks.extend(rng.sample(by_k[k], min(per_stratum, len(by_k[k]))))
    corners = rng.sample(g["deploy"], min(extra_corners, len(g["deploy"])))
    picks.extend(corners)
    picks.append(g["worst"])
    # de-duplicate by identity of (pos, id(obs))
    seen, out = set(), []
    for r in picks:
        key = (r[0], id(r[1]))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out, sorted(by_k), len(corners)


def main():
    t0 = time.time()
    grep, g, R = GATE.check()
    recs, strata, ncorner = sample(g)
    print("interpreter oracle on %d records (%d strata, %d corners)"
          % (len(recs), len(strata), ncorner), flush=True)

    rows = []
    for i, r in enumerate(recs):
        want = list(arms.interpreter_output(g, r))
        row = {"pos": r[0], "w_plus_h": arms.stage3_iterations(r),
               "interpreter": want, "agree": {}}
        for n in RUNGS.LADDER:
            row["agree"][n] = list(R[n]["module"].run(r)) == want
        row["agree"]["A"] = list(g["arms"]["A"]["value"](r)) == want
        rows.append(row)
        print("   %2d/%2d pos=%-5d k=%-3d %s"
              % (i + 1, len(recs), r[0], row["w_plus_h"],
                 "all agree" if all(row["agree"].values()) else row["agree"]),
              flush=True)

    rep = {"oracle": "tcn typed interpreter, Program.run via Value(input_type, rec)",
           "records": len(recs), "strata": strata, "corner_records": ncorner,
           "seconds": round(time.time() - t0, 1),
           "all_agree": all(all(x["agree"].values()) for x in rows),
           "per_rung_agreements": {n: sum(x["agree"][n] for x in rows)
                                   for n in RUNGS.LADDER + ["A"]},
           "rows": rows}
    (OUT / "interp.json").write_text(json.dumps(rep, indent=1))
    print("all_agree:", rep["all_agree"], "->", OUT / "interp.json")
    return rep


if __name__ == "__main__":
    main()
