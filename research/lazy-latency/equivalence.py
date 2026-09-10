"""PREREGISTRATION section 4 -- the gate.  No timing is reported unless this passes.

Two links, exactly the oracle chain `research/lazy-guard` section 6.1 declared and
FINDINGS section 48 certified:

  interpreter == A      on a stratified sample (stage 3) or 1,024 inputs (1, 2)
  A == B (== C)         bit-identical on EVERY record of EVERY distribution used

Digests over the concatenated canonical outputs are written to
`out/equivalence.json` so a reader can re-check without re-running.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arms  # noqa: E402
from tcn.types import canonical  # noqa: E402

OUT = arms.OUT
INTERP_SAMPLE_MINI = 1024
INTERP_SAMPLE_S3 = 32


def canon(x):
    return json.dumps(canonical(x), sort_keys=True, separators=(",", ":"))


def digest(values):
    h = hashlib.sha256()
    for v in values:
        h.update(canon(v).encode())
        h.update(b"\n")
    return h.hexdigest()


def stratified_sample(g, k):
    """Records covering every achieved-extent stratum, plus every corner stratum."""
    by = {}
    for r in g["uniform"]:
        by.setdefault(arms.stage3_extent(r), r)
    for r in g["deploy"]:
        by.setdefault(("corner", arms.stage3_extent(r)), r)
    picks = [by[k2] for k2 in sorted(by, key=repr)]
    return picks[:k] if len(picks) > k else picks


def check(stage, verbose=True):
    t0 = time.time()
    g = arms.build(stage)
    rep = {"stage": stage,
           "distributions": {"uniform": len(g["uniform"]),
                             "deploy": len(g["deploy"]) if g["deploy"] else 0,
                             "worst": 1},
           "worst_note": g["worst_note"],
           "arms": {k: v["what"] for k, v in g["arms"].items()}}

    # ---- link 1: the shipped typed interpreter == arm A -------------------
    if stage == 3:
        sample = stratified_sample(g, INTERP_SAMPLE_S3)
        why = ("the interpreter costs ~0.45 s per record on the 3,072-byte "
               "carrier, so a stratified sample, not the full 2,883")
    else:
        step = max(1, len(g["full"]) // INTERP_SAMPLE_MINI)
        sample = g["full"][::step][:INTERP_SAMPLE_MINI]
        why = "1,024 inputs of the exhausted domain"
    t1 = time.time()
    bad = None
    for i, rec in enumerate(sample):
        want = arms.interpreter_output(g, rec)
        got = g["arms"]["A"]["value"](rec)
        if canon(want) != canon(got):
            bad = {"index": i, "want": repr(want)[:200], "got": repr(got)[:200]}
            break
    rep["interpreter_equals_A"] = {
        "checked": len(sample) if bad is None else bad["index"],
        "exact": bad is None, "counterexample": bad, "sampling": why,
        "seconds": round(time.time() - t1, 2)}
    if verbose:
        print("  interpreter == A on %d: %s (%.1fs)"
              % (len(sample), bad is None, time.time() - t1))

    # ---- link 2: A == B (== C) on every record of every distribution ------
    dists = {"uniform": g["uniform"]}
    if g["deploy"]:
        dists["deploy"] = g["deploy"]
    dists["worst"] = [g["worst"]]
    if stage in (1, 2) and len(g["uniform"]) != len(g["full"]):
        dists["full_domain"] = g["full"]

    rep["arm_equivalence"] = {}
    all_ok = bad is None
    for dname, recs in dists.items():
        digs, mism = {}, None
        for aname, arm in sorted(g["arms"].items()):
            vals = [arm["value"](r) for r in recs]
            digs[aname] = digest(vals)
            if aname == "A":
                ref = vals
            elif mism is None:
                for i, (u, v) in enumerate(zip(ref, vals)):
                    if canon(u) != canon(v):
                        mism = {"arm": aname, "index": i, "record": repr(recs[i])[:120],
                                "A": repr(u)[:160], aname: repr(v)[:160]}
                        break
        ok = mism is None and len(set(digs.values())) == 1
        all_ok = all_ok and ok
        rep["arm_equivalence"][dname] = {
            "records": len(recs), "bit_identical": ok, "digests": digs,
            "counterexample": mism}
        if verbose:
            print("  %-12s %5d records  A==B%s : %s"
                  % (dname, len(recs), "==C" if "C" in g["arms"] else "", ok))

    rep["gate_passed"] = all_ok
    rep["seconds"] = round(time.time() - t0, 2)
    return rep, g


def main():
    report = {}
    for stage in (1, 2, 3):
        print("== equivalence gate, stage %d" % stage)
        rep, _ = check(stage)
        report["stage%d" % stage] = rep
    report["all_gates_passed"] = all(v["gate_passed"] for v in report.values()
                                     if isinstance(v, dict))
    (OUT / "equivalence.json").write_text(json.dumps(report, indent=1, default=str))
    print("-> %s   all gates passed: %s"
          % (OUT / "equivalence.json", report["all_gates_passed"]))
    return report


if __name__ == "__main__":
    main()
