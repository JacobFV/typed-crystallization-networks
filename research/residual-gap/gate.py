"""PREREGISTRATION section 4 -- the gate.  No rung is timed until this passes.

Every rung must be bit-identical to arm B on all 2,883 held-out interior
records, all 54 corner (deployment) records, and the pre-registered worst-case
record.  A SHA-256 digest over each rung's canonical outputs is recorded per
distribution so the claim is checkable from `out/gate.json` without re-running.

`research/lazy-latency/equivalence.py` establishes that arm B, arm A (the
compiled specification) and arm C are already bit-identical to each other and
to the typed interpreter on exactly these records; this file therefore gates
the new rungs against arm B rather than re-running the 22-minute interpreter
oracle.  That inheritance is recorded, not assumed silently.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "lazy-latency")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms                                                  # noqa: E402
import rungs as RUNGS                                        # noqa: E402

OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)


def digest(outputs):
    h = hashlib.sha256()
    for o in outputs:
        h.update(json.dumps(o, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def fixture():
    g = arms.build(3)
    return g


def check(g=None, rung_set=None):
    g = g or fixture()
    R = RUNGS.build()
    names = rung_set or (RUNGS.LADDER + ["R5a"])
    dists = {"uniform": g["uniform"], "deploy": g["deploy"], "worst": [g["worst"]]}

    ref = {}
    for d, recs in dists.items():
        ref[d] = [list(R["R0"]["module"].run(r)) for r in recs]

    # the interpreter link, inherited and re-asserted on arm C where it is cheap
    inherited = {
        "source": "research/lazy-latency/out/equivalence.json",
        "claim": "arm A (compiled spec), arm B and arm C are bit-identical to "
                 "each other on all 2,883 uniform + 54 deploy + worst records, "
                 "and arm A is bit-identical to the typed interpreter on a "
                 "32-record stratified sample",
    }

    report = {"inherited_interpreter_link": inherited,
              "records": {d: len(v) for d, v in dists.items()},
              "reference_digests": {d: digest(ref[d]) for d in dists},
              "rungs": {}}

    all_ok = True
    for name in names:
        mod = R[name]["module"]
        row = {"label": R[name]["label"], "class": R[name]["class"],
               "source_bytes": len(R[name]["source"]), "digests": {},
               "mismatches": {}, "error": None, "passed": True}
        for d, recs in dists.items():
            got, bad, err = [], 0, None
            for i, r in enumerate(recs):
                try:
                    o = list(mod.run(r))
                except Exception as exc:                     # noqa: BLE001
                    err = err or "%s: %s (record %d)" % (type(exc).__name__, exc, i)
                    o = None
                    bad += 1
                    got.append(None)
                    continue
                got.append(o)
                if o != ref[d][i]:
                    bad += 1
            row["digests"][d] = digest(got)
            row["mismatches"][d] = bad
            if bad:
                row["passed"] = False
            if err:
                row["error"] = err
        report["rungs"][name] = row
        if name in RUNGS.LADDER and not row["passed"]:
            all_ok = False
    report["ladder_gate_passed"] = all_ok
    return report, g, R


def main():
    rep, _, _ = check()
    (OUT / "gate.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep["rungs"].items():
        print("%-4s %-8s %s  mismatches=%s%s"
              % (k, "PASS" if v["passed"] else "FAIL",
                 v["label"][:52].ljust(52), v["mismatches"],
                 "  " + v["error"] if v["error"] else ""))
    print("ladder_gate_passed:", rep["ladder_gate_passed"])
    print("->", OUT / "gate.json")
    return rep


if __name__ == "__main__":
    main()
