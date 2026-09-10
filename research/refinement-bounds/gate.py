"""PREREGISTRATION section 2.2 -- the gates.  Nothing is timed before these pass.

The variable here is the **declared type**, not the emitter, so the reference is
arm A0 -- the artifact `research/emitter-guards/measure.py` timed -- compiled by
the same emitter.  Section 59's gate compared two emitters on one program; this
compares two programs under one emitter, which is the harder direction: arm B's
program is a *different* `Program` with different node types, and the claim is
that it computes the identical function.

Gate A   arm X vs arm A0, compiled, on 24 held-out screenshots, both
         `validate=True` and `validate=False`, value **or** exception type at
         the identical edge.
Gate B   arm X against the **typed interpreter** on the same cases (stratified,
         since section 48 measures ~17 s per `visual` case there).

`research/residual-gap/RESULTS.md` sec 2.1 records a transform that passed on 54
deployment records and failed on 243 of 2,883 held-out ones; a mismatch here is
reported in that form, `k / N`, and the arm is dropped rather than repaired.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "compiled-runtime"),
           str(ROOT / "research" / "emitter-guards")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util                                 # noqa: E402

import arms                                            # noqa: E402
from tcn.types import Value                            # noqa: E402

# section 59's outcome/canon/digest, imported rather than re-implemented.
_spec = importlib.util.spec_from_file_location(
    "emitter_guards_gate", ROOT / "research" / "emitter-guards" / "gate.py")
_EG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_EG)
canon, digest, outcome = _EG.canon, _EG.digest, _EG.outcome

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
SEEDS = arms.SEEDS                                     # 200..223, section 59's gate domain


def _interp(program, registry, types, c):
    ins = {k: Value.of(types[k], c[k]) for k in types}
    out, st = program.run(ins, registry=registry)
    return ({k: v.decoded for k, v in out.items()},
            {k: v.decoded for k, v in st.items()})


def run(test_arms=("A1", "B"), interp_limit=6, inline_bounded=None):
    fixtures = arms._track()
    from common import FLAT, episode
    eps = [episode(s, "test", **FLAT) for s in SEEDS]
    cases = [{"observation": tuple(e["pixels"]), "_w": e["width"], "_h": e["height"]}
             for e in eps]
    positions = sum(len(e["pixels"]) // 3 for e in eps)

    ref_f, ref_res, ref_mod = arms.compiled("A0", seeds=SEEDS)
    ref_rows = []
    for c in cases:
        for validate in (True, False):
            ref_rows.append(outcome(lambda: ref_mod.run(dict(c), validate=validate)))
    report = {"reference_arm": "A0",
              "reference_source_sha256": arms.source_sha(ref_res),
              "gate_domain": "%d held-out test screenshots (seeds %d-%d), %d interior "
                             "positions each" % (len(SEEDS), SEEDS[0], SEEDS[-1],
                                                 len(ref_f["program"].constants[0][1].decoded)),
              "screenshots": len(SEEDS), "raster_pixels_total": positions,
              "cases": len(cases), "arms": {}}

    for arm in test_arms:
        t0 = time.time()
        f, res, mod = arms.compiled(arm, seeds=SEEDS, inline_bounded=inline_bounded)
        rows, bad = [], []
        for i, c in enumerate(cases):
            for validate in (True, False):
                got = outcome(lambda: mod.run(dict(c), validate=validate))
                rows.append(got)
                if got != ref_rows[len(rows) - 1]:
                    bad.append({"case": i, "seed": SEEDS[i], "validate": validate,
                                "reference": ref_rows[len(rows) - 1], "arm": got})
        interp_bad, checked = [], 0
        types = dict(f["program"].inputs)
        for i, c in enumerate(cases[:interp_limit]):
            want = outcome(lambda: _interp(f["program"], f["registry"], types, c))
            got = outcome(lambda: mod.run(dict(c), validate=True))
            checked += 1
            if want != got:
                interp_bad.append({"case": i, "seed": SEEDS[i], "interpreter": want, "arm": got})
        report["arms"][arm] = {
            "source_sha256": arms.source_sha(res),
            "program_digest": f["program"].digest,
            "nodes_in_rect_module": None,
            "comparisons": len(rows),
            "gate_A_mismatches": len(bad),
            "gate_A_mismatch_fraction": "%d / %d" % (len(bad), len(rows)),
            "gate_A_examples": bad[:3],
            "gate_A_digest_reference": digest(ref_rows),
            "gate_A_digest_arm": digest(rows),
            "gate_A_passed": not bad and digest(ref_rows) == digest(rows),
            "gate_B_checked": checked,
            "gate_B_mismatches": len(interp_bad),
            "gate_B_examples": interp_bad[:2],
            "gate_B_passed": not interp_bad,
            "guards": {k: v for k, v in res.stats.items() if k.startswith("guard_")},
            "scalar_canonicalisations": res.stats["scalar_canonicalisations"],
            "source_bytes": res.stats["source_bytes"],
            "source_lines": res.stats["source_lines"],
            "seconds": round(time.time() - t0, 1),
        }
        print("  %s gate A %s (%s), gate B %s (%d checked), %.0fs" % (
            arm, "PASS" if report["arms"][arm]["gate_A_passed"] else "FAIL",
            report["arms"][arm]["gate_A_mismatch_fraction"],
            "PASS" if report["arms"][arm]["gate_B_passed"] else "FAIL", checked,
            time.time() - t0), flush=True)
    report["all_passed"] = all(a["gate_A_passed"] and a["gate_B_passed"]
                               for a in report["arms"].values())
    return report


def main(argv):
    inline = "--inline-bounded" in argv
    test_arms = [a for a in argv[1:] if not a.startswith("-")] or ["A1", "B"]
    report = run(tuple(test_arms), inline_bounded=True if inline else None)
    name = "gate_inline.json" if inline else "gate.json"
    (OUT / name).write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print("all_passed:", report["all_passed"])


if __name__ == "__main__":
    main(sys.argv)
