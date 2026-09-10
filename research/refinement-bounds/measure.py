"""PREREGISTRATION section 2.3 -- three cost measures on the four arms.

Timing protocol is `research/lazy-latency/latency.py`'s, reached through
`research/emitter-guards/measure.py` so that no third harness is built: the
statistics (`summarize`, `ratio_ci`), the sweep discipline (`sweep_ns`, `pin`)
and the bootstrap are that file's, imported.

Arms are timed **round-robin inside every repeat**, so all of them see the same
drift.  Arm `N` is arm `A0` compiled a second time -- **byte-identical source** --
and is the null control that fixes this host's instrument floor at `visual`'s
own magnitude, in this session rather than by citation.

**No timing is recorded before bit-identity is re-asserted on the exact cases
about to be timed.**  That is separate from, and additional to, `gate.py`.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "compiled-runtime"),
           str(ROOT / "research" / "lazy-guard"), str(ROOT / "research" / "emitter-guards")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cost as lazy_cost                               # noqa: E402  the bytecode meter
import arms as ARMS                                    # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EG_GATE = _load("eg_gate", "research/emitter-guards/gate.py")
_EG_MEASURE = _load("eg_measure", "research/emitter-guards/measure.py")
outcome, digest = _EG_GATE.outcome, _EG_GATE.digest
summarize, ratio_ci, sweep_ns, pin, loadavg = (_EG_MEASURE.summarize, _EG_MEASURE.ratio_ci,
                                               _EG_MEASURE.sweep_ns, _EG_MEASURE.pin,
                                               _EG_MEASURE.loadavg)

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
REPEATS = 21
MIN_SWEEP_NS = 2_000_000
SEEDS = (200, 201, 202)                                # the three timed cases, as section 59


def build(arm_names):
    """`B+` is arm B compiled with `inline_bounded=True`; every other arm is
    compiled exactly as `main` compiles it."""
    built = {}
    for arm in arm_names:
        real, ib = ("B", True) if arm == "B+" else (arm, None)
        f, res, mod = ARMS.compiled(real, seeds=SEEDS, inline_bounded=ib)
        built[arm] = {"fixture": f, "result": res, "module": mod,
                      "sha": hashlib.sha256(res.source.encode()).hexdigest()}
    return built


def identity_before_timing(built, cases, ref="A0"):
    rows = {}
    for arm, b in built.items():
        rows[arm] = [outcome(lambda c=c, m=b["module"], v=v: m.run(dict(c), validate=v))
                     for c in cases for v in (True, False)]
    return {"comparisons": len(rows[ref]),
            "digests": {a: digest(r) for a, r in rows.items()},
            "identical_to_%s" % ref: {a: (r == rows[ref]) for a, r in rows.items()},
            "all_identical": all(r == rows[ref] for r in rows.values())}


def primitives(program, registry, cases):
    """Executed leaf operators -- a property of the *program*, per arm."""
    from tcn.types import Value
    types = dict(program.inputs)
    real = registry.exact
    tally = {"n": 0}

    def counted(op, args):
        if not op.name.startswith("module:"):
            tally["n"] += 1
        return real(op, args)

    per_case = []
    registry.exact = counted
    try:
        for c in cases:
            tally["n"] = 0
            program.run({k: Value.of(types[k], c[k]) for k in types}, registry=registry)
            per_case.append(tally["n"])
    finally:
        registry.exact = real
    return {"instrument": "Registry.exact applications, module calls excluded",
            "cases": len(per_case), "worst_ops": max(per_case), "best_ops": min(per_case),
            "expected_ops": statistics.mean(per_case)}


def bytecodes(built, cases):
    out = {}
    for arm, b in built.items():
        counts = [lazy_cost.count_opcodes(lambda c=c, m=b["module"]: m.run(dict(c), validate=False))
                  for c in cases]
        out[arm] = {"expected": statistics.mean(counts), "worst": max(counts),
                    "best": min(counts), "cases": len(counts)}
    return out


def timings(built, cases, repeats=REPEATS):
    names = list(built)
    calls = {a: (lambda c, m=built[a]["module"]: m.run(dict(c), validate=False)) for a in names}
    probe = max(1, sweep_ns(calls[names[0]], cases))
    mult = max(1, math.ceil(MIN_SWEEP_NS / probe))
    recs = list(cases) * mult
    for a in names:                                     # warm-up, discarded
        calls[a](cases[0])
        sweep_ns(calls[a], recs)
    samples = {a: [] for a in names}
    for _ in range(repeats):
        for a in names:                                 # round-robin inside each repeat
            samples[a].append(sweep_ns(calls[a], recs) / 1e3 / len(recs))
    return samples, {"sweep_calls": len(recs), "case_multiplier": mult}


def run(arm_names=("A0", "N", "A1", "B", "B+"), repeats=REPEATS):
    t0 = time.time()
    built = build(arm_names)
    p0 = built["A0"]["fixture"]["program"]
    cases = [{k: c[k] for k, _ in p0.inputs} for c in built["A0"]["fixture"]["cases"]]
    ident = identity_before_timing(built, cases)
    rep = {"arms": list(arm_names), "cases": len(cases), "repeats": repeats,
           "source_sha256": {a: b["sha"] for a, b in built.items()},
           "program_digest": {a: b["fixture"]["program"].digest for a, b in built.items()},
           "guards": {a: {k: v for k, v in b["result"].stats.items() if k.startswith("guard_")}
                      for a, b in built.items()},
           "scalar_canonicalisations": {a: b["result"].stats["scalar_canonicalisations"]
                                        for a, b in built.items()},
           "source_bytes": {a: b["result"].stats["source_bytes"] for a, b in built.items()},
           "identity_before_timing": ident}
    rep["null_control_pairs"] = [[a, c] for i, a in enumerate(arm_names)
                                 for c in arm_names[i + 1:]
                                 if built[a]["sha"] == built[c]["sha"]]
    if not ident["all_identical"]:
        rep["timed"] = False
        rep["reason"] = "bit-identity failed on the timed cases; nothing timed"
        return rep
    rep["timed"] = True
    rep["primitives"] = {a: primitives(b["fixture"]["program"], b["fixture"]["registry"], cases)
                         for a, b in built.items()}
    rep["bytecodes"] = bytecodes(built, cases)
    rep["cpu_affinity"] = pin()
    rep["loadavg_before"] = loadavg()
    samples, shape = timings(built, cases, repeats)
    rep["latency_us"] = {a: summarize(v) for a, v in samples.items()}
    rep["sweep"] = shape
    rep["loadavg_after"] = loadavg()
    rep["ratios_A0_over_arm"] = {a: ratio_ci(samples["A0"], samples[a]) for a in arm_names}
    rep["bytecode_ratios_A0_over_arm"] = {
        a: rep["bytecodes"]["A0"]["expected"] / rep["bytecodes"][a]["expected"]
        for a in arm_names}
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    tag = args.pop(0) if args and args[0].startswith("run") else "run1"
    names = tuple(args) if args else ("A0", "N", "A1", "B", "B+")
    rep = run(names)
    path = OUT / ("measure_%s.json" % tag)
    path.write_text(json.dumps(rep, indent=1, default=str))
    print("null-control pairs (byte-identical source):", rep["null_control_pairs"])
    print("identity:", rep["identity_before_timing"]["identical_to_A0"])
    if rep["timed"]:
        for a in names:
            r = rep["ratios_A0_over_arm"][a]
            print("%-4s bytecodes %10.1f (A0/x %.4f)   wall %9.2f us   A0/x %.4f "
                  "CI[%.4f, %.4f]%s"
                  % (a, rep["bytecodes"][a]["expected"], rep["bytecode_ratios_A0_over_arm"][a],
                     rep["latency_us"][a]["median_us"], r["ratio"], r["ci95_lo"], r["ci95_hi"],
                     "  (contains 1)" if r["contains_1"] else ""))
    else:
        print("NOT TIMED:", rep["reason"])
    print("->", path)


if __name__ == "__main__":
    main(sys.argv)
