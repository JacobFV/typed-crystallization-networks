"""PREREGISTRATION section 3 -- the gates.  Nothing is timed before these pass.

Gate A  bit-identity against the compiler at `ee7d63c`, on every case, under
        both `validate=True` and `validate=False`, output dict and state dict,
        equal value *or* equal exception type at the identical edge.
Gate B  bit-identity against the typed interpreter (`Program.run` on `Value`),
        which is the oracle `tcn/compile.py` has always answered to.

The reference compiler is read out of git rather than copied, so it cannot
silently drift from the thing this track claims to be identical to.  Its three
relative imports are rewritten to absolute ones and it is executed as a
standalone module; nothing else about it is changed.

The gate domains are deliberately much larger than the four timing cases:
`research/residual-gap/RESULTS.md` sec 2.1 records a transform that passed on the
54-record deployment distribution and failed on 243 of 2,883 held-out records.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time
import types as _types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                        # noqa: E402
from tcn.compile import compile_program                # noqa: E402
from tcn.types import Value                            # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
BASE_REV = "ee7d63c"


# ---------------------------------------------------------------------------
# the reference compiler, read out of git
# ---------------------------------------------------------------------------
def baseline_compiler(rev=BASE_REV):
    src = subprocess.run(["git", "-C", str(ROOT), "show", "%s:tcn/compile.py" % rev],
                         check=True, capture_output=True, text=True).stdout
    src = (src.replace("from .graph import", "from tcn.graph import")
              .replace("from .operators import", "from tcn.operators import")
              .replace("from .types import", "from tcn.types import"))
    mod = _types.ModuleType("tcn_compile_baseline")
    mod.__dict__["__file__"] = "<%s:tcn/compile.py>" % rev
    exec(compile(src, mod.__dict__["__file__"], "exec"), mod.__dict__)
    return mod, hashlib.sha256(src.encode()).hexdigest()


# ---------------------------------------------------------------------------
# canonical, hashable rendering of an outcome
# ---------------------------------------------------------------------------
def canon(x):
    if isinstance(x, frozenset) or isinstance(x, set):
        return ["set"] + sorted(json.dumps(canon(v), sort_keys=True) for v in x)
    if isinstance(x, tuple) or isinstance(x, list):
        return [canon(v) for v in x]
    if isinstance(x, dict):
        return {k: canon(v) for k, v in sorted(x.items())}
    if isinstance(x, bool):
        return ["bool", x]
    if isinstance(x, float):
        return ["float", repr(x)]                       # exact, including -0.0 / nan
    return x


def outcome(fn):
    """Either the canonical value or the exception type and message."""
    try:
        out, state = fn()
        return {"ok": True, "out": canon(out), "state": canon(state)}
    except Exception as exc:                            # the error contract is semantics
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def digest(records):
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps(r, sort_keys=True, default=str).encode())
        h.update(b"\x1e")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# gate domains -- much wider than the timing cases
# ---------------------------------------------------------------------------
def gate_cases(name, f):
    """Every case the gate runs, and a one-line description of the domain."""
    if name == "mixed":
        xs = [(-100 + i) / 100.0 for i in range(0, 201, 2)]
        cases = [{"a": a, "b": b, "x": x}
                 for a in (True, False) for b in (True, False) for x in xs]
        return cases, "2 x 2 booleans x 101 values of x on [-1, 1] step 0.02"
    if name == "language":
        fixtures._use("language-capability")
        import common as LC
        eps = LC.dataset(400, seed0=100000, split="test")
        eps = [e for e in eps if 8 <= e["length"] <= 14]
        cases = [{"text": e["text"].decoded} for e in eps]
        return cases, "every test episode of declared length 8-14 in a 400-episode draw"
    if name == "visual":
        fixtures._use("visual-ladder")
        from common import FLAT, episode
        eps = [episode(s, "test", **FLAT) for s in range(200, 224)]
        cases = [{"observation": tuple(e["pixels"])} for e in eps]
        return cases, "24 held-out test screenshots, 961 interior positions each"
    if name == "computer":
        base = dict(f["cases"][0])
        cases = []
        for n, body in enumerate([b"note 0\n", b"note 3\n", b"note 9\n", b"note 12\n",
                                  b"a\n", b"", b"x" * 40 + b"\n", b"note -1\n",
                                  b"note 4096\n", b"\n"]):
            c = dict(base)
            buf = [0] * len(base["terminal"][1])
            buf[:len(body)] = list(body)
            c["terminal"] = (len(body), tuple(buf))
            cases.append(c)
        return cases, "10 synthetic terminal documents including empty and out-of-range"
    raise KeyError(name)


# ---------------------------------------------------------------------------
def run_artifact(name, base_mod, interp_limit):
    t0 = time.time()
    f = fixtures.FIXTURES[name]()
    program, registry = f["program"], f["registry"]
    old = base_mod.compile_program(program, registry)
    new = compile_program(program, registry)
    mo, mn = old.module("old"), new.module("new")
    cases, what = gate_cases(name, f)

    rows_old, rows_new, mismatches = [], [], []
    for i, c in enumerate(cases):
        for validate in (True, False):
            a = outcome(lambda: mo.run(dict(c), validate=validate))
            b = outcome(lambda: mn.run(dict(c), validate=validate))
            rows_old.append(a)
            rows_new.append(b)
            if a != b:
                mismatches.append({"case": i, "validate": validate, "old": a, "new": b})

    # gate B -- the typed interpreter
    types = dict(program.inputs)
    interp_checked, interp_bad = 0, []
    for i, c in enumerate(cases[:interp_limit]):
        want = outcome(lambda: _interp(program, registry, types, c))
        got = outcome(lambda: mn.run(dict(c), validate=True))
        interp_checked += 1
        if want != got:
            interp_bad.append({"case": i, "interpreter": want, "new": got})

    return {
        "artifact": name,
        "gate_domain": what,
        "cases": len(cases),
        "comparisons": len(rows_old),
        "gate_A_mismatches": len(mismatches),
        "gate_A_examples": mismatches[:5],
        "gate_A_digest_old": digest(rows_old),
        "gate_A_digest_new": digest(rows_new),
        "gate_A_passed": not mismatches and digest(rows_old) == digest(rows_new),
        "gate_B_checked": interp_checked,
        "gate_B_mismatches": len(interp_bad),
        "gate_B_examples": interp_bad[:3],
        "gate_B_passed": not interp_bad,
        "guards_old": {k: v for k, v in old.stats.items() if k.startswith("guard_")},
        "guards_new": {k: v for k, v in new.stats.items() if k.startswith("guard_")},
        "source_bytes_old": old.stats["source_bytes"],
        "source_bytes_new": new.stats["source_bytes"],
        "source_lines_old": old.stats["source_lines"],
        "source_lines_new": new.stats["source_lines"],
        "seconds": round(time.time() - t0, 1),
    }


def _interp(program, registry, types, c):
    ins = {k: Value.of(types[k], c[k]) for k in types}
    out, st = program.run(ins, registry=registry)
    return ({k: v.decoded for k, v in out.items()},
            {k: v.decoded for k, v in st.items()})


def main(argv):
    base_mod, base_sha = baseline_compiler()
    # the typed interpreter costs ~17 s per `visual` case (FINDINGS sec 48), so
    # gate B is stratified there and exhaustive everywhere else
    limits = {"mixed": 10 ** 9, "language": 10 ** 9, "computer": 10 ** 9, "visual": 6}
    only = argv[1:] or ["mixed", "language", "computer", "visual"]
    report = {"baseline_rev": BASE_REV, "baseline_source_sha256": base_sha, "artifacts": {}}
    for name in only:
        print("== gate %s" % name, flush=True)
        r = run_artifact(name, base_mod, limits[name])
        report["artifacts"][name] = r
        print("   A %s (%d comparisons, %d mismatches)  B %s (%d checked, %d mismatches)  %.1fs"
              % ("PASS" if r["gate_A_passed"] else "FAIL", r["comparisons"],
                 r["gate_A_mismatches"], "PASS" if r["gate_B_passed"] else "FAIL",
                 r["gate_B_checked"], r["gate_B_mismatches"], r["seconds"]), flush=True)
    report["all_passed"] = all(v["gate_A_passed"] and v["gate_B_passed"]
                               for v in report["artifacts"].values())
    path = OUT / ("gate.json" if len(only) == 4 else "gate_%s.json" % "_".join(only))
    path.write_text(json.dumps(report, indent=1, default=str))
    print("all_passed: %s  -> %s" % (report["all_passed"], path))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
