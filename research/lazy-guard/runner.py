"""Shared measurement, certification and deployment for the staged arms.

Every number an arm reports comes from here, so the specification and the
resynthesized algorithm are always measured by the same instrument.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cost
import ir
from tcn.compile import compile_program
from tcn.types import Value, canonical

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def canon(x):
    return json.dumps(canonical(x), sort_keys=True, separators=(",", ":"))


def digest_outputs(outs):
    h = hashlib.sha256()
    for o in outs:
        h.update(canon(o).encode())
        h.update(b"\n")
    return h.hexdigest()


def spec_profile(fx, domain):
    """Executed primitives of the frozen specification, exhaustively."""
    r = fx["registry"]
    meter = cost.Meter()
    ev = cost.NativeEval(r, meter)
    port = fx["port"]
    prof, outs = cost.profile(
        lambda c: next(iter(ev.run(fx["program"], {port: c[port]})[0].values())),
        domain(fx), meter)
    prof["output_digest"] = digest_outputs(outs)
    return prof, outs


def alg_profile(fx, prog, domain):
    """Executed primitives of an `IRProgram`, exhaustively, same meter."""
    r = fx["registry"]
    ev = ir.Evaluator(r)
    port = fx["port"]
    prof, outs = cost.profile(lambda c: ev.run(prog, {port: c[port]}), domain(fx), ev.meter)
    prof["output_digest"] = digest_outputs(outs)
    return prof, outs


def certify(fx, prog, domain, spec_outs=None):
    """Exhaustive equivalence certificate against the typed interpreter.

    The oracle here is `Program.run` itself -- the typed interpreter, not a
    re-statement of it -- so the certificate is against the artifact the
    repository ships.
    """
    p, r = fx["program"], fx["registry"]
    port, t = fx["port"], fx["input_type"]
    ev = ir.Evaluator(r)
    n = 0
    for c in domain(fx):
        got = ev.run(prog, {port: c[port]})
        want = next(iter(p.run({port: Value.of(t, c[port])}, registry=r)[0].values())).decoded
        if got != want:
            return {"exact": False, "counterexample": c, "checked": n,
                    "got": repr(got), "want": repr(want)}
        n += 1
    return {"exact": True, "checked": n, "counterexample": None,
            "oracle": "tcn.graph.Program.run (typed interpreter)"}


def bytecode_classes(fn, cases_by_class):
    """Executed bytecodes per control-flow class, and within-class constancy."""
    out = {}
    for cls, cases in cases_by_class.items():
        counts = [cost.count_opcodes(lambda c=c: fn(c)) for c in cases]
        out[str(cls)] = {"bytecodes": counts[0], "constant_within_class": len(set(counts)) == 1,
                         "samples": counts}
    return out


def expected_bytecodes(per_class, weights):
    return sum(weights[c] * per_class[str(c)]["bytecodes"] for c in weights)


def worst_bytecodes(per_class):
    return max(v["bytecodes"] for v in per_class.values())


def deploy(fx, prog, name, domain, spec_outs):
    """Criterion 5: standalone stdlib-only Python, bit-identical to arm A.

    Run under `/usr/bin/python3 -I` with an empty `sys.path[0]`, no `tcn`, no
    torch, no numpy -- the same deployment property `research/compiled-runtime`
    established in FINDINGS sec 48.
    """
    r = fx["registry"]
    src = ir.Lowerer(r).program(prog, entry="run")
    path = OUT / ("%s_algorithm.py" % name)
    path.write_text(src)

    port = fx["port"]
    cases = [c[port] for c in domain(fx)]
    inp = OUT / ("%s_inputs.json" % name)
    inp.write_text(json.dumps(cases))

    driver = OUT / ("%s_deploy.py" % name)
    driver.write_text(
        "import hashlib, json, sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))\n"
        "import %s_algorithm as A\n"
        "assert 'tcn' not in sys.modules and 'torch' not in sys.modules and 'numpy' not in sys.modules\n"
        "cases = json.load(open(pathlib.Path(__file__).resolve().parent / '%s_inputs.json'))\n"
        "def canon(x):\n"
        "    if isinstance(x, frozenset):\n"
        "        return sorted((canon(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))\n"
        "    if isinstance(x, (tuple, list)):\n"
        "        return [canon(v) for v in x]\n"
        "    return x\n"
        "h = hashlib.sha256()\n"
        "for c in cases:\n"
        "    o = A.run(tuple(c))\n"
        "    h.update(json.dumps(canon(o), sort_keys=True, separators=(',', ':')).encode()); h.update(b'\\n')\n"
        "print(json.dumps({'digest': h.hexdigest(), 'cases': len(cases),\n"
        "                  'modules': sorted(m for m in sys.modules if m in ('tcn','torch','numpy'))}))\n"
        % (name, name))

    proc = subprocess.run(["/usr/bin/python3", "-I", str(driver)],
                          capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    ok = proc.returncode == 0
    info = json.loads(proc.stdout) if ok else {"stderr": proc.stderr[-2000:]}
    want = digest_outputs(spec_outs)
    return {"source_path": str(path), "source_bytes": len(src),
            "interpreter": "/usr/bin/python3 -I", "returncode": proc.returncode,
            "stdlib_only": ok, "generated_digest": info.get("digest"),
            "spec_digest": want,
            "bit_identical": ok and info.get("digest") == want,
            "cases": info.get("cases"), "detail": info}


def compiled_spec_module(fx):
    res = compile_program(fx["program"], fx["registry"])
    return res.module(), res
