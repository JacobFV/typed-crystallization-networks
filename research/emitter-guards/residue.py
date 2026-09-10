"""Why a guard could not be proved redundant -- the residue, by operator and type.

Re-runs the emitter with the interval decision instrumented, so every guard that
*stayed* is attributed to the node that emitted it, with the declared operand
types and the interval the lattice was able to prove.  This is the raw data
behind RESULTS.md's "guards the declared type does not bound" table.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                        # noqa: E402
from tcn import compile as C                           # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def instrument():
    """Wrap the two guard decisions so each one records why it went as it did."""
    log = []
    real_scalar = C._Compiler._emit_scalar
    real_lower = C._Compiler._lower

    def emit_scalar(self, dst, expr, op, s=()):
        before = self.stats["guard_range_eliminated"]
        out = real_scalar(self, dst, expr, op, s)
        if self._fast_kind(op) == "int":
            kept = self.stats["guard_range_eliminated"] == before
            log.append({"guard": "G1", "kept": kept, "operator": op.name,
                        "operand_types": [_t(x) for x in op.inputs],
                        "operand_intervals": [self.src_iv(x) for x in s],
                        "output_type": _t(op.output),
                        "proved": self._expr_iv(op, s) if s else None})
        return out

    def lower(self, dst, op, s, scope):
        b_ix = self.stats["guard_index_eliminated"]
        b_dz = self.stats["guard_zerodiv_eliminated"]
        b_sh = self.stats["guard_shift_eliminated"]
        e_ix = self.stats["guard_index_emitted"]
        e_dz = self.stats["guard_zerodiv_emitted"]
        e_sh = self.stats["guard_shift_emitted"]
        out = real_lower(self, dst, op, s, scope)
        if op.name.startswith("module:") or op.name in {"map", "filter"}:
            # these lower by emitting a whole module body, whose own guards are
            # logged by their own `_lower` calls; a counter delta here would
            # double count them
            return out
        for cls, key, be, ee in (("G2", "index", b_ix, e_ix), ("G3", "zerodiv", b_dz, e_dz),
                                 ("G4", "shift", b_sh, e_sh)):
            gone = self.stats["guard_%s_eliminated" % key] > be
            stayed = self.stats["guard_%s_emitted" % key] > ee
            if gone or stayed:
                log.append({"guard": cls, "kept": stayed, "operator": op.name,
                            "operand_types": [_t(x) for x in op.inputs],
                            "operand_intervals": [self.src_iv(x) for x in s],
                            "output_type": _t(op.output), "proved": None})
        return out

    C._Compiler._emit_scalar = emit_scalar
    C._Compiler._lower = lower
    return log, (real_scalar, real_lower)


def _t(t):
    if t.kind == "int":
        return "int%d%s%s%s" % (t.bits, "s" if t.encoding.signed else "u",
                                "" if t.encoding.kind == "integer" else "/" + t.encoding.kind,
                                "" if t.bounds is None else " bounds%s" % (t.bounds,))
    if t.kind == "tuple":
        return "tuple[%d]" % len(t.items)
    return t.kind


def main():
    log, (rs, rl) = instrument()
    report = {}
    try:
        for name in ("mixed", "language", "visual", "computer"):
            f = fixtures.FIXTURES[name]()
            log.clear()
            res = C.compile_program(f["program"], f["registry"])
            kept = collections.Counter()
            gone = collections.Counter()
            for e in log:
                key = "%s %s(%s) -> %s" % (e["guard"], e["operator"],
                                           ", ".join(e["operand_types"]), e["output_type"])
                (kept if e["kept"] else gone)[key] += 1
            report[name] = {
                "stats": {k: v for k, v in res.stats.items() if k.startswith("guard_")},
                "kept_by_shape": dict(sorted(kept.items(), key=lambda kv: -kv[1])),
                "eliminated_by_shape": dict(sorted(gone.items(), key=lambda kv: -kv[1])),
                "kept_examples": [e for e in log if e["kept"]][:6],
            }
            print("==", name)
            for k, v in report[name]["kept_by_shape"].items():
                print("   KEPT %5d  %s" % (v, k))
            for k, v in report[name]["eliminated_by_shape"].items():
                print("   gone %5d  %s" % (v, k))
    finally:
        C._Compiler._emit_scalar = rs
        C._Compiler._lower = rl
    (OUT / "residue.json").write_text(json.dumps(report, indent=1, default=str))
    print("-> %s" % (OUT / "residue.json"))


if __name__ == "__main__":
    main()
