"""Two standing checks that this track's changes are inert where they claim to be.

1. **The core change is inert with `inline_bounded=False`.**  `tcn/compile.py` at
   §59's `ee15cb4` is read out of git -- not copied -- and asked to compile every
   arm this track builds; the source it emits must be **byte-identical** to what
   the patched compiler emits with the flag off.  A refinement bound is the only
   thing the patch can reach, so arm B is included deliberately: with the flag
   off it must still fall through to `_canon_fn` exactly as before.

2. **`harden_all` is `Program.harden(chosen)`.**  `arms.harden_all` exists only
   because a scaffold that gains a node cannot be frozen from a stored `chosen`
   dict.  On the three unmodified scaffolds the two must agree by program digest.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import types as _types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as ARMS                                     # noqa: E402
from tcn.compile import compile_program                 # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
BASE_REV = "ee15cb4"                                    # section 59's core change


def baseline_compiler(rev=BASE_REV):
    src = subprocess.run(["git", "-C", str(ROOT), "show", "%s:tcn/compile.py" % rev],
                         check=True, capture_output=True, text=True).stdout
    src = (src.replace("from .graph import", "from tcn.graph import")
              .replace("from .operators import", "from tcn.operators import")
              .replace("from .types import", "from tcn.types import"))
    mod = _types.ModuleType("tcn_compile_at_%s" % rev)
    mod.__dict__["__file__"] = "<%s:tcn/compile.py>" % rev
    exec(compile(src, mod.__dict__["__file__"], "exec"), mod.__dict__)
    return mod, hashlib.sha256(src.encode()).hexdigest()


def main():
    base, sha = baseline_compiler()
    rep = {"baseline_rev": BASE_REV, "baseline_source_sha256": sha, "arms": {}}
    for arm in ("A0", "A1", "B"):
        f = ARMS.build(arm, seeds=(200, 201, 202))
        old = base.compile_program(f["program"], f["registry"])
        new = compile_program(f["program"], f["registry"])
        rep["arms"][arm] = {
            "bytes": len(new.source.encode()),
            "sha_at_%s" % BASE_REV: hashlib.sha256(old.source.encode()).hexdigest(),
            "sha_patched_flag_off": hashlib.sha256(new.source.encode()).hexdigest(),
            "identical": old.source == new.source}
        print("%-3s %d bytes  identical to the emitter at %s with the flag off: %s"
              % (arm, len(new.source.encode()), BASE_REV, rep["arms"][arm]["identical"]))

    # ---- harden_all == harden(chosen) on the unmodified scaffolds ------------
    ARMS._track()
    from common import FLAT, Registry, episode, load
    import rung3_widgets as R
    reg = Registry()
    probe = episode(0, "train", **FLAT)
    W, H = probe["width"], probe["height"]
    offs = R.offset_pool(W)
    found = load("rung3")
    p0 = R.same_scaffold(reg, W, H)
    same = {"S0": (p0, found["s0"]["chosen"], found["s0"]["chosen"])}
    m0 = reg.register_module(p0.harden(found["s0"]["chosen"]))
    p1 = R.corner_scaffold(reg, W, H, m0, offs)
    same["S1"] = (p1, found["s1"]["chosen"],
                  {k: v for k, v in found["s1"]["chosen"].items()
                   if k in ("back_a", "back_b", "corner")})
    p2 = R.rect_scaffold(reg, W, H, m0, offs)
    same["S2"] = (p2, found["s2"]["chosen"],
                  {k: v for k, v in found["s2"]["chosen"].items() if k in ("step_w", "step_h")})
    rep["harden_all"] = {}
    for tag, (prog, full, choices) in same.items():
        a, b = prog.harden(full), ARMS.harden_all(prog, choices)
        rep["harden_all"][tag] = {"harden_digest": a.digest, "harden_all_digest": b.digest,
                                  "identical": a.digest == b.digest}
        print("%s harden_all == harden(chosen): %s" % (tag, a.digest == b.digest))
    rep["all_identical"] = (all(v["identical"] for v in rep["arms"].values())
                            and all(v["identical"] for v in rep["harden_all"].values()))
    (OUT / "inert.json").write_text(json.dumps(rep, indent=1, sort_keys=True))
    print("all_identical:", rep["all_identical"])


if __name__ == "__main__":
    main()
