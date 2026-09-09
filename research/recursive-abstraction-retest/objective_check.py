"""Could any objective the repo already has prefer the abstracted program?

Track 5's F3 said description size and cost never enter the synthesis objective.
That is still true on main: `SoftProgram.complexity()` is referenced once, in
`tcn/training.py`, behind `mdl_weight` which defaults to 0; `tcn/synthesis.py::fit`
-- the path that learns and crystallizes modules and synthesizes composites --
optimizes `probe_loss + 0.001*(step/steps)*entropy` and nothing else.

But suppose it were switched on. `complexity()` is a softmax-weighted sum of
operator *cost*, and F1 put a module call's cost at exact parity with its inlined
body (SS1.1). So the term that exists could not prefer abstraction even if it were
enabled. The quantity that does favour it -- description bits -- has no
differentiable surrogate anywhere. This measures both halves of that claim.
"""
import json
from pathlib import Path
import torch
from tcn.operators import Registry
from tcn.learning import SoftProgram
import common as C
import solvability as S

out = {}
r = Registry()
name = r.register_module(C.minimal_module(r, "maj"))
prog = C.wide_scaffold(r, name)

def held(sel):
    m = SoftProgram(prog, r)
    for k, i in sel.items():
        m.freeze(k, i)
    return m

flat = S.flat_selections(prog)
mod = S.module_selections_wide(prog)
mf, mm = held(flat), held(mod)
hf = prog.harden(flat).validate(r)
hm = prog.harden(mod).validate(r)
out["complexity_term"] = {
    "flat_route": round(float(mf.complexity().detach()), 3),
    "module_route": round(float(mm.complexity().detach()), 3),
    "note": "SoftProgram.complexity() over the whole hardened scaffold; equal or "
            "higher for the module route means the existing MDL term cannot prefer it",
}
out["pruned_truth"] = {
    "flat_execution_cost": C.prune(hf).execution_cost(r),
    "module_execution_cost": C.prune(hm).execution_cost(r),
    "flat_description_bits": C.prune(hf).description_bits(r),
    "module_description_bits": C.prune(hm).description_bits(r),
}
out["mdl_default"] = "tcn/training.py mdl_weight defaults to 0.0; tcn/synthesis.py::fit has no MDL term at all"
Path("objective_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
