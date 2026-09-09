"""Does the relaxed mixture stay inside the set of achievable values?

For every free node, compare the range of the *mixed* value the relaxation
actually propagates against the union of the ranges of its individual
candidates.  A mixture that leaves the achievable set is a value no program can
produce, so the gradient it carries is about a fictitious program.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tcn.operators import Registry
from tcn.learning import SoftProgram, tensor, relaxed
from common import dump


def excursion(program, examples, registry, signals=None):
    m = SoftProgram(program, registry)
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in examples]) for k, _ in program.inputs}
    with torch.no_grad():
        _, _, trace = m(inputs, return_trace=True)
    rows = {}
    values = dict(trace)
    for node in program.nodes:
        if len(node.candidates) < 2: continue
        cand = []
        for c in node.candidates:
            try: cand.append(relaxed(registry, c.operator, [values[s] for s in c.sources], 1.))
            except Exception: pass
        if not cand: continue
        lo = min(float(x.min()) for x in cand); hi = max(float(x.max()) for x in cand)
        mix = values[node.name]
        mlo, mhi = float(mix.min()), float(mix.max())
        span = max(hi - lo, 1e-12)
        rows[node.name] = {"candidates": len(node.candidates), "achievable": [lo, hi],
                           "mixture": [mlo, mhi],
                           "outside": bool(mlo < lo - 1e-6 or mhi > hi + 1e-6),
                           "magnitude_ratio": max(abs(mlo), abs(mhi)) / max(max(abs(lo), abs(hi)), 1e-12)}
    return rows


if __name__ == "__main__":
    out = {}
    from rung1_signal import frequency_program, sample_examples, LADDER, FREQ_SIGNAL, WINDOW
    ex = sample_examples(range(0, 60))[:10]
    for name, free in LADDER.items():
        r = Registry(); out[f"rung1:{name}"] = excursion(frequency_program(r, free=free), ex, r)
    from rung3_geometry import centre_program, mask_program, examples as gex, BG, AND, OR
    g = gex(range(0, 24), 2)
    r = Registry(); out["rung3:centre_free_address_R2"] = excursion(centre_program(r, 2, free_address=True), g, r)
    r = Registry(); out["rung3:centre_colour_search_R2"] = excursion(
        centre_program(r, 2, free_address=False, pool=tuple(range(256))), g, r)
    r = Registry(); out["rung3:mask_pinned_address_R2"] = excursion(
        mask_program(r, 2, free_address=False, pool=(24, 30, 43, 15), and_menu=(AND, OR)), g, r)
    dump("excursion", out)
    for k, v in out.items():
        n = len(v); bad = sum(1 for x in v.values() if x["outside"])
        worst = max((x["magnitude_ratio"] for x in v.values()), default=0.)
        print(f"{k:36s} free nodes={n:3d} mixtures outside the achievable range={bad:3d} "
              f"worst |mixture|/|achievable| = {worst:.3g}")

