"""Does normalising the probe's field scale (ARCHITECTURE section 8) rescue rung 1?

Measured answer: no.  Adam's per-parameter normalisation absorbs the rescaling,
and on the 288-program scaffold the eight per-seed exact errors are bit-identical
at weights 1, 20, 100 and 400.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tcn.operators import Registry
from tcn.graph import Signal
from rung1_signal import frequency_program, sample_examples, F, LADDER
from common import local_fit, dump


def main():
    tr = sample_examples(range(0, 60))[:10]
    out = {}
    for level in ("L2_trig", "L4_all"):
        for w in (1., 20., 100., 400.):
            sig = (Signal("fr", "frequency", ("latent",), F, "mse", weight=w),)
            rows = []
            for seed in range(8):
                r = Registry(); p = frequency_program(r, free=LADDER[level])
                torch.manual_seed(seed)
                try:
                    m, rep = local_fit(p, tr, sig, steps=600, freeze=False, registry=r, polish=0,
                                       init_noise=.5, seed=seed)
                    rows.append({"seed": seed, "err": rep["exact_max_error"],
                                 "ok": rep["exact_conformance"], "sel": m.selections()})
                except Exception as exc:
                    rows.append({"seed": seed, "err": None, "ok": False, "error": repr(exc)[:80]})
            ok = sum(r["ok"] for r in rows)
            out[f"{level}_weight{w}"] = {"successes": ok, "n": len(rows), "rows": rows}
            print(f"{level} weight={w:6.1f}: {ok}/{len(rows)}",
                  [None if r["err"] is None else round(r["err"], 4) for r in rows], flush=True)
    dump("rung1_signal_weight", out)


if __name__ == "__main__":
    main()
