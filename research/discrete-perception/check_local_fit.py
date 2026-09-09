"""`common.local_fit` must be `tcn.synthesis.fit` when `init_noise` is zero.

Every per-seed gradient number in this track comes from `local_fit`, because
`SoftProgram` zero-initialises its choice logits and `torch.manual_seed` does not
vary synthesis at all.  This checks that the copy differs from the shipped
function only by that addition: at `init_noise=0` the two must agree bit for bit
on the shipped `examples/mixed.py` fixture, at two budgets, with and without the
constant polish.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

from common import dump, local_fit, report
from examples.mixed import problem
from tcn.synthesis import fit


def main():
    out = {"cases": []}
    for steps in (60, 200):
        for polish in (0, 200):
            p, signals, examples = problem()
            torch.manual_seed(0)
            _, a = fit(p, examples, signals, steps=steps, freeze=False, polish=polish)
            p, signals, examples = problem()
            torch.manual_seed(0)
            _, b = local_fit(p, examples, signals, steps=steps, freeze=False, polish=polish,
                             init_noise=0.)
            same = (a["exact_max_error"] == b["exact_max_error"]
                    and a["relaxed_loss"] == b["relaxed_loss"])
            out["cases"].append({"steps": steps, "polish": polish, "identical": bool(same),
                                 "fit": {k: a[k] for k in ("relaxed_loss", "exact_max_error")},
                                 "local_fit": {k: b[k] for k in ("relaxed_loss", "exact_max_error")}})
            report(f"steps={steps} polish={polish} identical", same)
    # and that noise actually changes the outcome, so the seeds mean something
    p, signals, examples = problem()
    losses = []
    for seed in range(4):
        p, signals, examples = problem()
        _, r = local_fit(p, examples, signals, steps=60, freeze=False, polish=0,
                         init_noise=.5, seed=seed)
        losses.append(r["relaxed_loss"])
    out["init_noise_varies_outcome"] = len(set(losses)) > 1
    out["noise_losses"] = losses
    report("init_noise=0.5 produces distinct outcomes across seeds",
           out["init_noise_varies_outcome"])
    out["zero_noise_is_deterministic"] = all(c["identical"] for c in out["cases"])
    dump("check_local_fit", out)


if __name__ == "__main__":
    main()
