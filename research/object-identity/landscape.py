"""Is the corrected surrogate informative about the choice it is supposed to settle?

A `k/n` from a training run confounds the landscape with the optimiser.  This
measures the landscape directly: with every other choice pinned to a program
that conforms exactly, it evaluates the relaxed loss at each candidate threshold
under several temperatures, and asks whether the minimum is at a threshold that
is actually exact.

That is the cheapest possible test of the coordinator's correction on this rung:
if the corrected surrogate puts its minimum on the right constant, relaxation
can settle the choice and only the optimiser is at fault; if it does not, the
temperature is still wrong.
"""
from __future__ import annotations
import json, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "discrete-perception"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import tcn.learning as L
from common import exact_error, report
from common2 import OUT
from rung4_segment import (collinear_scaffold, dump, same_examples, same_signals, THRESHOLDS)
from surrogate_fix import scaled_relaxed, SHIPPED
from unfreeze import unfreeze
from tcn.learning import SoftProgram, tensor as _tensor
from tcn.operators import COMPARE, Registry

R = 8
POOL = tuple(range(0, 512, 16))          # a coarse grid over the wide pool


def fixed_tau(tau):
    def relaxed(registry, op, xs, temperature=1.):
        if op.name in COMPARE and op.gradient != "none":
            a, b = xs
            if op.name == "eq":
                return torch.exp(-((a - b) ** 2).sum(-1, keepdim=True) / tau)
            d = b - a if op.name in {"lt", "le"} else a - b
            return torch.sigmoid(d / tau)
        return SHIPPED(registry, op, xs, temperature)
    return relaxed


def main():
    r = Registry()
    train = same_examples(range(8), R, "train", None, seed=3)
    sig = same_signals()
    # The `le` nodes have one candidate each, so `SoftProgram` would treat them
    # as frozen and evaluate them with `exact_tensor(...).detach()` -- which
    # bypasses `relaxed` entirely and makes every temperature policy give the
    # identical landscape.  Unfreezing them is what makes this a measurement of
    # the surrogate rather than of the exact semantics.
    program = unfreeze(collinear_scaffold(r, R, thresholds=POOL),
                       only=("rg_le", "gb_le", "rb_le"))
    names = [n.name for n in program.nodes]
    # pin every choice but `thr` to the combination the exhaustive search returned
    base = {n: 0 for n in names}
    base["m1"] = 1; base["same"] = 2; base["shifted"] = 0
    exact = [i for i, T in enumerate(POOL)
             if exact_error(program, train, sig, r, selections={**base, "thr": i}) <= 1e-6]
    report("thresholds in this grid that are exact on training",
           f"{[POOL[i] for i in exact]}")

    model = SoftProgram(program, r)
    inputs = {k: torch.stack([_tensor(ex["inputs"][k]) for ex in train])
              for k, _ in program.inputs}
    targets = {s.target: torch.stack([_tensor(ex["targets"][s.target]) for ex in train])
               for s in sig}

    def loss_at(i):
        model.trials = {}
        with torch.no_grad():
            for n, p in zip(program.nodes, model.choices):
                p.zero_()
                sel = base.get(n.name, 0) if n.name != "thr" else i
                p[sel] = 20.                      # a hard one-hot, so this is the landscape
            _, _, tr = model(inputs, return_trace=True)
            return float(model.probe_loss(tr, targets, sig))

    rows = {}
    policies = [("shipped tau=1", fixed_tau(1.)),
                ("tau=8", fixed_tau(8.)), ("tau=32", fixed_tau(32.)),
                ("tau=128", fixed_tau(128.)), ("tau=1024", fixed_tau(1024.)),
                ("operand (mean|operand|)", scaled_relaxed("operand")),
                ("carrier (2^bits)", scaled_relaxed("carrier"))]
    for label, fn in policies:
        L.relaxed = fn
        try:
            losses = [loss_at(i) for i in range(len(POOL))]
        finally:
            L.relaxed = SHIPPED
        best = min(range(len(POOL)), key=lambda i: losses[i])
        rows[label] = {"losses": losses, "argmin_index": best, "argmin_threshold": POOL[best],
                       "argmin_is_exact": best in exact,
                       "loss_spread": max(losses) - min(losses)}
        report(f"[{label}] argmin T / exact? / loss spread",
               f"{POOL[best]} / {best in exact} / {max(losses)-min(losses):.3e}")
    dump("landscape", {"pool": list(POOL), "exact_indices": exact,
                       "exact_thresholds": [POOL[i] for i in exact],
                       "pinned": base, "policies": rows})


if __name__ == "__main__":
    main()
