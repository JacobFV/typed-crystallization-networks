"""Re-examine every negative gradient result against a LIVE comparison surrogate.

The coordinator's correction: a gradient of exactly 0.0 is not evidence about
learnability, it is evidence the relaxation was invalid there.  `eq`'s surrogate
`exp(-(a-b)^2/tau)` is exactly 0.0 in float32 for `|a-b| >= 11` at tau=1, and
scaling by the carrier width fixes the address benchmark.

This track hit the same fault in the other comparison: `le` relaxes to
`sigmoid((b-a)/tau)` at tau=1, which is exactly 0.0 for a gap of 89 or more, and
the operands here are products of two bytes.  So every gradient arm recorded in
sections 3 and 4 has to be re-run with a surrogate that is alive at the
operating distance, and the operating distance has to be reported beside it.

Three things are measured:

1. the distribution of the operating gap `|d - T|` actually seen at the `le`
   nodes, and the shipped surrogate's value and derivative there;
2. the gradient arm with the surrogate scaled, under two policies -- the
   coordinator's `tau = 2^bits` and this track's `tau = mean|operand|`;
3. whether the choices that were dead for a *declared* reason (the offset,
   behind `pack`'s `gradient="none"`) come back.  They should not: that boundary
   is a declaration, not a numerical accident, and the distinction is the point.

Nothing under `tcn/` is modified.  `relaxed` is replaced at runtime, in this
process only, and the replacement is byte-identical to the shipped one except
for the temperature used by COMPARE.
"""
from __future__ import annotations
import argparse, json, math, pathlib, statistics, sys, time
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "discrete-perception"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import tcn.learning as L
from common import exact_error, accuracy, gradient_arm, report, summarise
from common2 import OUT
from rung4_segment import (collinear_scaffold, choice_gradients, dump, same_examples,
                           same_signals, THRESHOLDS)
from unfreeze import unfreeze, address_sharpness
from tcn.operators import COMPARE, Registry
from tcn.search import space_size
import common2 as C
import common as DP

SHIPPED = L.relaxed
R = 8


def scaled_relaxed(policy):
    """`tcn.learning.relaxed` with one line changed: the COMPARE temperature."""
    def relaxed(registry, op, xs, temperature=1.):
        if op.name in COMPARE and op.gradient != "none":
            a, b = xs
            if policy == "carrier":                     # the coordinator's rule
                tau = float(2 ** op.inputs[0].bits)
            elif policy == "operand":                   # scale by the operand magnitude
                tau = float(max(1., torch.maximum(a.detach().abs(),
                                                  b.detach().abs()).mean()))
            elif policy.startswith("fixed"):            # scale by the DECISION MARGIN
                tau = float(policy[5:])
            else:
                tau = temperature
            if op.name == "eq":
                return torch.exp(-((a - b) ** 2).sum(-1, keepdim=True) / tau)
            d = b - a if op.name in {"lt", "le"} else a - b
            return torch.sigmoid(d / tau)
        return SHIPPED(registry, op, xs, temperature)
    return relaxed


def operating_gaps(r, thresholds, train, n=64):
    """The gap |d - T| actually seen at each `le` node, and the surrogate there."""
    program = collinear_scaffold(r, R, thresholds=thresholds)
    sel = {node.name: 0 for node in program.nodes}
    gaps = {"rg_le": [], "gb_le": [], "rb_le": []}
    T = thresholds[len(thresholds) // 2]
    sel["thr"] = thresholds.index(T)
    for ex in train[:n]:
        _, _, tr = program.execute(ex["inputs"], registry=r, selections=sel)
        for k in gaps:
            d = tr[k[:2] + "_a"].decoded
            gaps[k].append(abs(d - T))
    out = {"threshold_used": T}
    for k, v in gaps.items():
        med = statistics.median(v)
        x = torch.tensor([float(med)], requires_grad=True)
        s = torch.sigmoid(-x / 1.0)
        s.backward()
        out[k] = {"median_gap": med, "mean_gap": sum(v) / len(v),
                  "max_gap": max(v), "min_gap": min(v),
                  "fraction_at_or_past_underflow": sum(1 for g in v if g >= 89) / len(v),
                  "shipped_surrogate_at_median_gap": float(s.detach()),
                  "shipped_derivative_at_median_gap": float(x.grad)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--width", type=int, default=0, help="0 = coarse pool, else 0..width-1")
    ap.add_argument("--policies", default="operand,carrier")
    ap.add_argument("--pin-offset", action="store_true",
                    help="pin `shifted` to +1 pixel: removes the one choice that is "
                         "behind a DECLARED gradient boundary, leaving only choices "
                         "the relaxed backend can in principle see")
    ap.add_argument("--tag", default="surrogate_fix")
    args = ap.parse_args()
    r = Registry()
    thresholds = THRESHOLDS if not args.width else tuple(range(args.width))
    train = same_examples(range(8), R, "train", None, seed=3)
    held = same_examples(range(100, 108), R, "test", None, seed=4)
    sig = same_signals()
    result = {"arguments": vars(args), "threshold_pool": len(thresholds),
              "space_size": space_size(collinear_scaffold(r, R, thresholds=thresholds))}

    result["operating_gaps"] = operating_gaps(r, thresholds, train)
    for k, v in result["operating_gaps"].items():
        if isinstance(v, dict):
            report(f"[gap] {k}: median / fraction past underflow / shipped surrogate there",
                   f"{v['median_gap']} / {v['fraction_at_or_past_underflow']:.3f} / "
                   f"{v['shipped_surrogate_at_median_gap']:.3g}")
    dump(args.tag, result)

    result["address_sharpness"] = address_sharpness()
    report("relaxed `index`: weight on the true address",
           json.dumps(result["address_sharpness"]))
    # Unfreeze ONLY the comparison nodes: that restores the gradient to `thr`
    # without replacing the exact `index` reads with a five-byte blur.
    COMPARE_NODES = ("rg_le", "gb_le", "rb_le")
    def make():
        p = unfreeze(collinear_scaffold(r, R, thresholds=thresholds), only=COMPARE_NODES)
        if args.pin_offset:
            import dataclasses
            p = dataclasses.replace(p, nodes=tuple(
                dataclasses.replace(n, candidates=n.candidates[:1], selected=0)
                if n.name == "shifted" else n for n in p.nodes))
        return p
    build = make
    result["unfrozen_nodes"] = list(COMPARE_NODES)
    result["offset_pinned"] = bool(args.pin_offset)
    result["space_size_searched"] = space_size(build())
    for policy in args.policies.split(","):
        L.relaxed = scaled_relaxed(policy)
        try:
            g = choice_gradients(build, train[:64], sig, r)
            report(f"[{policy}] choice logit |grad|_1", json.dumps(g["choice_grad_l1"]))
            rows = gradient_arm(build, tuple(range(args.seeds)), train, sig, r,
                                steps=400, lr=.15, init_noise=.5, label=policy,
                                tolerance=1e-6, held=held)
        finally:
            L.relaxed = SHIPPED
        result[policy] = {"choice_gradients": g, "rows": rows, "summary": summarise(rows),
                          "held_exact": sum(1 for x in rows if x.get("held_err") == 0.)}
        report(f"[{policy}] gradient successes / held-out exact",
               f"{result[policy]['summary']['successes']}/{len(rows)} / "
               f"{result[policy]['held_exact']}")
        dump(args.tag, result)
    dump(args.tag, result)


if __name__ == "__main__":
    main()
