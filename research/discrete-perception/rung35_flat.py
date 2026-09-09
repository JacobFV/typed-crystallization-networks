"""The undecomposed arm of rung 3.5, measured on its own.

The staged arm freezes the rung-3 foreground module and searches 48 programs.
This searches the same two-position predicate with nothing frozen: both pixels'
three channel comparisons, both foreground combinators, the neighbour offset and
the edge combinator, all free.  It is the control for "what does the
decomposition buy".
"""
from __future__ import annotations
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import dump, enumerate_reference, gradient_arm, random_reference, report, summarise
from rung35_window import edge_signals, flat_scaffold, window_examples
from tcn.operators import Registry
from tcn.search import space_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--held", type=int, default=6)
    ap.add_argument("--per-image", type=int, default=48)
    ap.add_argument("--budget", type=int, default=40000)
    ap.add_argument("--gradient-seeds", type=int, default=3)
    ap.add_argument("--tag", default="rung35_flat")
    a = ap.parse_args()
    R, tol, r = a.resolution, 1e-6, Registry()
    train = window_examples(tuple(range(a.train)), R, "train", a.per_image, seed=3)
    held = window_examples(tuple(range(100, 100 + a.held)), R, "test", a.per_image, seed=4)
    prog = flat_scaffold(r, R); sig = edge_signals()
    out = {"arguments": vars(a), "space_size": space_size(prog), "train_examples": len(train)}
    report("flat space", out["space_size"])
    part = enumerate_reference(prog, train, sig, r, tolerance=tol, max_programs=a.budget)
    part["programs_per_second"] = part["evaluated"] / max(1e-9, part["seconds"])
    part["projected_exhaustive_seconds"] = out["space_size"] / max(1e-9, part["programs_per_second"])
    out["enumerate_partial"] = part
    report("flat evaluated/solved/exhausted/rate/projected s",
           f"{part['evaluated']} / {part['solved']} / {part['exhausted']} / "
           f"{part['programs_per_second']:.0f} / {part['projected_exhaustive_seconds']:.4g}")
    dump(a.tag, out)
    out["random_control"] = random_reference(prog, train, sig, r, draws=500, tolerance=tol)
    dump(a.tag, out)
    g = gradient_arm(lambda: flat_scaffold(r, R), tuple(range(a.gradient_seeds)), train, sig, r,
                     steps=400, lr=.15, init_noise=.5, label="grad flat", tolerance=tol, held=held)
    out["gradient"] = {"rows": g, "summary": summarise(g), "init_noise": .5}
    report("flat gradient successes", f"{out['gradient']['summary']['successes']}/{len(g)}")
    dump(a.tag, out)

if __name__ == "__main__":
    main()
