"""How much of rung 3 is the scaffold?  Free the channel layout and re-measure.

`rung3_mask.module_scaffold` supplies one structural prior beyond the positional
pattern itself: that a pixel's three channels sit at `pos`, `pos+1`, `pos+2`.
That is the raster layout, and it is supplied, not searched.

Here the two channel offsets are searched as well, over {1,2,3} each, so the
module has to discover the interleaving of an RGB image alongside the background
colour and the Boolean combination: 9 x 32,000 = 288,000 programs.  The address
of every read is still *computed from the module's own position*, never chosen
from a free menu over the whole image, which is the distinction the perception
ladder measured relaxation to fail at.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BACKGROUND, BYTE, IDX, Builder, accuracy, dump, enumerate_reference,
                    exact_error, gradient_arm, record_type, report, summarise)
from rung3_mask import POOL, pixel_examples, signals
from tcn.operators import Registry
from tcn.search import space_size
from tcn.types import Value

OFFSETS = (1, 2, 3)


def scaffold(registry, resolution, pool=POOL, offsets=OFFSETS):
    REC = record_type(resolution)
    consts = tuple((f"byte_{v}", Value.of(BYTE, v)) for v in pool)
    consts += tuple((f"d{k}", Value.of(IDX, k)) for k in offsets)
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.choice("g_at", [("add", ("pos", f"d{k}"), None) for k in offsets])
    b.choice("b_at", [("add", ("pos", f"d{k}"), None) for k in offsets])
    b.add("red", "index", ["obs", "pos"])
    b.add("green", "index", ["obs", "g_at"])
    b.add("blue", "index", ["obs", "b_at"])
    for name, source in (("cmp_r", "red"), ("cmp_g", "green"), ("cmp_b", "blue")):
        b.choice(name, [("eq", (source, f"byte_{v}"), None) for v in pool])
    b.choice("rg", [(f"truth_{t}", ("cmp_r", "cmp_g"), None) for t in range(16)])
    b.choice("foreground", [(f"truth_{t}", ("rg", "cmp_b"), None) for t in range(16)])
    b.add("record", "tuple", ["pos", "foreground"])
    return b.program((("y", "record"),))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--held", type=int, default=6)
    ap.add_argument("--per-image", type=int, default=48)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--gradient-seeds", type=int, default=6)
    ap.add_argument("--tag", default="offsets_free")
    args = ap.parse_args()

    R = args.resolution
    tol = 1e-6
    r = Registry()
    train = pixel_examples(tuple(range(args.train)), R, "train", args.per_image, seed=1,
                           objects=args.objects)
    held = pixel_examples(tuple(range(100, 100 + args.held)), R, "test", args.per_image,
                          seed=2, objects=args.objects)
    sig = signals()
    prog = scaffold(r, R)
    result = {"arguments": vars(args), "space_size": space_size(prog),
              "train_examples": len(train)}
    report("free-offset space", result["space_size"])

    # `enumerate_fit` for the first solution, prefix-reusing enumeration for the
    # certificate: the two return the same conforming set (see incremental.py).
    import time
    from incremental import incremental_conforming
    from tcn.search import enumerate_fit, evaluate
    t0 = time.perf_counter()
    first = enumerate_fit(prog, train, sig, registry=r, tolerance=tol,
                          stop_at_first=True).to_dict()
    first["wall"] = time.perf_counter() - t0
    result["enumerate_first"] = first
    report("enumerate_fit stop_at_first solved/evaluated/seconds",
           f"{first['solved']} / {first['evaluated']} / {first['seconds']:.2f}")
    enum = incremental_conforming(prog, train, sig, r, tolerance=tol)
    enum["space_size"] = space_size(prog)
    enum["solved"] = enum["count"] > 0
    enum["selections"] = enum["conforming"][0] if enum["conforming"] else None
    validated = [s for s in enum["conforming"]
                 if (lambda e: e is not None and e <= tol)(evaluate(prog, s, held, sig, r, tol))]
    enum["conforming_held_out"] = len(validated)
    enum["evaluated"] = enum["space_size"]
    enum.pop("conforming", None)
    enum["held_out_max_error"] = exact_error(prog, held, sig, r, selections=enum["selections"])
    enum["held_out_accuracy"] = accuracy(prog, held, sig, r, selections=enum["selections"])
    result["enumerate_fit"] = enum
    report("exhaustive solved/exhausted/unique/conforming/held-out-conforming/seconds",
           f"{enum['solved']} / {enum['exhausted']} / {enum['unique']} / {enum['count']} / "
           f"{enum['conforming_held_out']} / {enum['seconds']:.1f}")

    grad = gradient_arm(lambda: scaffold(r, R), tuple(range(args.gradient_seeds)),
                        train, sig, r, steps=400, lr=.15, init_noise=.5,
                        label="grad offsets", tolerance=tol, held=held)
    result["gradient"] = {"rows": grad, "summary": summarise(grad), "init_noise": .5}
    report("gradient successes",
           f"{result['gradient']['summary']['successes']}/{len(grad)}")
    dump(args.tag, result)


if __name__ == "__main__":
    main()
