"""What the commitment costs: candidate-space width and success on an existing fixture.

Two measurements.

**1. Candidate width.**  `tcn.graph.legal_candidates` is the enumerator every
scaffold builds on.  Run it over a byte-bearing port set with the whole operator
inventory, before and after a declared `interpret` node exists in the port set.
This is the honest "before/after", because the widening is not automatic: a byte
is *still* not numeric, so the arithmetic family opens only where a program has
paid for a commitment node and the node's declared output type is a magnitude.

**2. An existing fixture.**  `research/discrete-perception`'s rung-3 foreground
module -- `fg(pixel) = combine(eq(r,c1), eq(g,c2), eq(b,c3))` over a byte pool --
is the tree's canonical byte task, exhausted at 32,000 programs.  Rebuild it
here, then rebuild it with the commitment available so that the channel
predicates may also be `lt`/`le`/`gt`/`ge` against a magnitude threshold, and
report space size, wall clock, conforming count, uniqueness and held-out error
for both.  A larger conforming set at the same held-out error is the real cost:
the search still succeeds, but its tie-break has more ways to be wrong.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BYTE, IDX, MAG8, Builder, all_conforming, dump, exact_error, local_fit,
                    report)
from tcn.graph import Signal, legal_candidates
from tcn.operators import BINARY, COMPARE, LOGIC, UNARY, Registry
from tcn.search import evaluate, space_size
from tcn.types import BOOL, Value, product

INVENTORY = sorted(BINARY | UNARY | COMPARE | LOGIC | {
    "sum", "mean", "reduce_min", "reduce_max", "count", "identity", "mux",
    "tuple", "project", "index", "encode", "decode", "quantize", "dequantize",
    "pack", "unpack", "interpret"} | {f"truth_{t}" for t in range(16)})

# The rung-3 byte pool, copied from `research/discrete-perception/rung3_mask.py`.
POOL = (0, 24, 30, 43, 255)


# --------------------------------------------------------------------------
# 1. candidate width
# --------------------------------------------------------------------------
def widths(registry):
    """Legal candidates per declared node output, on a byte-only port set and a committed one."""
    uncommitted = {"px": BYTE, "c": BYTE, "i": IDX}
    committed = uncommitted | {"m": MAG8, "t": MAG8}
    out = {}
    for label, ports in (("bytes only", uncommitted), ("bytes + committed magnitude", committed)):
        row = {}
        for out_label, out_type in (("bool", BOOL), ("byte", BYTE), ("magnitude int[8]", MAG8),
                                    ("index int[16]", IDX)):
            try:
                row[out_label] = len(legal_candidates(registry, INVENTORY, ports, out_type,
                                                      arities=(1, 2), limit=1 << 20))
            except ValueError:
                row[out_label] = None
        out[label] = row
    return out


# --------------------------------------------------------------------------
# 2. the rung-3 fixture, before and after
# --------------------------------------------------------------------------
def pixel_examples(seeds, resolution, split="train", per_image=48, seed=0, objects=6):
    """`fg(i)` from the `object_ids` probe -- the rung-3 target, one pixel at a time."""
    from tcn.generation import Action, Host
    from common import VEC3
    rows = []
    rng = random.Random(seed)
    PIX = product(BYTE, BYTE, BYTE)
    for s in seeds:
        h = Host.create("geometry", seed=s, split=split,
                        configuration={"resolution": resolution, "objects": objects, "horizon": 4})
        rec = h.step([Action("camera", arguments=(("delta", Value.of(VEC3, (-2.4, -2.4, -4.2))),))])
        pixels = rec.actor_view().observations["pixels"].decoded[3]
        ids = rec.probes["object_ids"].decoded
        n = resolution * resolution
        chosen = range(n) if per_image is None else sorted(rng.sample(range(n), min(per_image, n)))
        for i in chosen:
            rows.append({"inputs": {"px": Value.of(PIX, tuple(pixels[3 * i:3 * i + 3]))},
                         "targets": {"fg": Value.of(BOOL, ids[i] >= 0)}})
    rng.shuffle(rows)
    return rows


def mask_scaffold(registry, pool=POOL, committed=False):
    """`combine(cmp(r,.), cmp(g,.), cmp(b,.))`, with 16-way truth-table combinators.

    `committed=False` is the shipped space: the only legal channel predicate on
    `role="byte"` is `eq` against a constant byte, so each channel node has
    `len(pool)` candidates.  `committed=True` inserts one `interpret` node per
    channel and lets the predicate also be an ordering comparison against a
    magnitude threshold, which is exactly what the commitment buys and exactly
    what it costs.
    """
    consts = tuple((f"byte_{v}", Value.of(BYTE, v)) for v in pool)
    if committed:
        consts += tuple((f"mag_{v}", Value.of(MAG8, v)) for v in pool)
    b = Builder(registry, (("px", product(BYTE, BYTE, BYTE)),), consts)
    for i, ch in enumerate(("red", "green", "blue")):
        b.add(ch, "project", ["px"], params={"index": i})
        cands = [("eq", (ch, f"byte_{v}"), None, None) for v in pool]
        if committed:
            b.add(f"{ch}_mag", "interpret", [ch], out=MAG8)
            cands += [(op, (f"{ch}_mag", f"mag_{v}"), None, None)
                      for op in ("eq", "lt", "le", "gt", "ge") for v in pool]
        b.choice(f"cmp_{ch}", cands)
    b.choice("rg", [(f"truth_{t}", ("cmp_red", "cmp_green"), None, None) for t in range(16)])
    b.choice("fg", [(f"truth_{t}", ("rg", "cmp_blue"), None, None) for t in range(16)])
    return b.program((("y", "fg"),))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--per-image", type=int, default=48)
    ap.add_argument("--budget", type=int, default=32000)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--grad-seeds", type=int, default=4)
    ap.add_argument("--grad-steps", type=int, default=300)
    ap.add_argument("--tag", default="cost")
    args = ap.parse_args()

    r = Registry()
    tol = 1e-6
    result = {"arguments": vars(args)}

    result["candidate_widths"] = widths(r)
    print("legal candidates per declared node output")
    for label, row in result["candidate_widths"].items():
        print(f"  {label:<32s} " + "  ".join(f"{k}={v}" for k, v in row.items()))

    sig = (Signal("fg", "fg", ("core",), BOOL, "bce"),)
    train = pixel_examples(range(args.train), args.resolution, "train", args.per_image, seed=1)
    held = pixel_examples(range(100, 100 + args.held), args.resolution, "test", None, seed=2)
    result["train_rows"] = len(train)
    result["held_rows"] = len(held)
    result["foreground_fraction_train"] = sum(
        ex["targets"]["fg"].decoded for ex in train) / len(train)
    report("rung-3 train rows / held-out rows / foreground fraction",
           f"{len(train)} / {len(held)} / {result['foreground_fraction_train']:.3f}")

    for label, committed in (("shipped (byte only)", False), ("with commitment", True)):
        prog = mask_scaffold(r, committed=committed)
        n = space_size(prog)
        names = [x.name for x in prog.nodes]
        # Both arms get the SAME evaluation budget -- the one that exhausts the
        # shipped space -- so "does search degrade" is answered under an equal
        # cost rather than by letting the wider space spend more.
        t0 = time.perf_counter()
        conf = all_conforming(prog, train, sig, r, tolerance=tol, limit=args.budget)
        wall = time.perf_counter() - t0
        arm = {"space_size": n, "nodes": len(prog.nodes), "seconds": wall,
               "evaluated": conf["evaluated"], "exhausted": conf["evaluated"] >= n,
               "conforming": conf["count"], "unique": conf["unique"] and conf["evaluated"] >= n,
               "programs_per_second": conf["evaluated"] / max(1e-9, wall),
               "projected_exhaustive_seconds": n / max(1e-9, conf["evaluated"] / max(1e-9, wall))}
        arm["conforming_density_prefix"] = conf["count"] / max(1, conf["evaluated"])
        # `itertools.product` varies the last node fastest, so a budget smaller
        # than the space explores an order-biased prefix. The unbiased statement
        # about dilution is the density under uniform random draws.
        rng = random.Random(7)
        counts = [len(x.candidates) for x in prog.nodes]
        hits = 0
        t1 = time.perf_counter()
        for _ in range(args.draws):
            sel = {k: rng.randrange(c) for k, c in zip(names, counts)}
            err = evaluate(prog, sel, train, sig, r, tol)
            hits += err is not None and err <= tol
        arm["random_draws"] = {"draws": args.draws, "hits": hits,
                               "density": hits / args.draws,
                               "seconds": time.perf_counter() - t1}
        arm["conforming_density"] = hits / args.draws
        arm["projected_conforming"] = arm["conforming_density"] * n
        if conf["count"]:
            errs = [exact_error(prog, held, sig, r, {k: s.get(k, 0) for k in names})
                    for s in conf["conforming"][:200]]
            arm["held_out_max_error_first"] = errs[0]
            arm["held_out_exact_fraction"] = sum(e == 0. for e in errs) / len(errs)
            arm["held_out_checked"] = len(errs)
        # relaxed arm: the same fixture, gradient descent, explicit logit noise
        grows = []
        for seed in range(args.grad_seeds):
            t0 = time.perf_counter()
            _, exported, info = local_fit(prog, train, sig, steps=args.grad_steps, lr=.15,
                                          registry=r, init_noise=.5, seed=seed)
            sel = {k: 0 for k in [x.name for x in exported.nodes]}
            err = exact_error(exported, train, sig, r, sel)
            grows.append({"seed": seed, "seconds": time.perf_counter() - t0,
                          "exact_max_error": err, "conforming": err <= tol,
                          "held_out_max_error": exact_error(exported, held, sig, r, sel)})
        arm["gradient"] = {"rows": grows, "init_noise": .5, "steps": args.grad_steps,
                           "successes": sum(x["conforming"] for x in grows),
                           "held_out_exact": sum(x["held_out_max_error"] == 0. for x in grows)}
        result[label.replace(" ", "_")] = arm
        report(f"{label}: space / evaluated / seconds / conforming / unique",
               f"{n} / {conf['evaluated']} / {wall:.1f} / {conf['count']} / {arm['unique']}")
        if conf["count"]:
            report(f"{label}: first conforming held-out max error / held-out-exact fraction",
                   f"{arm['held_out_max_error_first']} / {arm['held_out_exact_fraction']:.4f}")
        report(f"{label}: uniform random conforming density ({args.draws} draws) / projected conforming",
               f"{arm['conforming_density']:.5f} / {arm['projected_conforming']:.0f}")
        report(f"{label}: gradient train-exact / held-out-exact of {args.grad_seeds} seeds",
               f"{arm['gradient']['successes']} / {arm['gradient']['held_out_exact']}")
        dump(args.tag, result)

    a, b = result["shipped_(byte_only)"], result["with_commitment"]
    result["cost"] = {
        "space_ratio": b["space_size"] / a["space_size"],
        "conforming_density_ratio": b["conforming_density"] / max(1e-12, a["conforming_density"]),
        "projected_exhaustive_seconds_ratio":
            b["projected_exhaustive_seconds"] / max(1e-9, a["projected_exhaustive_seconds"]),
        "held_out_exact_fraction_delta":
            b.get("held_out_exact_fraction", 0.) - a.get("held_out_exact_fraction", 0.),
        "gradient_success_delta": b["gradient"]["successes"] - a["gradient"]["successes"],
        "gradient_held_out_exact_delta": b["gradient"]["held_out_exact"] - a["gradient"]["held_out_exact"]}
    report("cost: space x / conforming density x / projected exhaustive x",
           f"{result['cost']['space_ratio']:.1f}x / "
           f"{result['cost']['conforming_density_ratio']:.2f}x / "
           f"{result['cost']['projected_exhaustive_seconds_ratio']:.1f}x")
    report("cost: held-out-exact fraction delta / gradient success delta",
           f"{result['cost']['held_out_exact_fraction_delta']:+.4f} / "
           f"{result['cost']['gradient_success_delta']:+d}")
    dump(args.tag, result)
    print(f"\nwrote {(pathlib.Path(__file__).parent / 'out' / (args.tag + '.json'))}")


if __name__ == "__main__":
    main()
