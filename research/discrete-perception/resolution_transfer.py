"""The resolution climb: search once, apply at every width.

The per-position module's discrete space does not depend on the image, because
every address it reads is computed from its own position argument.  So the
selections found at one resolution are the selections at every resolution, and
the climb costs no additional search at all -- only application.

Measured here for each resolution:

  * the module searched at `--search-resolution` is hardened, re-typed for the
    wider observation (the *selections* transfer verbatim; only the input type
    changes) and applied at every position by the three-node caller;
  * exact error against the whole dense `object_ids >= 0` probe, and the
    `filter`+`count` aggregate against the true foreground count;
  * what a fresh exhaustive search *would* cost at that width, from the measured
    evaluation rate of a capped `enumerate_fit` over the same 32,000 programs.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BYTE, IDX, Builder, bytes_type, dump, enumerate_reference, episode,
                    exact_error, pixel_starts, positional_caller, report)
from rung3_mask import (POOL, keep_flag_module, module_scaffold, pixel_examples, signals,
                        target_set)
from tcn.operators import Registry
from tcn.search import space_size
from tcn.types import BOOL, Value, product, setof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-resolution", type=int, default=8)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--per-image", type=int, default=48)
    ap.add_argument("--apply", type=int, default=3)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--probe-budget", type=int, default=2000)
    ap.add_argument("--tag", default="resolution_transfer")
    args = ap.parse_args()

    tol = 1e-6
    r = Registry()
    SR = args.search_resolution
    sig = signals()

    # search once, at the smallest width
    # Section 4's lesson applied: the conforming set is under-determined, so the
    # module is selected by exactness at EVERY position of a disjoint validation
    # split, not by candidate order.  `--per-image 0` supervises every position.
    per = None if args.per_image <= 0 else args.per_image
    train = pixel_examples(tuple(range(args.train)), SR, "train", per, seed=1,
                           objects=args.objects)
    valid = pixel_examples(tuple(range(100, 108)), SR, "test", None, seed=2,
                           objects=args.objects)
    scaffold = module_scaffold(r, SR, POOL)
    from incremental import incremental_conforming
    from tcn.search import evaluate
    t0 = time.perf_counter()
    ident = incremental_conforming(scaffold, train, sig, r, tolerance=tol)
    filtered = [s for s in ident["conforming"]
                if (lambda e: e is not None and e <= tol)(
                    evaluate(scaffold, s, valid, sig, r, tol))]
    selections = filtered[0] if filtered else ident["conforming"][0]
    result = {"arguments": vars(args), "search_resolution": SR,
              "search_seconds": time.perf_counter() - t0,
              "space_size": space_size(scaffold),
              "conforming_train": ident["count"],
              "conforming_train_and_validation": len(filtered),
              "selections": selections, "rows": []}
    report("searched once at R=%d: conforming train / + validation / space" % SR,
           f"{ident['count']} / {len(filtered)} / {space_size(scaffold)}")

    for R in args.resolutions:
        scaffold_R = module_scaffold(r, R, POOL)
        assert space_size(scaffold_R) == result["space_size"]
        module = scaffold_R.harden(selections)
        name = r.register_module(module)
        BT = bytes_type(R)
        starts = pixel_starts(R)
        caller = positional_caller(r, BT, starts, [name])
        LAB = product(IDX, BOOL)
        keep = r.register_module(keep_flag_module(r, LAB))
        b = Builder(r, (("observation", BT),),
                    (("positions", Value.of(setof(IDX, len(starts)), starts)),
                     ("empty", Value.of(setof(BT, 1), ()))))
        b.add("held", "insert", ["empty", "observation"])
        b.add("records", "pair", ["positions", "held"])
        b.add("mapped", "map", ["records"], params={"module": name})
        b.add("kept", "filter", ["mapped"], params={"module": keep})
        b.add("total", "count", ["kept"])
        counter = b.program((("y", "total"),))

        errs, counts, secs, fg, total = [], [], [], 0, 0
        for s in range(200, 200 + args.apply):
            pixels, probes = episode(s, R, split="test", objects=args.objects)
            labels = [probes["object_ids"][i] >= 0 for i in range(len(starts))]
            fg += sum(labels); total += len(labels)
            data = Value.of(BT, pixels)
            t0 = time.perf_counter()
            got = caller.run({"observation": data}, registry=r)[0]["mapped"]
            secs.append(time.perf_counter() - t0)
            tgt = target_set(labels, R)
            errs.append(max((abs(x - y) for x, y in zip(got.flat(), tgt.flat())), default=0.))
            counts.append(abs(counter.run({"observation": data}, registry=r)[0]["y"].decoded
                              - sum(labels)))

        # what a fresh search would cost at this width
        probe = pixel_examples(tuple(range(2)), R, "train", 24, seed=7, objects=args.objects)
        capped = enumerate_reference(scaffold_R, probe, sig, r, tolerance=tol,
                                     max_programs=args.probe_budget)
        rate = capped["evaluated"] / max(1e-9, capped["seconds"])

        row = {"resolution": R, "observation_width": 3 * R * R, "positions": len(starts),
               "caller_nodes": len(caller.nodes),
               "applied_pixels": total, "foreground_fraction": fg / max(1, total),
               "max_error_over_dense_probe": max(errs),
               "max_count_error": max(counts),
               "median_apply_seconds": sorted(secs)[len(secs) // 2],
               "space_size": space_size(scaffold_R),
               "search_probe": {"programs_per_second": rate,
                                "projected_exhaustive_seconds": space_size(scaffold_R) / max(1e-9, rate),
                                "evaluated": capped["evaluated"],
                                "seconds": capped["seconds"],
                                "records": len(probe)}}
        result["rows"].append(row)
        report(f"R={R:3d} width {3*R*R:6d} nodes {len(caller.nodes)} err {max(errs)} "
               f"count-err {max(counts)}",
               f"apply {row['median_apply_seconds']:.2f}s, fresh-search projection "
               f"{row['search_probe']['projected_exhaustive_seconds']:.4g}s")
        dump(args.tag, result)


if __name__ == "__main__":
    main()
