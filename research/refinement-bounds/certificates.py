"""PREREGISTRATION section 3 -- what the declared bound costs everywhere else.

A refinement bound narrows a type, and a narrower type could in principle change
what the scaffold's search enumerates, which candidates conform, and therefore
the certificate attached to every number `research/visual-ladder` reports.  This
re-runs the exhaustive sweeps and the parse in the bounded arm and compares them
field by field with `research/visual-ladder/out/rung3.json` and
`out/rung3_root.json`.

FALSIFICATION F6: if `space_size`, `evaluated`, `conforming`, `certificate`,
`exhausted`, `unique`, the selected candidate, the rectangle counts, the 227
parent links or the 12 exact trees move, that is a **regression**, reported as
one however fast the artifact runs.

    python certificates.py space          # space_size only, seconds
    python certificates.py search         # the exhaustive sweeps, ~40 min
    python certificates.py parse          # section 33's parse, ~10 min
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as ARMS                                     # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
REF = ROOT / "research" / "visual-ladder" / "out"
CERT_FIELDS = ("space_size", "evaluated", "conforming", "certificate", "exhausted",
               "unique", "solved", "node_evaluations")


def _setup(variant, bounded):
    """Registry, module handles and the three scaffolds, in one arm."""
    ARMS._track()
    from common import FLAT, IDX, Registry, address_type, episode, load
    import rung3_widgets as R
    reg = Registry()
    probe = episode(0, "train", **FLAT)
    W, H = probe["width"], probe["height"]
    A = address_type(W, H) if bounded else IDX
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(reg, W, H, addr=A)
    found = load("rung3")
    m0 = reg.register_module(ARMS.harden_all(p0, found["s0"]["chosen"]))
    if variant == "widgets":
        p1 = R.corner_scaffold(reg, W, H, m0, offsets, addr=A)
        p2 = R.rect_scaffold(reg, W, H, m0, offsets, addr=A, reformulated_clamp=bounded)
    else:
        import rung3_root as RR
        p1 = RR.corner_scaffold_masked(reg, W, H, m0, offsets, addr=A)
        p2 = RR.rect_scaffold_clamped(reg, W, H, m0, offsets, addr=A,
                                      reformulated_clamp=bounded)
    return reg, R, A, W, H, offsets, p0, p1, p2, FLAT


def space(argv=()):
    from tcn.search import space_size
    rep = {}
    for variant in ("widgets", "root"):
        for bounded in (False, True):
            _r, _R, _A, _W, _H, _o, p0, p1, p2, _F = _setup(variant, bounded)
            rep["%s_%s" % (variant, "bounded" if bounded else "declared_IDX")] = {
                "S0_space": space_size(p0), "S1_space": space_size(p1),
                "S2_space": space_size(p2),
                "S0_nodes": len(p0.nodes), "S1_nodes": len(p1.nodes), "S2_nodes": len(p2.nodes)}
    rep["reference"] = {k: json.load(open(REF / ("%s.json" % f)))[k]["space_size"]
                        for f, ks in (("rung3", ("s0", "s1", "s2")),) for k in ks}
    rep["reference_root"] = {k: json.load(open(REF / "rung3_root.json"))[k]["space_size"]
                             for k in ("s1", "s2")}
    return rep


def search(argv=()):
    """The exhaustive sweeps, in both arms, with the certificate preserved."""
    ARMS._track()
    from common import sweep
    rep = {}
    variants = [a for a in argv if a in ("widgets", "root")] or ["widgets", "root"]
    for variant in variants:
        for bounded in (False, True):
            key = "%s_%s" % (variant, "bounded" if bounded else "declared_IDX")
            reg, R, A, W, H, offsets, p0, p1, p2, FLAT = _setup(variant, bounded)
            if variant == "widgets":
                s0tr = R.same_examples(range(6), "train", 48, seed=1, addr=A, **FLAT)
                s1tr = R.corner_examples(range(6), "train", 120, seed=4, addr=A, **FLAT)
                s2tr = R.rect_examples(range(6), "train", addr=A, **FLAT)
            else:
                import rung3_root as RR
                s0tr = R.same_examples(range(6), "train", 48, seed=1, addr=A, **FLAT)
                s1tr = RR.corner_examples_all(range(6), "train", 120, seed=4, addr=A, **FLAT)
                s2tr = RR.rect_examples_all(range(6), "train", addr=A, **FLAT)
            rep[key] = {}
            for tag, prog, ex, sig in (("s0", p0, s0tr, R.same_signals()),
                                       ("s1", p1, s1tr, R.corner_signals()),
                                       ("s2", p2, s2tr, R.rect_signals(A))):
                t0 = time.time()
                res = sweep(prog, ex, sig, reg, 1e-6)
                rep[key][tag] = {k: res.get(k) for k in CERT_FIELDS}
                rep[key][tag]["selections"] = res.get("selections")
                rep[key][tag]["train_records"] = len(ex)
                rep[key][tag]["seconds"] = round(time.time() - t0, 1)
                print("  %-22s %s space %s evaluated %s conforming %s cert %s (%.0fs)"
                      % (key, tag, res.get("space_size"), res.get("evaluated"),
                         res.get("conforming"), res.get("certificate"), time.time() - t0),
                      flush=True)
                (OUT / "certificates_search.json").write_text(json.dumps(rep, indent=1,
                                                                        default=str))
    return rep


def parse(argv=()):
    """Section 33's parse: 227 / 227 links and 12 / 12 exact trees, in both arms."""
    ARMS._track()
    from common import bytes_type, episode, load
    from tcn.types import Value
    rep = {}
    for bounded in (False, True):
        reg, R, A, W, H, offsets, p0, p1, p2, FLAT = _setup("root", bounded)
        import rung3_root as RR
        found = load("rung3_root")
        m1 = reg.register_module(ARMS.harden_all(p1, {k: v for k, v in found["s1"]["chosen"].items()
                                                      if k in ("back_a", "back_b", "corner")}))
        m2 = reg.register_module(ARMS.harden_all(p2, {k: v for k, v in found["s2"]["chosen"].items()
                                                      if k in ("step_w", "step_h")}))
        rows = []
        for seed in range(200, 212):
            ep = episode(seed, "test", **FLAT)
            BT = bytes_type(ep["width"], ep["height"])
            positions = RR.all_positions(ep)
            parser = R.assembly(reg, BT, positions, m1, m2, addr=A)
            t0 = time.perf_counter()
            got = parser.run({"observation": Value.of(BT, ep["pixels"])}, registry=reg)[0]
            predicted = [list(map(int, r)) for r in sorted(got["mapped"].decoded)]
            row = RR.score_with_root(predicted, ep) | {"seed": seed,
                                                       "seconds": time.perf_counter() - t0,
                                                       "positions": len(positions)}
            rows.append(row)
            print("  %s seed %d rects %d/%d exact=%s links %d/%d tree_exact=%s (%.0fs)"
                  % ("bounded  " if bounded else "IDX      ", seed, row["rects_predicted"],
                     row["rects_true"], row["rects_exact"], row["parent_links_correct"],
                     row["widgets_in_probe"], row["tree_exact"], row["seconds"]), flush=True)
        rep["bounded" if bounded else "declared_IDX"] = {
            "screens": len(rows),
            "widgets_in_probe": sum(r["widgets_in_probe"] for r in rows),
            "rects_predicted": sum(r["rects_predicted"] for r in rows),
            "screens_rects_exact": sum(r["rects_exact"] for r in rows),
            "roots_predicted": sum(r["roots_predicted"] for r in rows),
            "parent_links_correct": sum(r["parent_links_correct"] for r in rows),
            "parent_links_wrong": sum(r["parent_links_wrong"] for r in rows),
            "trees_exact": sum(r["tree_exact"] for r in rows),
            "per_screen": rows}
        (OUT / "certificates_parse.json").write_text(json.dumps(rep, indent=1, default=str))
    ref = json.load(open(REF / "rung3_root.json"))["parse"]
    rep["reference_totals"] = ref.get("totals", ref)
    (OUT / "certificates_parse.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def main(argv):
    what = argv[1] if len(argv) > 1 else "space"
    rep = {"space": space, "search": search, "parse": parse}[what](argv[2:])
    if what == "space":
        (OUT / "certificates_space.json").write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps(rep, indent=1))
    elif what == "parse":
        for k in ("declared_IDX", "bounded"):
            r = rep[k]
            print("%-12s links %d/%d  trees %d/%d  rects %d  roots %d"
                  % (k, r["parent_links_correct"], r["widgets_in_probe"], r["trees_exact"],
                     r["screens"], r["rects_predicted"], r["roots_predicted"]))


if __name__ == "__main__":
    main(sys.argv)
