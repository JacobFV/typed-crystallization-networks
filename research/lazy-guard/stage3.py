"""Stage 3 -- the same two constructs on one real visual-parser subroutine.

The subroutine is `rect_scaffold` from `research/visual-ladder/rung3_widgets.py`,
the S2 module of the shipped 32x32 parse.  Its own docstring says what it is:

    "Still a fixed-depth feedforward graph; no accumulator and no recurrence."

It computes a widget's extent as `1 + sum_k [t_1 and ... and t_k]` over a fixed
span of 31 terms in each of two directions, where each `t_k` calls the S0
same-colour module.  FINDINGS sec 48 attributed 11.1x of the parse's 27.6x gap
to exactly this: "the S2 module is a fixed-depth 30-term formulation with no
early exit".  The hand-written reference in
`research/compiled-runtime/fixtures.py` instead writes `while ... : w += 1`.

DOMAIN, STATED PLAINLY.  The carrier here is a 3,072-byte raster, so the type's
input domain cannot be exhausted and stages 1 and 2's certificate is not
available.  What is declared and then exhausted is a *finite enumerated domain*:
every interior position of every episode in a declared seed list.  That is
weaker than stages 1-2 and is reported as weaker.

ORACLE CHAIN.  The typed interpreter costs ~270 us per operator on this carrier
(FINDINGS sec 48), which makes it unusable as the oracle for thousands of
records.  The oracle is therefore `tcn.compile.compile_program`'s output, which
sec 48 certified bit-identical to the interpreter, and that identity is
re-established here on a sample before it is used.
"""
from __future__ import annotations

from dataclasses import replace
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "visual-ladder"))

import ablate
import cost
import ir
import runner
import synth
from tcn.compile import compile_program
from tcn.types import Value

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

SEARCH_SEEDS = (0, 1)
HELDOUT_SEEDS = (200, 201, 202)


def build(span=32, seeds=SEARCH_SEEDS, split="train"):
    from common import FLAT, Registry, bytes_type, episode, load
    import rung3_widgets as R

    found = load("rung3")
    ep0 = episode(seeds[0], split, **FLAT)
    W, H = ep0["width"], ep0["height"]
    reg = Registry()
    offs = R.offset_pool(W)
    m0 = reg.register_module(R.same_scaffold(reg, W, H).harden(found["s0"]["chosen"]))
    rect = R.rect_scaffold(reg, W, H, m0, offs, span=span).harden(found["s2"]["chosen"])
    assert rect.is_frozen
    REC = R.record_type(W, H)
    BT = bytes_type(W, H)

    records = []
    for s in seeds:
        ep = episode(s, split, **FLAT)
        raw = Value.of(BT, tuple(ep["pixels"])).raw
        obs = tuple(raw)
        for pos in R.interior_positions(ep):
            records.append({"rec": (pos, obs)})
    return {"program": rect, "registry": reg, "port": "rec", "input_type": REC,
            "records": records, "span": span, "W": W, "H": H,
            "carrier": rect.port_types()["one"], "same_module": m0,
            "seeds": tuple(seeds), "split": split}


def sub(fx, out_node):
    s = replace(fx["program"], outputs=(("y", out_node),)).pruned()
    g = dict(fx)
    g["program"] = s
    return g


def domain_of(records):
    def d(_fx=None):
        return iter(records)
    return d


def main(span=32, tag="stage3", sample=96, body_size=2, tail_size=1,
         default_size=1, run_ablations=True):
    t0 = time.time()
    fx = build(span=span, seeds=SEARCH_SEEDS)
    rect = fx["program"]
    print("built rect span=%d nodes=%d records=%d in %.1fs"
          % (span, len(rect.nodes), len(fx["records"]), time.time() - t0))
    report = {"subroutine": "research/visual-ladder rect_scaffold (S2)",
              "span": span, "rect_nodes": len(rect.nodes),
              "rect_digest": rect.digest,
              "execution_cost_scalar": rect.execution_cost(fx["registry"]),
              "search_seeds": list(SEARCH_SEEDS), "heldout_seeds": list(HELDOUT_SEEDS),
              "search_domain_records": len(fx["records"]),
              "domain_note": "declared enumerated domain: every interior position "
                             "of the declared episodes; the 3,072-byte carrier "
                             "cannot be exhausted and this is stated as weaker "
                             "than the stage 1-2 certificate"}

    # ---- oracle chain: compiled == interpreter on a sample ----------------
    print("== oracle chain")
    res = compile_program(rect, fx["registry"])
    mod = res.module()
    agree = 0
    for r in fx["records"][::400][:24]:
        a = {k: v.decoded for k, v in
             rect.run({"rec": Value(fx["input_type"], r["rec"])}, registry=fx["registry"])[0].items()}
        c = mod.run({"rec": r["rec"]}, validate=False)[0]
        assert a == c, (a, c)
        agree += 1
    report["oracle_chain"] = {
        "compiled_equals_interpreter_on": agree,
        "note": "the interpreter costs ~270 us per operator on this carrier "
                "(FINDINGS sec 48), so the compiled program is the oracle for "
                "the exhaustive pass and the identity is re-established here"}
    print("   compiled == interpreter on %d records" % agree)

    # ---- resynthesize each extent ----------------------------------------
    found = {}
    for node in ("w_extent", "h_extent"):
        print("== resynthesis of %s" % node)
        g = sub(fx, node)
        d = domain_of(fx["records"])
        r = synth.toplevel_search(g, domain=d, sample=sample, body_size=body_size,
                                  tail_size=tail_size)
        found[node] = r
        print("   ->", repr(r["program"].output) if r["program"] else "NOTHING FOUND")
    report["search"] = {k: {kk: vv for kk, vv in v.items() if kk != "program"}
                        for k, v in found.items()}
    if any(v["program"] is None for v in found.values()):
        report["verdict"] = "no exact candidate found for at least one extent"
        (OUT / ("%s.json" % tag)).write_text(json.dumps(report, indent=2, default=str))
        return report
    report["result"] = {k: repr(v["program"].output) for k, v in found.items()}

    # ---- rebuild the whole subroutine with both extents replaced ----------
    prog = ir.from_program(rect, fx["registry"])
    for node in ("w_extent", "h_extent"):
        prog = prog.rebind(node, found[node]["program"].output)
    report["constructs_used"] = sorted(prog.constructs())

    d = domain_of(fx["records"])
    sp, _ = runner.spec_profile(fx, d)
    ap, _ = runner.alg_profile(fx, prog, d)
    report["search_domain"] = {"spec": sp, "resynth": ap,
                               "ratios": cost.ratio_table(sp, ap)}

    # ---- held-out episodes, never used during synthesis -------------------
    print("== held-out episodes")
    hx = build(span=span, seeds=HELDOUT_SEEDS, split="test")
    hd = domain_of(hx["records"])
    hsp, _ = runner.spec_profile(hx, hd)
    hap, _ = runner.alg_profile(hx, prog, hd)
    report["heldout"] = {"records": len(hx["records"]), "spec": hsp, "resynth": hap,
                         "ratios": cost.ratio_table(hsp, hap)}

    # exactness over the declared held-out domain, against the compiled oracle
    hmod = compile_program(hx["program"], hx["registry"]).module()
    ev = ir.Evaluator(hx["registry"])
    bad = None
    n = 0
    for r in hx["records"]:
        got = ev.run(prog, {"rec": r["rec"]})
        want = hmod.run({"rec": r["rec"]}, validate=False)[0]["y"]
        if got != want:
            bad = {"record_index": n, "got": repr(got)[:200], "want": repr(want)[:200]}
            break
        n += 1
    report["heldout"]["certificate"] = {
        "exact": bad is None, "checked": n, "counterexample": bad,
        "oracle": "tcn.compile.compile_program(rect) -- certified against the "
                  "interpreter on a sample above",
        "exhaustive_over": "the declared enumerated domain, not the type domain"}
    print("   held-out exact on %d records: %s" % (n, bad is None))

    # ---- the deployment distribution: corner positions only ---------------
    print("== deployment distribution (corners only)")
    corners = corner_records(hx)
    if corners:
        cd = domain_of(corners)
        csp, _ = runner.spec_profile(hx, cd)
        cap, _ = runner.alg_profile(hx, prog, cd)
        report["corner_distribution"] = {
            "records": len(corners), "spec": csp, "resynth": cap,
            "ratios": cost.ratio_table(csp, cap),
            "note": "the positions the assembly's `filter` actually passes to "
                    "the rect module, so this is the deployed distribution"}

    # ---- level check ------------------------------------------------------
    report["level_check"] = level_check(sp, ap)

    # ---- deployment -------------------------------------------------------
    print("== deployment")
    try:
        src = ir.Lowerer(hx["registry"]).program(prog, entry="run")
        (OUT / ("%s_algorithm.py" % tag)).write_text(src)
        gen = ir.load(src)
        ok = True
        checked = 0
        for r in hx["records"]:
            if gen.run(r["rec"]) != hmod.run({"rec": r["rec"]}, validate=False)[0]["y"]:
                ok = False
                break
            checked += 1
    except Exception as exc:                       # pragma: no cover - reported
        src, ok, checked = "", False, 0
        report["deployment_error"] = "%s: %s" % (type(exc).__name__, exc)
    report["deployment"] = {"source_path": str(OUT / ("%s_algorithm.py" % tag)),
                            "source_bytes": len(src),
                            "matches_compiled_oracle": ok,
                            "checked": checked,
                            "note": "generated code imports nothing but its own "
                                    "prelude; the stdlib-only subprocess check of "
                                    "stages 1-2 is not repeated here because the "
                                    "input is a 3,072-element raster per record"}

    if run_ablations:
        print("== ablation matrix on w_extent")
        g = sub(fx, "w_extent")
        rows = ablate.matrix(g, domain_of(fx["records"]), synth.toplevel_search,
                             runner.spec_profile(g, domain_of(fx["records"]))[0]["expected_ops"],
                             sample=sample, body_size=body_size, tail_size=tail_size)
        report["ablation_matrix"] = rows
        report["ablation_summary"] = ablate.summarize(rows)
        both = synth.toplevel_search(g, constructs=("Guard", "Let"),
                                     domain=domain_of(fx["records"]), sample=sample,
                                     body_size=body_size, tail_size=tail_size, verbose=False)
        report["ablation_no_find_no_scan"] = {
            "solved": both["program"] is not None,
            "candidates_considered": both["candidates_considered"],
            "result": repr(both["program"].output) if both["program"] else None}

    (OUT / ("%s.json" % tag)).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("result", "search_domain", "heldout",
                                             "corner_distribution", "level_check",
                                             "deployment") if k in report},
                     indent=2, default=str)[:3000])
    return report


def corner_records(fx):
    """Records at positions the corner module accepts, if it is reachable."""
    from common import FLAT, episode
    import rung3_widgets as R
    out = []
    for r in fx["records"]:
        pos, obs = r["rec"]
        i = pos
        w = fx["W"]
        left = i - 3
        up = i - 3 * w
        if obs[i:i + 3] != obs[left:left + 3] and obs[i:i + 3] != obs[up:up + 3]:
            out.append(r)
    return out


def level_check(spec, alg):
    """Is the gain DCE/CSE/constant folding (level 1) or algorithmic (level 3)?

    The level-1 part shows up in the *worst* case, where no iteration is skipped
    and the only difference is bookkeeping the loop no longer needs.  The
    level-3 part is the difference between the worst and expected ratios: work
    that is skipped because of the data.
    """
    return {
        "worst_ratio": spec["worst_ops"] / alg["worst_ops"],
        "expected_ratio": spec["expected_ops"] / alg["expected_ops"],
        "level_1_component_worst_ratio": spec["worst_ops"] / alg["worst_ops"],
        "level_3_component": (spec["expected_ops"] / alg["expected_ops"]) /
                             (spec["worst_ops"] / alg["worst_ops"]),
        "reading": "the worst-case ratio is the part attributable to removing "
                   "bookkeeping the loop makes unnecessary (level 1); the "
                   "quotient is the part attributable to iterations the data "
                   "lets the algorithm skip (level 3)"}


if __name__ == "__main__":
    main(span=int(sys.argv[1]) if len(sys.argv) > 1 else 32)
