"""Is the positional-reuse caller now DISCOVERED rather than hand-written?

The previous track's D4: on main at the time, `legal_candidates` resolved every
candidate with empty parameters, so `project`, `map`, `filter` and `join` were
never proposed and every scaffold in that track supplied its candidates
explicitly.  `operator_parameters` has since been merged.  This measures the
before/after directly -- the "before" is reproduced faithfully by passing
`parameters={...: [{}]}`, which is exactly what the old code did -- and then
searches the three caller nodes with free candidate sets instead of writing them.
"""
from __future__ import annotations
import json, sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "discrete-perception"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import bytes_type, exact_error, report, IDX
from common2 import OUT
from rung4_segment import (collinear_scaffold, dump, same_examples, same_labels,
                           same_signals, window_positions, THRESHOLDS)
from rung4_segment import offsets  # noqa: F401
from tcn.graph import Candidate, Node, Program, legal_candidates, Signal
from tcn.operators import Registry, STRUCTURAL
from tcn.scaffold import positional_scaffold
from tcn.search import enumerate_fit, space_size
from tcn.types import BOOL, Value, product, setof
import common as DP

NAMES = tuple(sorted(STRUCTURAL))


def main():
    R = 8
    r = Registry()
    tol = 1e-6
    # a module to call: the rung-4 collinearity module with the selections the
    # wide search returned (threshold 144, offset +3, XOR-free combinators)
    prog = collinear_scaffold(r, R)
    wd = json.loads((OUT / "wide_threshold.json").read_text()) if (OUT / "wide_threshold.json").exists() else None
    sel = None
    if wd and wd.get("validation_filtered"):
        # the wide arm's selections index a 512-value pool; rebuild that scaffold
        prog = collinear_scaffold(r, R, thresholds=tuple(range(wd["threshold_pool"])))
        sel = wd["validation_filtered"]["selections"]
    else:
        from rung4_segment import same_examples as _se
        raise SystemExit("run wide_threshold.py first")
    module = prog.harden(sel)
    name = r.register_module(module)
    obs_t = bytes_type(R)
    positions = window_positions(R)

    ports = {"observation": obs_t,
             "positions": setof(IDX, len(positions)),
             "empty": setof(obs_t, 1)}
    out = {"module": name, "ports": {k: v.kind for k, v in ports.items()}}

    # ---- what legal_candidates proposes, before and after ----
    def count(parameters, output=None, arities=(1, 2)):
        c = legal_candidates(r, NAMES, ports, output, arities=arities, limit=1 << 22,
                             parameters=parameters)
        return c

    old = {n: [{}] for n in ("project", "map", "filter", "join")}
    held = count(None)
    was = count(old)
    def by_op(cs):
        d = {}
        for c in cs:
            d[c.operator.name] = d.get(c.operator.name, 0) + 1
        return dict(sorted(d.items()))
    out["candidates_now"] = by_op(held)
    out["candidates_with_empty_parameters"] = by_op(was)
    report("legal_candidates now", json.dumps(out["candidates_now"]))
    report("legal_candidates with empty parameters (the old behaviour)",
           json.dumps(out["candidates_with_empty_parameters"]))

    # ---- the three caller nodes, searched instead of written ----
    holder = setof(obs_t, 1)
    locations = setof(IDX, len(positions))
    consts = (("positions", Value.of(locations, positions)), ("empty", Value.of(holder, ())))
    p0 = dict(ports)
    held_c = legal_candidates(r, NAMES, p0, holder, arities=(1, 2), limit=1 << 22)
    n1 = Node("held", holder, held_c, "core", 1, None)
    p1 = p0 | {"held": holder}
    rec_t = setof(product(IDX, obs_t), len(positions))
    rec_c = legal_candidates(r, NAMES, p1, rec_t, arities=(1, 2), limit=1 << 22)
    n2 = Node("records", rec_t, rec_c, "core", 2, None)
    p2 = p1 | {"records": rec_t}
    lab_t = setof(product(IDX, BOOL), len(positions))
    lab_c = legal_candidates(r, NAMES, p2, lab_t, arities=(1, 2), limit=1 << 22)
    n3 = Node("mapped", lab_t, lab_c, "core", 3, None)
    caller = Program((("observation", obs_t),), (n1, n2, n3), (("mapped", "mapped"),),
                     consts, input_depths=(("observation", 0),)).validate(r)
    out["discovered_caller"] = {"candidates_per_node":
                                [len(n1.candidates), len(n2.candidates), len(n3.candidates)],
                                "space_size": space_size(caller)}
    report("discovered caller: candidates per node / space",
           f"{out['discovered_caller']['candidates_per_node']} / {space_size(caller)}")

    # supervise it against the dense probe on two episodes
    LAB = product(IDX, BOOL)
    examples = []
    for s in (0, 1):
        pixels, probes = DP.episode(s, R, split="train", objects=6)
        labels = same_labels(probes, R)
        examples.append({"inputs": {"observation": Value.of(obs_t, pixels)},
                         "targets": {"mapped": Value.of(setof(LAB, len(positions)),
                                                        tuple((p, labels[p]) for p in positions))}})
    sig = (Signal("mapped", "mapped", ("core",), lab_t, "mse"),)
    t0 = time.perf_counter()
    res = enumerate_fit(caller, examples, sig, registry=r, tolerance=tol, max_programs=200000)
    out["search"] = res.to_dict() | {"wall": time.perf_counter() - t0}
    report("discovered caller search: solved/exhausted/unique/evaluated/s",
           f"{res.solved}/{res.exhausted}/{res.unique}/{res.evaluated}/{res.seconds:.1f}")

    # ---- and the merged tcn.scaffold.positional_scaffold agrees ----
    hand = positional_scaffold(r, obs_t, positions, [name], index=IDX)
    o1, _, _ = hand.execute({"observation": Value.of(obs_t, DP.episode(2, R, split="test",
                                                                      objects=6)[0])}, registry=r)
    o2, _, _ = caller.execute({"observation": Value.of(obs_t, DP.episode(2, R, split="test",
                                                                        objects=6)[0])},
                              registry=r, selections=res.selections)
    out["merged_scaffold_agrees"] = bool(o1["mapped"].decoded == o2["mapped"].decoded)
    out["merged_scaffold_nodes"] = len(hand.nodes)
    report("tcn.scaffold.positional_scaffold agrees with the discovered caller",
           out["merged_scaffold_agrees"])
    dump("discoverable", out)


if __name__ == "__main__":
    main()
