"""Apply the searched same-object module at every position, at several widths.

The module's *input type* names the observation width, so a module hardened at
one resolution cannot be registered against another -- the previous track's
resolution climb rebuilds the scaffold at each width and reuses the
*selections*, which is legitimate because the program never names an absolute
address.  Doing it the other way raises `map: operator signature mismatch`, and
that is what it means.

Both callers are exercised: the copy in `research/discrete-perception/common.py`
and the merged `tcn.scaffold.positional_scaffold`, and their outputs are asserted
equal.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "discrete-perception"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import IDX, bytes_type, positional_caller, report
import common as DP
from common2 import OUT
from rung4_segment import collinear_scaffold, dump, same_labels, window_positions
from tcn.operators import Registry
from tcn.scaffold import positional_scaffold
from tcn.types import BOOL, Value, product, setof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="8,16,24,32")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--objects", type=int, default=6)
    args = ap.parse_args()
    r = Registry()
    wd = json.loads((OUT / "wide_threshold.json").read_text())
    pool = tuple(range(wd["threshold_pool"]))
    sel = wd["validation_filtered"]["selections"]
    report("selections from the wide arm (T = %d)" % wd["validation_filtered"]["threshold"],
           json.dumps(sel))
    rows = []
    for RR in [int(x) for x in args.widths.split(",")]:
        # rebuild at this width, reuse the selections: the module computes every
        # address from its own position argument, so only the input type changes
        module = collinear_scaffold(r, RR, thresholds=pool).harden(sel)
        name = r.register_module(module)
        obs_t = bytes_type(RR)
        starts = window_positions(RR)
        caller = positional_caller(r, obs_t, starts, [name])
        merged = positional_scaffold(r, obs_t, starts, [name], index=IDX)
        LAB = product(IDX, BOOL)
        t0 = time.perf_counter(); worst = 0.; agree = True; wrong = 0; slots = 0
        for s in range(200, 200 + args.episodes):
            pixels, probes = DP.episode(s, RR, split="test", objects=args.objects)
            obs = Value.of(obs_t, pixels)
            labels = same_labels(probes, RR)
            want = Value.of(setof(LAB, len(starts)), tuple((p, labels[p]) for p in starts))
            got, _, _ = caller.execute({"observation": obs}, registry=r)
            got2, _, _ = merged.execute({"observation": obs}, registry=r)
            agree &= got["mapped"].decoded == got2["mapped"].decoded
            g = dict(got["mapped"].decoded); w = dict(want.decoded)
            wrong += sum(1 for k in w if g.get(k) != w[k]); slots += len(w)
            worst = max(worst, 0. if got["mapped"].decoded == want.decoded else 1.)
        rows.append({"resolution": RR, "observation_bytes": 3 * RR * RR,
                     "positions": len(starts), "caller_nodes": len(caller.nodes),
                     "merged_scaffold_nodes": len(merged.nodes),
                     "merged_agrees": bool(agree),
                     "slots": slots, "wrong_slots": wrong,
                     "accuracy": 1 - wrong / max(1, slots), "max_error": worst,
                     "seconds": time.perf_counter() - t0})
        report(f"[apply R={RR}] bytes/positions/caller nodes/wrong slots/merged agrees/s",
               f"{3*RR*RR} / {len(starts)} / {len(caller.nodes)} / "
               f"{wrong}/{slots} / {agree} / {rows[-1]['seconds']:.2f}")
        dump("apply", {"arguments": vars(args), "selections": sel,
                       "threshold": wd["validation_filtered"]["threshold"], "rows": rows})


if __name__ == "__main__":
    main()
