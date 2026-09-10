"""Mine the full corpus under both identity relations, per band, and publish
every arm's module into a real `tcn.library.Library` on disk.

One process, a few seconds per band: the expensive part of this track is the
enumeration, not the mining.  Writes `out/mined_*.json` (the complete ranked
table for every corpus and identity) and `out/modules_<band>.json` (what each
arm actually got, with digests, so §55's distinctness check can be made from
raw data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _paths  # noqa: F401

import armlib
import family
import mining

OUT = Path(__file__).resolve().parent / "out"

BANDS = ("C-trace", "C-minall")

# arm -> (identity, selection rule) for the arms whose module is mined
MINED = {
    "arm2_syntactic": ("syntactic", "rank1"),
    "arm2s_semantic": ("semantic", "rank1"),
    "arm4b_wrong_mined": ("syntactic", "runnerup"),
    "arm4s_runnerup": ("semantic", "runnerup"),
    "arm4s_matched": ("semantic", "node_matched"),
}
LABEL = {"arm2_syntactic": "syn_rank1", "arm2s_semantic": "sem_rank1",
         "arm4b_wrong_mined": "syn_runnerup", "arm4s_runnerup": "sem_runnerup",
         "arm4s_matched": "sem_matched"}


def main(bands=BANDS):
    OUT.mkdir(exist_ok=True)
    for band in bands:
        lib = armlib.lib_for(band)
        lib.mkdir(exist_ok=True)
        tables = {}
        for ident in ("semantic", "syntactic"):
            tables[ident] = mining.run("w4", band, (), ident,
                                       label=f"w4_{band}_{ident}")
            t = tables[ident]
            print(f"{band:9s} {ident:9s} entries {t['entries']:4d} eligible "
                  f"{t['eligible']:3d} rank1_is_window {t['rank1_is_window']} "
                  f"window_rank {t['best_window_rank']} window_classes "
                  f"{t['window_classes']}", flush=True)
        off = mining.run("x4", band, (), "semantic", label=f"x4_{band}_semantic")
        print(f"{band:9s} offfamily entries {off['entries']:4d} eligible "
              f"{off['eligible']:3d} rank1_is_window {off['rank1_is_window']}",
              flush=True)

        published = {}
        for arm, (ident, rule) in MINED.items():
            row = mining.pick(tables[ident], rule)
            if row is None:
                published[arm] = {"no_candidate": True, "rule": rule,
                                  "identity": ident}
                print(f"  {arm:22s} NO CANDIDATE under rule {rule}", flush=True)
                continue
            published[arm] = mining.publish(tables[ident], row, LABEL[arm], lib)
            published[arm]["rule"] = rule
            published[arm]["identity"] = ident
            print(f"  {arm:22s} rank {row['rank']:3d} arity {row['arity']} "
                  f"nodes {row['nodes']} tasks {len(row['tasks'])} window "
                  f"{row['is_window']} {row['digest'][:12]}", flush=True)
        # the off-family arm: rank-1 arity-4 class of the F'' pooled table
        orow = mining.pick(off, "rank1_windowarity") or mining.pick(off, "rank1")
        published["arm4s_offfamily"] = mining.publish(off, orow, "sem_off", lib)
        published["arm4s_offfamily"].update(rule="rank1_windowarity",
                                            identity="semantic", family="x4")
        print(f"  {'arm4s_offfamily':22s} rank {orow['rank']:3d} arity "
              f"{orow['arity']} nodes {orow['nodes']} {orow['digest'][:12]}",
              flush=True)
        # the exploratory 'best window class whatever its rank', both identities
        for ident in ("semantic", "syntactic"):
            w = mining.pick(tables[ident], "best_window")
            if w is not None:
                published[f"best_window_{ident}"] = mining.publish(
                    tables[ident], w, f"win_{ident}", lib)

        (OUT / f"modules_{band}.json").write_text(
            json.dumps(published, indent=2, sort_keys=True))
        digests = {a: p.get("digest") for a, p in published.items()
                   if not p.get("no_candidate")}
        print(f"  distinct classes among arms: "
              f"{len(set(digests.values()))} of {len(digests)}", flush=True)
        print("  ", json.dumps(digests, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or BANDS)
