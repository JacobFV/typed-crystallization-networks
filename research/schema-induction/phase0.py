"""Phase 0: the validity checks that must pass BEFORE any arm of this track runs.

Reads only the `gap 0` and `gap 1` episode caches committed by
`research/integrated-flagship/out/`. It never constructs, reads or names a
`gap 2` or `gap 3` episode: those are the blind final split (PREREGISTRATION
§5), and this file is the one that must be auditable as not having touched them.

Three things are established here, all with exact integers:

  V2  EXPRESSIVITY.  For each relation and each admission configuration, the
      exact set of single-constant step specifications `op(base, o)` that place
      the click inside the target widget on *every* episode of that relation.
      S0's own offset pool is intersected with it. This is what makes the
      `gap 1` counterexample a removal of *expressiveness* rather than a
      narrowing of options (coordinator, 2026-09-10): no re-selection inside
      S0's pools can succeed, because the required displacement is not sayable.

  V3  WIDEN-INSUFFICIENCY.  The declared generic literal source L of the
      `WIDEN` family is the set of integer constants the scaffold already
      declares. The check is `max(L union S0_offsets) < min(required)`, so no
      finite sequence of `WIDEN` edits over L can reach the repair. If this
      fails the experiment has collapsed into "restore a candidate" and the
      design must change before anything else runs (F2).

  V8  COST PILOT.  The exact 48-episode conformer counter's wall clock as a
      function of the step pool's size, so the per-arm budget in
      PREREGISTRATION §7 is measured rather than guessed.

    python phase0.py                 # V2 + V3          -> out/phase0.json
    python phase0.py --pilot         # V2 + V3 + V8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLAGSHIP = ROOT / "research" / "integrated-flagship"
OUT = HERE / "out"

WIDTH = 16
N_PIXELS = 256
RAW = 3 * N_PIXELS
ADMISSION_GAPS = (0, 1)
RELATIONS = ("above", "left_of", "right_of")

# S0, verbatim from `research/visual-ladder/rung3_widgets.offset_pool(16)` as
# imported by `research/integrated-flagship/family.py`. Retyped here only so
# this check does not depend on the import path; V1 asserts they are equal.
S0_OFFSETS = (6, 3 * WIDTH + 3, 3, 3 * WIDTH, 9)
S0_OPS = ("add", "sub")
S0_BASES = ("lo", "hi")

# The declared generic literal source of `WIDEN` (PREREGISTRATION §3.2): the
# integer constants the scaffold `family.build` already declares, and nothing
# else. k0..k15 are the text-address constants, `three` is the raster stride,
# `one_i`/`two_i` are the scoring head's, and the o<off> constants are S0's own
# offsets. No element of this set was chosen with the answer in view.
WIDEN_SOURCE = tuple(sorted(set(range(16)) | {1, 2, 3} | set(S0_OFFSETS)))


def load(gap):
    p = FLAGSHIP / "out" / f"episodes_gap{gap}.json"
    return json.loads(p.read_text())


def anchor_bounds(ep):
    """(lo, hi) the schema's matcher yields for the anchor's colour: the first
    and last raw byte address whose pixel carries that colour."""
    pixels = np.array(ep["pixels"], dtype=np.int64)
    widgets = {w["id"]: w for w in ep["widgets"]}
    colour = widgets[ep["anchor"]]["colour"]
    base = np.arange(N_PIXELS) * 3
    same = ((pixels[base] == colour[0]) & (pixels[base + 1] == colour[1])
            & (pixels[base + 2] == colour[2]))
    idx = np.flatnonzero(same)
    assert idx.size, "the anchor's colour occurs nowhere: the episode is malformed"
    return int(idx[0]) * 3, int(idx[-1]) * 3


def episode_hits(ep):
    """hit[base, op, offset]: does `op(base, offset)` click inside the target?"""
    lo, hi = anchor_bounds(ep)
    owner = np.array(ep["owner"], dtype=np.int64)
    offs = np.arange(RAW)
    out = np.zeros((len(S0_BASES), len(S0_OPS), RAW), dtype=bool)
    for bi, value in enumerate((lo, hi)):
        for oi, sign in enumerate((+1, -1)):          # add, sub
            click = (value + sign * offs) % (1 << 16)
            q = click // 3
            inside = q < N_PIXELS
            row = np.zeros(RAW, dtype=bool)
            row[inside] = owner[q[inside]] == ep["target"]
            out[bi, oi] = row
    return out


def as_ranges(values):
    out = []
    for v in sorted(int(x) for x in values):
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return [[a, b] for a, b in out]


def universal_steps(gap):
    """Per relation: the step specs conforming on every episode of that relation."""
    payload = load(gap)
    by_relation = defaultdict(list)
    for ep in payload["episodes"]:
        by_relation[ep["relation"]].append(ep)
    out = {}
    for relation, episodes in by_relation.items():
        conj = None
        for ep in episodes:
            h = episode_hits(ep)
            conj = h if conj is None else (conj & h)
        solutions = {}
        for bi, base in enumerate(S0_BASES):
            for oi, op in enumerate(S0_OPS):
                offsets = [int(o) for o in np.flatnonzero(conj[bi, oi])]
                if offsets:
                    solutions[f"{op}({base})"] = offsets
        out[relation] = {"n_episodes": len(episodes), "solutions": solutions}
    return out


def v2_expressivity():
    rows = {}
    for gap in ADMISSION_GAPS:
        rows[f"gap{gap}"] = universal_steps(gap)
    report = {"width": WIDTH, "s0_offsets": list(S0_OFFSETS), "per_gap": {}}
    for key, per_relation in rows.items():
        entry = {}
        for relation, data in per_relation.items():
            covered, solution_ranges = [], {}
            for spec, offsets in data["solutions"].items():
                solution_ranges[spec] = as_ranges(offsets)
                covered += [[spec, o] for o in sorted(set(offsets) & set(S0_OFFSETS))]
            entry[relation] = {
                "n_episodes": data["n_episodes"],
                "solution_ranges": solution_ranges,
                "s0_solutions": covered,
                "s0_expressible": bool(covered),
                "min_required_offset": min((min(v) for v in data["solutions"].values()),
                                           default=None)}
        report["per_gap"][key] = entry
    # the counterexample, stated as the two integers that decide it
    above1 = report["per_gap"]["gap1"]["above"]
    report["counterexample"] = {
        "configuration": "gap 1", "relation": "above",
        "min_required_offset": above1["min_required_offset"],
        "max_s0_offset": max(S0_OFFSETS),
        "s0_expressible": above1["s0_expressible"],
        "holds": (not above1["s0_expressible"]
                  and above1["min_required_offset"] > max(S0_OFFSETS))}
    report["s0_solves_gap0"] = all(
        report["per_gap"]["gap0"][r]["s0_expressible"] for r in RELATIONS)
    return report


def v3_widen(expressivity):
    reach = sorted(set(WIDEN_SOURCE) | set(S0_OFFSETS))
    need = expressivity["counterexample"]["min_required_offset"]
    above1 = expressivity["per_gap"]["gap1"]["above"]["solution_ranges"]
    required = set()
    for spec, ranges in above1.items():
        for a, b in ranges:
            required.update(range(a, b + 1))
    return {"widen_source": list(WIDEN_SOURCE),
            "reachable_offsets": reach,
            "max_reachable": max(reach),
            "min_required_offset": need,
            "reachable_and_required": sorted(set(reach) & required),
            "holds": max(reach) < need and not (set(reach) & required),
            "reading": ("no finite sequence of WIDEN edits over the scaffold's own "
                        "declared integer constants can reach the `gap 1` repair")}


def v8_pilot(sizes):
    """Wall clock of the exact 48-episode counter against the step pool's size."""
    sys.path.insert(0, str(FLAGSHIP))
    import cache          # noqa: E402
    import engine         # noqa: E402
    import family         # noqa: E402
    rows = []
    payload = cache.load(0)
    episodes = engine.Episodes(payload["episodes"])
    for n_offsets in sizes:
        pool = family.pools("schema", WIDTH)
        offsets = tuple(range(3, 3 * (n_offsets + 1), 3))
        step = [(op, b, o) for op in S0_OPS for b in S0_BASES for o in offsets]
        for slot in ("X1", "X2", "X3"):
            pool[slot] = list(step)
        started = time.perf_counter()
        tables = engine.Tables(episodes, pool)
        restriction = engine.Restriction(pool)
        total = engine.Counter(tables, restriction).total
        rows.append({"n_offsets": n_offsets, "n_step_candidates": len(step),
                     "conformers_48ep": str(total),
                     "seconds": time.perf_counter() - started,
                     "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                     / (1024 * 1024)})
        print(rows[-1], flush=True)
    return {"gap": 0, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="*", default=[5, 10, 20, 40])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    report = {"format": "schema-induction/phase0-1"}
    report["V2_expressivity"] = v2_expressivity()
    report["V3_widen_insufficiency"] = v3_widen(report["V2_expressivity"])
    if a.pilot:
        report["V8_cost_pilot"] = v8_pilot(a.sizes)
    (OUT / "phase0.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "V8_cost_pilot"},
                     indent=1)[:4000])
    print("V2 counterexample holds:", report["V2_expressivity"]["counterexample"]["holds"])
    print("V2 S0 solves gap 0:", report["V2_expressivity"]["s0_solves_gap0"])
    print("V3 widen-insufficiency holds:", report["V3_widen_insufficiency"]["holds"])


if __name__ == "__main__":
    main()
