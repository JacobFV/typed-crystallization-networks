"""The two tables `RESULTS.md` reports, re-derived from raw JSON.

Nothing here recomputes anything: it reads `out/arms_enum.json`,
`out/ranked_<band>.json` and `out/heldout.json` and joins them, so every
number in the write-up can be traced to a file on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

OBJECTIVES = ("O1", "O2", "O3", "O4", "B1", "B2", "WIN")
ARMS = ("arm1_none", "arm2_syntactic", "arm2s_semantic", "arm3_authored",
        "arm4_wrong_authored", "arm4b_wrong_mined", "arm4s_runnerup",
        "arm4s_matched", "arm4s_offfamily")


def arm_table(band):
    rows = json.loads((OUT / "arms_enum.json").read_text())
    mods = json.loads((OUT / f"modules_{band}.json").read_text())
    print(f"### arm table, band {band} -- task L1_w4_bdae_xor_c")
    print("| arm | module | space | evaluated | exhausted | certificate | conforming |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        r = next(x for x in rows if x["band"] == band and x["arm"] == arm
                 and x["task"] == "L1_w4_bdae_xor_c")
        dig = mods.get(arm, {}).get("digest")
        print(f"| `{arm}` | {dig[:12] if dig else ('hand' if 'authored' in arm else '--')} "
              f"| {r['space_size']:,} | {r['evaluated']:,} | {str(r['exhausted']).lower()} "
              f"| `{r['certificate']}` | **{r['conforming']}** |")
    print()
    print("| arm | H_par5 | H_x4 |")
    print("|---|---|---|")
    for arm in ARMS:
        c = {t: next(x for x in rows if x["band"] == band and x["arm"] == arm
                     and x["task"] == t)["conforming"] for t in ("H_par5", "H_x4")}
        print(f"| `{arm}` | {c['H_par5']} | {c['H_x4']} |")
    print()
    digests = {a: mods[a]["digest"] for a in ARMS
               if a in mods and not mods[a].get("no_candidate")}
    print(f"distinct mined classes among the arms: "
          f"**{len(set(digests.values()))}** of {len(digests)} mined arms")
    for a, d in sorted(digests.items()):
        print(f"  {a:22s} {d}")
    print()


def loo_table(band):
    data = json.loads((OUT / "heldout.json").read_text())
    plan = [p for p in data["plan"] if p["band"] == band]
    rows = [r for r in data["rows"] if r["band"] == band]
    loo = sorted({p["held_out"] for p in plan})
    print(f"### leave-one-out transfer, band {band}")
    print("| objective | " + " | ".join(t.split("_")[0] for t in loo)
          + " | helps, of 5 | window is rank 1 |")
    print("|---" * (len(loo) + 3) + "|")
    for oid in OBJECTIVES:
        cells, helps, wins = [], 0, 0
        for t in loo:
            p = next((q for q in plan if q["objective"] == oid
                      and q["held_out"] == t), None)
            if p is None:
                cells.append("--")
                continue
            r = next(x for x in rows if x["task"] == t and x["spec"] == p["spec"])
            assert r["exhausted"] and r["evaluated"] == r["space_size"], r
            cells.append(str(r["conforming"]))
            helps += int(r["conforming"] > 0)
            wins += int(bool(p["is_window"]))
        print(f"| {oid} | " + " | ".join(cells) + f" | **{helps} of 5** | {wins} of 5 |")
    for label, spec in (("floor `none`", "none"), ("ceiling hand-`W4`", "hand:w4"),
                        ("wrong hand-`X4`", "hand:x4")):
        cells, helps = [], 0
        for t in loo:
            r = next(x for x in rows if x["task"] == t and x["spec"] == spec)
            assert r["exhausted"] and r["evaluated"] == r["space_size"], r
            cells.append(str(r["conforming"]))
            helps += int(r["conforming"] > 0)
        print(f"| {label} | " + " | ".join(cells) + f" | **{helps} of 5** | -- |")
    print()
    print("| spec | H_par5 | H_x4 |")
    print("|---|---|---|")
    specs = sorted({r["spec"] for r in rows})
    for s in specs:
        c = {}
        for t in ("H_par5", "H_x4"):
            m = [x for x in rows if x["task"] == t and x["spec"] == s]
            c[t] = m[0]["conforming"] if m else None
        print(f"| `{s}` | {c['H_par5']} | {c['H_x4']} |")
    print()


def main():
    for band in (sys.argv[1:] or ["C-trace", "C-minall"]):
        arm_table(band)
        loo_table(band)


if __name__ == "__main__":
    main()
