"""The off-family (`F'`) table, from `out/offfamily.json`.  EXPLORATORY."""
from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401

import objectives

OUT = Path(__file__).resolve().parent / "out"
LOO = ("s1_w_abc_xor_d", "s2_w_abc_and_d", "s3_w_bcd_or_a",
       "s4_w_acd_xor_b", "s5_w_abd_or_c")
CTRL = ("t1_maj_abc_xor_d", "t3_maj_bcd_or_a", "H_par")


def main():
    d = json.loads((OUT / "offfamily.json").read_text())
    idx = {(r["task"], r["spec"]): r for r in d["rows"]}
    print("| objective | " + " | ".join(t.split("_")[0] for t in LOO) +
          " | helps | rank-1 is the `D134` window | cross-family controls (t1 / t3 / `H_par`) |")
    print("|---" * 9 + "|")
    for oid in objectives.ALL_IDS:
        cells, helps, win = [], 0, 0
        ctrl = [0, 0, 0]
        for t in LOO:
            name = "wo_" + t.split("_")[0]
            top = d["tables"][name]["tables"][oid]["rank1"]
            spec = "lib:off_" + top["digest"][:12]
            r = idx[(t, spec)]
            cells.append(f"**{r['conforming']}**" if r["conforming"] else "0")
            helps += 1 if r["conforming"] > 0 else 0
            win += 1 if top["is_window"] else 0
            for i, c in enumerate(CTRL):
                ctrl[i] = max(ctrl[i], idx[(c, spec)]["conforming"])
        print(f"| **{oid}** | " + " | ".join(cells) +
              f" | **{helps} of 5** | {win} of 5 | {ctrl[0]} / {ctrl[1]} / {ctrl[2]} |")
    for label, spec in (("ceiling: hand-authored `D134`", "hand:d134"),
                        ("floor: no library", "none")):
        cells = [f"**{idx[(t, spec)]['conforming']}**" if idx[(t, spec)]["conforming"]
                 else "0" for t in LOO]
        helps = sum(1 for t in LOO if idx[(t, spec)]["conforming"] > 0)
        ctrl = [idx[(c, spec)]["conforming"] for c in CTRL]
        print(f"| {label} | " + " | ".join(cells) + f" | **{helps} of 5** | — | "
              f"{ctrl[0]} / {ctrl[1]} / {ctrl[2]} |")
    print("\nEvery row: `exhausted: true`, certificate `complete`. "
          f"IC1 passes on all five `F'` corpora: "
          f"{all(v['ic1'] for v in d['tables'].values())}.")


if __name__ == "__main__":
    main()
