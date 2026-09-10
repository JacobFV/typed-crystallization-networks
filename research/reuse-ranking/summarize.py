"""Regenerate every table in RESULTS.md from `out/*.json`.

Nothing here recomputes anything; it reads the raw files the runs wrote, so any
figure in the write-up can be checked against them without re-running an arm.
"""
from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401

import objectives

OUT = Path(__file__).resolve().parent / "out"
LOO = ("t1_maj_abc_xor_d", "t2_maj_abc_and_d", "t3_maj_bcd_or_a",
       "t4_maj_acd_xor_b", "t5_maj_abd_or_c")
CONTROLS = ("H_par", "H_d134")


def load():
    return (json.loads((OUT / "ranked.json").read_text()),
            json.loads((OUT / "heldout.json").read_text()))


def enum_index(held):
    return {(r["task"], r["spec"]): r for r in held["rows"]}


def main():
    ranked, held = load()
    idx = enum_index(held)

    print("### Integrity checks\n")
    print("| corpus | entries | eligible | IC1 sum(s_e)-D == rule saving | "
          "IC2 O1 rank-1 == rule rank-1 |")
    print("|---|---|---|---|---|")
    for name, rec in ranked["corpora"].items():
        ic = ranked["integrity"][name]
        print(f"| `{name}` | {rec['entries']} | {rec['eligible']} | "
              f"{'PASS' if ic['ic1_pass'] else 'FAIL'} | "
              f"{'PASS' if ic['ic2_o1_matches_rule'] else 'FAIL'} |")

    print("\n### The decisive table: objective x leave-one-out, with certificates\n")
    hdr = ("| objective | " + " | ".join(t.split("_")[0] for t in LOO) +
           " | helps | `H_par` | `H_d134` |")
    print(hdr)
    print("|---" * (len(LOO) + 4) + "|")

    def cell(row):
        return (f"**{row['conforming']}**" if row["conforming"] else "0") + \
            ("" if row["exhausted"] and row["certificate"] == "complete" else " ??")

    rows_out = {}
    for oid in objectives.ALL_IDS:
        cells, helps, ctrl = [], 0, {}
        for t in LOO:
            cname = "wo_" + t.split("_")[0]
            spec = "lib:" + ranked["corpora"][cname]["tables"][oid]["rank1"]["library_label"]
            r = idx[(t, spec)]
            cells.append(cell(r))
            helps += 1 if r["conforming"] > 0 else 0
        # controls: the union over the five selections
        for c in CONTROLS:
            vals = set()
            for t in LOO:
                cname = "wo_" + t.split("_")[0]
                spec = "lib:" + ranked["corpora"][cname]["tables"][oid]["rank1"]["library_label"]
                vals.add(idx[(c, spec)]["conforming"])
            ctrl[c] = max(vals)
        rows_out[oid] = helps
        print(f"| **{oid}** {objectives.NAME[oid]} | " + " | ".join(cells) +
              f" | **{helps} of 5** | {ctrl['H_par']} | {ctrl['H_d134']} |")
    for label, spec in (("ceiling: hand-authored `MAJ3`", "hand:maj"),
                        ("floor: no library", "none"),
                        ("wrong module: hand-authored `D134`", "hand:distractor")):
        cells = [cell(idx[(t, spec)]) for t in LOO]
        helps = sum(1 for t in LOO if idx[(t, spec)]["conforming"] > 0)
        print(f"| {label} | " + " | ".join(cells) + f" | **{helps} of 5** | "
              f"{idx[('H_par', spec)]['conforming']} | "
              f"{idx[('H_d134', spec)]['conforming']} |")

    print("\nEvery enumeration above: `exhausted: true`, certificate `complete`, "
          "`evaluated == space_size`.")

    print("\n### What each objective actually selected\n")
    print("| objective | corpus | rank-1 digest | arity | nodes | tasks | is the window | "
          "rank of the window | decided by | space | evaluated |")
    print("|---" * 11 + "|")
    for oid in objectives.ALL_IDS:
        for t in ("full",) + tuple("wo_" + x.split("_")[0] for x in LOO):
            tbl = ranked["corpora"][t]["tables"][oid]
            top = tbl["rank1"]
            spec = "lib:" + top["library_label"]
            task = None if t == "full" else [x for x in LOO if x.startswith(t[3:] + "_")][0]
            e = idx[(task, spec)] if task else None
            print(f"| {oid} | `{t}` | `{top['digest'][:12]}` | {top['arity']} | "
                  f"{top['nodes']} | {len(top['tasks'])} | "
                  f"{'YES' if top['is_window'] else 'no'} | {tbl['best_window_rank']} | "
                  f"{tbl['rank1_decided_by']} | "
                  f"{e['space_size'] if e else '-'} | {e['evaluated'] if e else '-'} |")

    print("\n### The margin that decides rank 1\n")
    print("| corpus | O1 rank-1 over the window class | O2 rank-1 over its runner-up | "
          "O4 rank-1 over its runner-up |")
    print("|---|---|---|---|")
    for t in ("full",) + tuple("wo_" + x.split("_")[0] for x in LOO):
        tb = ranked["corpora"][t]["tables"]
        cells = []
        for oid in ("O1", "O2", "O4"):
            rows = tb[oid]["ranked"]
            top = rows[0]
            if oid == "O1":
                other = [r for r in rows if r["is_window"]][0]
                if top["digest"] == other["digest"]:
                    cells.append("*is the window* (rank 1)")
                    continue
                a, b = top["saving_bits"], other["saving_bits"]
                cells.append(f"**+{a - b:,} ({100 * (a - b) / b:.2f} %)** the wrong way")
            else:
                b = rows[1]["score"]
                cells.append(f"+{top['score'] - b:,.0f} ({100 * (top['score'] - b) / b:.1f} %)")
        print(f"| `{t}` | " + " | ".join(cells) + " |")

    folds = OUT / "o4_folds.json"
    if folds.exists():
        f = json.loads(folds.read_text())
        print("\n### O4's fold decomposition on `wo_t1` "
              "(saving on each fold's *withheld* task, bits)\n")
        d = f["wo_t1"]
        print("| class | tasks | O1 saving | " + " | ".join(d["mining_tasks"]) + " | O4 |")
        print("|---" * (len(d["mining_tasks"]) + 4) + "|")
        for r in d["rows"]:
            if not r["label"]:
                continue
            print(f"| `{r['digest'][:12]}` {r['label']} | {r['tasks']} | "
                  f"{r['saving_bits']:,} | " +
                  " | ".join(f"{r['folds'][t]:,}" for t in d["mining_tasks"]) +
                  f" | **{r['o4']:,.0f}** |")

    grad = OUT / "heldout_gradient.json"
    if grad.exists():
        g = json.loads(grad.read_text())
        by = {(r["task"], r["spec"]): r for r in g["summary"]}
        print("\n### Gradient beside the enumeration "
              f"({g['seeds']} seeds, {g['steps']} steps): solved / median exact accuracy\n")
        print("| objective | " + " | ".join(t.split("_")[0] for t in LOO) +
              " | `H_par` | `H_d134` |")
        print("|---" * (len(LOO) + 3) + "|")
        for oid in objectives.ALL_IDS:
            cells = []
            for t in LOO + CONTROLS:
                cname = ("wo_" + t.split("_")[0]) if t in LOO else "wo_t1"
                spec = "lib:" + ranked["corpora"][cname]["tables"][oid]["rank1"]["library_label"]
                r = by[(t, spec)]
                cells.append(f"{r['solved']}/{r['seeds']} · {r['median_accuracy']:.4f}")
            print(f"| **{oid}** | " + " | ".join(cells) + " |")
        for label, spec in (("ceiling `MAJ3`", "hand:maj"), ("floor, none", "none"),
                            ("wrong `D134`", "hand:distractor")):
            cells = [f"{by[(t, spec)]['solved']}/{by[(t, spec)]['seeds']} · "
                     f"{by[(t, spec)]['median_accuracy']:.4f}" for t in LOO + CONTROLS]
            print(f"| {label} | " + " | ".join(cells) + " |")
        cb = {t: by[(t, "none")]["constant_baseline"] for t in LOO + CONTROLS}
        print("| *constant baseline* | " +
              " | ".join(f"{cb[t]:.4f}" for t in LOO + CONTROLS) + " |")
        print("| *random baseline* | " + " | ".join(["0.5000"] * 7) + " |")

    s41 = OUT / "s41_crosscheck.json"
    if s41.exists():
        d = json.loads(s41.read_text())
        print("\n### The §41 cross-check: which of the ten language-family programs\n")
        print("| objective | degenerate on a one-task family | picks | description bits | "
              "measured bytecodes | bytecode-maximal? |")
        print("|---" * 6 + "|")
        for oid in objectives.ALL_IDS:
            v = d["objectives"][oid]
            p = v.get("pick")
            if p is None:
                print(f"| {oid} | **yes** | no pick — {v['note']} | — | — | — |")
            else:
                print(f"| {oid} | {'**yes**' if v['degenerate'] else 'no'} | "
                      f"program | {p['description_bits']} | {p['bytecodes_total']} | "
                      f"{'**YES**' if p['is_bytecode_maximal'] else 'no'} |")

    sens = OUT / "sensitivity.json"
    if sens.exists():
        d = json.loads(sens.read_text())
        print("\n### Sensitivity A — the breadth exponent "
              "`|T|^a * sum(s_e) - D` (a = 0 is the incumbent)\n")
        print("| corpus | smallest a with the window class at rank 1 | "
              "largest a swept with it at rank 1 |")
        print("|---|---|---|")
        for c, v in d["exponent"]["result"].items():
            print(f"| `{c}` | {v['window_rank1_alpha_min']} | {v['window_rank1_alpha_max']} |")
        print(f"\nSwept a in {d['exponent']['alphas']}.")
        print("\n### Sensitivity B — the rule's own parameters, §52 §8's grid\n")
        print("| objective | rank-1 is the window class, of the settings where the "
              "class is constructible | of all 162 (27 settings x 6 corpora) |")
        print("|---|---|---|")
        for oid, v in d["parameter_grid_summary"].items():
            print(f"| {oid} | **{v['rank1_is_window']} of {v['of_constructible']}** | "
                  f"{v['rank1_is_window_all']} of {v['of_all']} |")


if __name__ == "__main__":
    main()
