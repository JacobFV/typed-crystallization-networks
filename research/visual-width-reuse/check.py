"""Resolve every pre-registered criterion against the raw JSON in `out/`.

Nothing here re-runs an arm.  It reads what the arms wrote and prints the verdict
for C1-C6 and F-a..F-f, so every claim in RESULTS.md traces to a file.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
STAGES = ("s0", "s1", "s2")
CERTIFIED = (24, 32)
HELD_OUT = (16, 40, 48)

# `research/visual-ladder/out/rung3_root.json`, the §33 record this must not move.
SECTION_33 = {"s0": {"space_size": 256, "conforming": 2, "certificate": "complete",
                     "distinct_functions": 1, "chosen": {"rg": 7, "same": 2}},
              "s1": {"space_size": 400, "conforming": 2, "certificate": "complete",
                     "chosen": {"back_a": 2, "back_b": 3, "corner": 1}},
              "s2": {"space_size": 25, "conforming": 1, "certificate": "unique",
                     "chosen": {"step_w": 2, "step_h": 3}},
              "parse": {"widgets_in_probe": 227, "rects_predicted": 227,
                        "parent_links_correct": 227, "parent_links_wrong": 0,
                        "trees_exact": 12, "screens_rects_exact": 12,
                        "corner_key_collisions": 0, "roots_predicted": 12}}


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def main():
    without = {w: load(f"enum_{w}") for w in CERTIFIED + HELD_OUT}
    with_ = {w: load(f"with_{w}") for w in HELD_OUT}
    construct, armf = load("construct"), load("armf")
    refusal, null = load("refusal"), load("null")

    rows, verdict = [], {}

    # ---- C1 / C3 / C6: the with-without table -----------------------------
    for w in HELD_OUT:
        a, b = without[w], with_[w]
        for stage in STAGES:
            sa, sb = a["stages"][stage], b["stages"][stage]
            rows.append({
                "resolution": w, "stage": stage,
                "observation_components": a["observation_components"],
                "space_size": sa["space_size"], "nodes": sa["nodes"],
                "without_programs": sa["walk"]["evaluated"],
                "without_rows": sa["walk"]["row_evaluations"],
                "without_node_units": sa["walk"]["node_units"],
                "without_seconds": sa["walk"]["seconds"],
                "without_prefix_seconds": sa["sweep"]["seconds"],
                "without_certificate": sa["walk"]["certificate"],
                "without_conforming": sa["walk"]["conforming"],
                "without_digest": sa["artifact_digest"],
                "without_held_accuracy": sa["held_accuracy"],
                "with_programs": sb["check"]["programs_evaluated"],
                "with_rows": sb["check"]["row_evaluations"],
                "with_node_units": sb["check"]["node_units"],
                "with_seconds": sb["check"]["seconds"],
                "with_certificate": "none",
                "with_digest": sb["artifact_digest"],
                "with_held_accuracy": sb["held_accuracy"],
                "same_program": sa["artifact_digest"] == sb["artifact_digest"],
                "conforms": sb["check"]["conforms"],
                "program_ratio": sa["walk"]["evaluated"] / sb["check"]["programs_evaluated"],
                "row_ratio": sa["walk"]["row_evaluations"] / sb["check"]["row_evaluations"],
                "node_ratio": sa["walk"]["node_units"] / sb["check"]["node_units"],
                "seconds_ratio": sa["walk"]["seconds"] / max(sb["check"]["seconds"], 1e-9),
                "best_constant": sa["best_constant"]["accuracy"],
                "whole_space_rows": sa.get("whole_space_held", {}).get("rows"),
                "whole_space_mean": sa.get("whole_space_held", {}).get("mean"),
                "whole_space_second_best": sa.get("whole_space_held", {}).get("second_best"),
                "whole_space_worst": sa.get("whole_space_held", {}).get("worst"),
                "chosen_accuracy_same_rows":
                    sa.get("whole_space_held", {}).get("chosen_accuracy_same_rows"),
                "best_constant_same_rows":
                    sa.get("whole_space_held", {}).get("best_constant_same_rows"),
            })

    pipeline = []
    for w in HELD_OUT:
        a, b = without[w]["pipeline"], with_[w]["pipeline"]
        pipeline.append({
            "resolution": w,
            "staged_space": a["staged_space"], "joint_space": a["joint_space"],
            "without_programs": a["programs_evaluated"], "with_programs": b["programs_evaluated"],
            "without_rows": a["row_evaluations"], "with_rows": b["row_evaluations"],
            "without_node_units": a["node_units"], "with_node_units": b["node_units"],
            "without_walk_seconds": a["walk_seconds"], "with_check_seconds": b["check_seconds"],
            "without_prefix_seconds": a["prefix_seconds"],
            "program_ratio": a["programs_evaluated"] / b["programs_evaluated"],
            "row_ratio": a["row_evaluations"] / b["row_evaluations"],
            "node_ratio": a["node_units"] / b["node_units"],
            "walk_seconds_ratio": a["walk_seconds"] / max(b["check_seconds"], 1e-9),
            "prefix_seconds_ratio": a["prefix_seconds"] / max(b["check_seconds"], 1e-9),
            "parse_without": without[w]["parse"]["totals"],
            "parse_with": with_[w]["parse"]["totals"],
        })

    verdict["C1 instantiation reproduces the enumerated program"] = all(
        r["same_program"] and r["conforms"] for r in rows)
    verdict["C2 the parse is component-for-component identical"] = all(
        {k: v for k, v in p["parse_without"].items() if k != "seconds"} ==
        {k: v for k, v in p["parse_with"].items() if k != "seconds"} for p in pipeline)
    # The with-arm's program *is* the without-arm's program (C1), so the
    # whole-space comparison is made on the rows the whole-space sweep scored.
    verdict["C3 beats constant / random / whole-space mean"] = all(
        r["with_held_accuracy"] > r["best_constant"]
        and (r["whole_space_mean"] is None or
             (r["chosen_accuracy_same_rows"] > r["whole_space_mean"]
              and r["chosen_accuracy_same_rows"] > r["whole_space_second_best"]
              and r["chosen_accuracy_same_rows"] > r["best_constant_same_rows"]))
        for r in rows)
    # C4 is resolved per wrong schema, because the two payloads are wrong in
    # different ways and a single boolean would hide which.
    fo = armf["schemas"]["frozen_offsets"]["per_resolution"]
    fs = armf["schemas"]["frozen_span"]["per_resolution"]
    verdict["C4a frozen offsets: 0 conforming at every resolution but 32"] = all(
        fo[str(w)][st]["walk"]["conforming"] == 0 and fo[str(w)][st]["walk"]["exhausted"]
        for st in ("s1", "s2") for w in (16, 24, 40, 48))
    verdict["C4a frozen offsets: refused by the format at one and at two widths"] = all(
        not armf["format"][k]["admitted"] for k in armf["format"] if k.startswith("frozen_offsets"))
    verdict["C4b frozen span: 0 conforming above the frozen value (40, 48)"] = all(
        fs[str(w)]["s2"]["walk"]["conforming"] == 0 and fs[str(w)]["s2"]["walk"]["exhausted"]
        for w in (40, 48))
    verdict["C4b frozen span: refused by the format at one and at two widths"] = all(
        not armf["format"][k]["admitted"] for k in armf["format"] if k.startswith("frozen_span"))
    verdict["C4c arm F is bit-identical to the right schema at 32"] = all(
        armf["schemas"][s]["per_resolution"]["32"][st].get("bit_identical_to_right")
        for s in armf["schemas"] for st in ("s1", "s2"))
    s33 = {stage: without[32]["stages"][stage] for stage in STAGES}
    verdict["C5 section 33 does not move"] = (
        all(s33[st]["space_size"] == SECTION_33[st]["space_size"]
            and s33[st]["walk"]["conforming"] == SECTION_33[st]["conforming"]
            and s33[st]["walk"]["certificate"] == SECTION_33[st]["certificate"]
            for st in STAGES)
        and s33["s0"]["distinct_functions"] == SECTION_33["s0"]["distinct_functions"]
        and all(s33[st]["chosen_free"] == SECTION_33[st]["chosen"] for st in STAGES)
        and all(without[32]["parse"]["totals"][k] == v
                for k, v in SECTION_33["parse"].items()))
    verdict["the selection vector is identical at all five resolutions"] = all(
        without[w]["stages"][st]["chosen_free"] == SECTION_33[st]["chosen"]
        for w in CERTIFIED + HELD_OUT for st in STAGES)
    verdict["F-a a resolution-32 artifact is refused elsewhere"] = (
        refusal["offered"]["32"]["accepted"]
        and all(not refusal["offered"][str(w)]["accepted"] for w in HELD_OUT))
    verdict["F-f the derived span is width-safe"] = all(
        without[w]["stages"]["s2"]["walk"]["conforming"] >= 1 for w in CERTIFIED + HELD_OUT)
    verdict["class admitted at 2 widths, vectors agree"] = (
        construct["all_admitted"] and not construct["vector_disagreements"])

    # ---- C6: does the saving exceed the space size? -----------------------
    c6 = []
    for p in pipeline:
        c6.append({"resolution": p["resolution"],
                   "largest_stage_space": max(without[p["resolution"]]["stages"][s]["space_size"]
                                              for s in STAGES),
                   "staged_space": p["staged_space"], "joint_space": p["joint_space"],
                   "program_ratio": p["program_ratio"], "row_ratio": p["row_ratio"],
                   "node_ratio": p["node_ratio"],
                   "walk_seconds_ratio": p["walk_seconds_ratio"],
                   "prefix_seconds_ratio": p["prefix_seconds_ratio"],
                   "row_ratio_exceeds_largest_space":
                       p["row_ratio"] > max(without[p["resolution"]]["stages"][s]["space_size"]
                                            for s in STAGES),
                   "prefix_seconds_ratio_exceeds_largest_space":
                       p["prefix_seconds_ratio"] >
                       max(without[p["resolution"]]["stages"][s]["space_size"] for s in STAGES)})
    verdict["C6 saving exceeds the largest stage space (prefix wall clock)"] = all(
        r["prefix_seconds_ratio_exceeds_largest_space"] for r in c6)
    verdict["C6 saving exceeds the largest stage space (row evaluations)"] = all(
        r["row_ratio_exceeds_largest_space"] for r in c6)

    print("== per stage, held-out resolutions ==")
    for r in rows:
        print(f"  res {r['resolution']:2d} {r['stage']}  space {r['space_size']:4d}  "
              f"without {r['without_programs']:4d} prog / {r['without_rows']:7d} rows / "
              f"{r['without_seconds']:7.1f}s walk / {r['without_prefix_seconds']:7.1f}s prefix "
              f"-> {r['without_certificate']:11s} | with 1 prog / {r['with_rows']:5d} rows / "
              f"{r['with_seconds']:5.2f}s -> none | same_program={r['same_program']} "
              f"acc {r['with_held_accuracy']:.4f} vs const {r['best_constant']:.4f} "
              f"space-mean {r['whole_space_mean']}")
    print("\n== pipeline ==")
    for p in pipeline:
        print(f"  res {p['resolution']:2d}  staged space {p['staged_space']}  joint "
              f"{p['joint_space']}  programs {p['without_programs']}->{p['with_programs']} "
              f"(x{p['program_ratio']:.0f})  rows x{p['row_ratio']:.0f}  nodes "
              f"x{p['node_ratio']:.1f}  walk s x{p['walk_seconds_ratio']:.0f}  prefix s "
              f"x{p['prefix_seconds_ratio']:.0f}")
        print(f"       parse without {json.dumps(p['parse_without'], sort_keys=True)}")
        print(f"       parse with    {json.dumps(p['parse_with'], sort_keys=True)}")
    print("\n== C6, the falsification section 55 flagged against itself ==")
    for r in c6:
        print(f"  res {r['resolution']:2d}  largest stage space {r['largest_stage_space']}  "
              f"rows x{r['row_ratio']:.1f} (exceeds: {r['row_ratio_exceeds_largest_space']})  "
              f"prefix wall x{r['prefix_seconds_ratio']:.1f} "
              f"(exceeds: {r['prefix_seconds_ratio_exceeds_largest_space']})")
    print("\n== arm F ==")
    for name, rec in armf["schemas"].items():
        for w in sorted(rec["per_resolution"], key=int):
            for st in ("s1", "s2"):
                d = rec["per_resolution"][w][st]
                print(f"  {name:15s} res {w:>2s} {st}: conforming {d['walk']['conforming']} "
                      f"{d['walk']['certificate']:11s} stored-vector conforms "
                      f"{d['stored_vector_conforms']} held acc "
                      f"{d['stored_vector_held_accuracy']:.4f} "
                      f"bit-identical-to-right {d.get('bit_identical_to_right')}")
    for k, v in armf["format"].items():
        print(f"  format {k}: admitted={v['admitted']} "
              f"{v.get('rule', '')[:110]}")
    print("\n== null control (section 59 instrument floor) ==")
    print(f"  median {null['median']:.4f}s spread {null['spread_fraction'] * 100:.2f}%")
    print("\n== criteria ==")
    for k, v in verdict.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    (OUT / "check.json").write_text(json.dumps(
        {"rows": rows, "pipeline": pipeline, "c6": c6, "verdict": verdict,
         "null": null, "refusal": refusal}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
