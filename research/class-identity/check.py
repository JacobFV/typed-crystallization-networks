"""Resolve every pre-registered criterion against the raw JSON in `out/`.

Nothing here re-runs an arm. It reads what the arms wrote and prints the verdict
for C1-C7 and F-a..F-f, so a claim in RESULTS.md can be traced to a file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "depth-encoding"))
sys.path.insert(0, str(ROOT))

OUT = HERE / "out"
WIDTHS = ("5", "7", "12")


def load(name):
    return json.loads((OUT / name).read_text())


def digest_of(kind, width, selections):
    """Rebuild the schema and freeze the vector -- the digest the enumeration means."""
    import run as DEP
    from tcn.search import frozen_selection
    _, program, registry = DEP.build(kind, int(width))
    full = {n.name: 0 for n in program.nodes}
    full.update(selections)
    return frozen_selection(program, full, registry).digest


def main():
    without, with_, base = load("without.json"), load("with.json"), load("baselines.json")
    armf, construct = load("armf.json"), load("construct.json")
    compat, inherit = load("q2_compat.json"), load("q2_inherit.json")
    partition, members = load("q2_partition.json"), load("q2_members.json")
    cross = load("crosscheck.json")
    rows, verdict = [], {}

    for w in WIDTHS:
        a2, a3 = without[w], with_["per_width"][w]
        sel = a2["conforming_selections"][0]
        enum_digest = digest_of("interpreter", w, sel)
        b = base["held_out"][w]
        rows.append({
            "width": int(w), "program_fields": 3 * int(w),
            "a2_programs": a2["evaluated"], "a2_episodes": a2["episodes"],
            "a2_steps": a2["steps"], "a2_seconds": a2["seconds"],
            "a2_certificate": a2["certificate"], "a2_conforming": a2["conforming"],
            "a2_best_return": a2["best_return"],
            "a2_second_best": a2["second_best_return"],
            "a2_whole_space_mean": a2["whole_space_mean"],
            "a2_selection": sel, "a2_digest": enum_digest,
            "a3_programs": a3["programs_evaluated"], "a3_episodes": a3["episodes"],
            "a3_steps": a3["steps"], "a3_seconds": a3["seconds"],
            "a3_certificate": a3["certificate"], "a3_mean_return": a3["mean_return"],
            "a3_digest": a3["artifact_digest"], "a3_nodes": a3["nodes"],
            "same_program": enum_digest == a3["artifact_digest"],
            "episode_ratio": a2["episodes"] / a3["episodes"],
            "seconds_ratio": a2["seconds"] / a3["seconds"],
            "best_constant": b["best_constant"], "uniform_random": b["uniform_random"]["mean"],
            "always_true": b["always_true"]["mean"], "always_false": b["always_false"]["mean"],
        })

    verdict["C1 same program, conforming"] = all(
        r["same_program"] and r["a3_mean_return"] >= 4.0 for r in rows)
    verdict["C2 episode cost < 1/100"] = all(r["episode_ratio"] > 100 for r in rows)
    verdict["C3 beats constant/random/space mean"] = all(
        r["a3_mean_return"] > max(r["best_constant"], r["uniform_random"],
                                  r["a2_whole_space_mean"]) for r in rows)
    verdict["C4 F1 and F2 both refused"] = (
        not armf["F1"]["admitted"] and not armf["F2"]["admitted"])
    verdict["C5 unsound format is wrong everywhere"] = (
        all(not armf["F3"]["per_width"][w]["conforms"] for w in WIDTHS)
        and all(armf["F4"][w]["conforming"] == 0 and armf["F4"][w]["certificate"] == "complete"
                for w in WIDTHS))
    verdict["C6 extra field disturbs nothing"] = (
        compat["all_identical"] and compat["stamped"]["verify_all_ok"]
        and not compat["stamped"]["errors"])
    verdict["crosscheck: shipped enumerator agrees"] = cross["all_agree"]
    verdict["class admitted at 2 widths"] = construct["admitted"] and construct["vectors_agree"]

    print("== Q1, per held-out width ==")
    for r in rows:
        print(f"  d={r['width']:2d} w={r['program_fields']:2d}  "
              f"without: {r['a2_programs']:4d} programs / {r['a2_episodes']:6d} ep / "
              f"{r['a2_seconds']:7.1f}s -> {r['a2_certificate']:8s} best {r['a2_best_return']}  |  "
              f"with: {r['a3_programs']} program / {r['a3_episodes']:3d} ep / "
              f"{r['a3_seconds']:.2f}s -> {r['a3_mean_return']}  |  "
              f"same_program={r['same_program']}  x{r['episode_ratio']:.0f} ep  "
              f"x{r['seconds_ratio']:.0f} s")
    print("\n== arm F ==")
    print("  F1 refusal:", armf["F1"].get("rule"))
    print("  F2 refusal:", armf["F2"].get("rule"),
          "| stored vector at width 2 scores", armf["F2"]["stored_vector_mean_return_at_width_2"])
    for w in WIDTHS:
        print(f"  F3 unsound-format result at d={w}: {armf['F3']['per_width'][w]['mean_return']} "
              f"(best constant {base['held_out'][w]['best_constant']}) | "
              f"F4 whole wrong space: {armf['F4'][w]['conforming']} conforming, "
              f"{armf['F4'][w]['certificate']}, best {armf['F4'][w]['best_return']}")
    print("\n== Q2 ==")
    print("  manifest entries", partition["manifest_entries"], "distinct digests",
          partition["distinct_digests"], "distinct classes", partition["distinct_classes"])
    print("  multi-member classes:", partition["classes_with_multiple_members"])
    print("  majority class pool:", members["recovered_members"], "members; nodes",
          members["node_counts"], "exec", members["execution_costs"], "dbits",
          members["description_bits"])
    print("  inherit: digest separates", inherit["digest_separates"],
          "class unifies", inherit["class_unifies"], "preferred", inherit["preferred"])
    print("  compat: identical to control", compat["all_identical"],
          "| extra keys survive a save:", compat["extra_keys_survive_save"])
    print("\n== criteria ==")
    for k, v in verdict.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    (OUT / "check.json").write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
