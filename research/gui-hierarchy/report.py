"""Render every table in RESULTS.md directly from out/*.json.

No number in the report is transcribed by hand.
"""
from __future__ import annotations

import json
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parent / "out"


def load(name):
    path = OUT / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def f(x, places=4):
    return "-" if x is None else f"{x:.{places}f}"


def bounds():
    data = load("bounds")
    if not data:
        return
    print("### R1 — widget edge from a two-pixel neighbourhood\n")
    table(["setting", "axis", "records", "majority", "oracle ceiling", "oracle advantage",
           "transfer", "unseen keys"],
          [[k.split(" | ")[0], k.split(" | ")[1], v["eval_records"], f(v["majority_baseline"]),
            f(v["oracle_accuracy"]), f(v["oracle_advantage"]), f(v["transfer_accuracy"]),
            f(v["unseen_key_fraction"], 3)]
           for k, v in data["rung1_edge"].items()])
    if "rung1_edge_pattern" in data:
        print("### R1b — the same target from the three `eq` bits the algebra can write\n")
        table(["setting", "axis", "majority", "pattern ceiling", "advantage", "transfer"],
              [[k.split(" | ")[0], k.split(" | ")[1], f(v["majority_baseline"]),
                f(v["oracle_accuracy"]), f(v["oracle_advantage"]), f(v["transfer_accuracy"])]
               for k, v in data["rung1_edge_pattern"].items()])
    print("### R2 — glyph from its bounding box\n")
    table(["context", "glyphs", "distinct codes seen", "majority", "oracle ceiling", "transfer",
           "unseen keys"],
          [[k, v["eval_records"], v["distinct_eval_keys"], f(v["majority_baseline"]),
            f(v["oracle_accuracy"]), f(v["transfer_accuracy"]), f(v["unseen_key_fraction"], 3)]
           for k, v in data["rung2_glyph"].items()])
    print("### R3 — widget kind from rendered appearance\n")
    table(["colour mode", "context", "widgets", "majority", "oracle ceiling", "oracle advantage",
           "transfer"],
          [[k.split(" | ")[0].replace("colour_mode=", ""), k.split(" | ")[1], v["eval_records"],
            f(v["majority_baseline"]), f(v["oracle_accuracy"]), f(v["oracle_advantage"]),
            f(v["transfer_accuracy"])]
           for k, v in data["rung3_kind"].items()])
    print("### R4 — parent from geometry alone\n")
    table(["screen", "non-root widgets", "smallest containing rectangle is the parent", "ties"],
          [[k, v["widgets"], f(v["smallest_container_correct"]), v["ties"]]
           for k, v in data["rung4_parent"].items()])


def dial():
    data = load("dial")
    if not data:
        return
    print("### The difficulty dial, measured on four axes\n")
    table(["dial setting", "widgets (mean)", "tree depth", "colours on screen", "obs bytes",
           "positions", "rung-1 majority", "rung-1 ceiling", "rung-1 space", "conforming"],
          [[k, f(v["widgets_mean"], 1), v["tree_depth_max"], f(v["distinct_colours_mean"], 1),
            v["observation_bytes"], v["positions_per_screen"], f(v["majority_baseline"]),
            f(v["oracle_ceiling"]), v["rung1_space"], v["rung1_conforming"]]
           for k, v in data.items()])


def arm_rows(data, key, label):
    arm = data.get(key)
    if not arm:
        return []
    enumeration = arm.get("enumerate_fit") or {}
    conforming = arm["conforming"]
    filtered = arm.get("validation_filtered", {})
    pick = arm.get("lexicographic_pick", {})
    noise = arm.get("gradient_noise", {}).get("summary", {})
    zero = arm.get("gradient_zero_init", {}).get("summary", {})
    return [[label, arm["space_size"], conforming["evaluated"], conforming["exhausted"],
             conforming["count"], conforming.get("distinct_functions"),
             "yes" if conforming["unique"] else "no",
             f(arm["random_control"]["density"]),
             f(conforming["seconds"], 1), conforming["node_evaluations"],
             pick.get("held_max_error"), filtered.get("survivors"),
             filtered.get("distinct_functions"),
             max(filtered.get("held_max_error") or [None], key=lambda v: -1 if v is None else v),
             f"{noise.get('successes')}/{noise.get('n')}",
             f"{noise.get('held_exact')}/{noise.get('n')}",
             f"{zero.get('successes')}/{zero.get('n')}",
             f(noise.get("median_seconds"), 1)]]


def rung1_data():
    """One file if the arms ran together, otherwise merge the per-arm files."""
    merged = load("rung1") or {}
    for arm in ("narrow", "wide", "free"):
        part = load(f"rung1_{arm}")
        if part:
            merged = {**part, **merged} if arm == "narrow" else {**merged, **part}
    return merged or None


def rung1():
    data = rung1_data()
    if not data:
        return
    print("### Rung one, three arms over the same target\n")
    rows = []
    rows += arm_rows(data, "narrow", "A. narrow offsets, channels paired (H2+H3 supplied)")
    rows += arm_rows(data, "wide_offsets_ablation", "B. H2 ablated: every offset 1..3W+3")
    rows += arm_rows(data, "free_operand_ablation", "C. H3 ablated: operand binding searched")
    table(["arm", "space", "evaluated", "exhausted", "conforming", "distinct functions",
           "unique", "random density", "sweep s", "node evals", "lex pick held-out err",
           "validation survivors", "survivor functions", "survivor held-out err",
           "gradient ok (noise .5)", "gradient held-exact", "gradient ok (noise 0)",
           "gradient median s"], rows)
    stage_c = data.get("stage_c")
    if stage_c:
        print("### Stage C — rung one frozen, registered, and CHOSEN by a second program\n")
        table(["candidates offered", "space", "conforming", "unique", "edge module selected",
               "caller nodes", "positions", "module execution cost", "module description bits",
               "caller description bits"],
              [[stage_c["candidates"], stage_c["space_size"], stage_c["conforming"],
                "yes" if stage_c["unique"] else "no",
                "yes" if stage_c["selected_is_edge_module"] else "no",
                stage_c["caller_nodes"], stage_c["positions"],
                stage_c["module_execution_cost"], stage_c["module_description_bits"],
                stage_c["caller_description_bits"]]])
    stage = data.get("stage_b")
    if stage:
        print("### Stage B — the found module applied at every position of fresh screens\n")
        table(["episodes", "positions per screen", "caller nodes", "max error", "median s"],
              [[stage["episodes"], stage["positions"], stage["caller_nodes"], stage["max_error"],
                f(stage["median_seconds"], 3)]])
    print("### Supervision\n")
    table(["train records", "validation records", "held-out records", "positive fraction",
           "observation bytes", "positions per screen"],
          [[data["narrow"]["train_records"], data["narrow"]["validation_records"],
            data["narrow"]["held_records"], f(data["positive_fraction"]),
            data["observation_bytes"], data["positions_per_screen"]]])


def selections():
    data = rung1_data()
    if not data:
        return
    print("### What each arm selected\n")
    rows = []
    for key, label in (("narrow", "A"), ("wide_offsets_ablation", "B"),
                       ("free_operand_ablation", "C")):
        arm = data.get(key)
        if not arm or "validation_filtered" not in arm:
            continue
        for s in arm["validation_filtered"]["selections"]:
            rows.append([label, arm["offsets"][s["shifted"]], f"truth_{s['rg']}",
                         f"truth_{s['edge']}",
                         ",".join(str(s[f"cmp_{c}"]) for c in "rgb")])
    table(["arm", "offset chosen (bytes)", "combinator 1", "combinator 2", "cmp_r,g,b candidate"],
          rows)


def surrogate():
    data = load("surrogate")
    if not data:
        return
    print("### `eq`'s training surrogate against the palette it has to compare\n")
    table(["palette_levels", "colours", "channel separation", "value at tau=1", "slope at tau=1",
           "value at tau=256", "slope at tau=256"],
          [[k, v["colours"], v["separation"], f"{v['surrogate_tau_1']:.2e}",
            f"{v['slope_tau_1']:.2e}", f"{v['surrogate_tau_256']:.2e}",
            f"{v['slope_tau_256']:.2e}"]
           for k, v in data["palette"].items()])
    print("### Distances actually compared at the rung-1 operating point\n")
    table(["palette_levels", "fraction equal", "fraction with live slope, tau=1",
           "fraction with live slope, tau=256", "distinct distances"],
          [[k.split("_")[-1], f(v["fraction_zero"]), f(v["fraction_live_tau_1"]),
            f(v["fraction_live_tau_256"]), str(v["distinct_distances"][:8])]
           for k, v in data.items() if k.startswith("operating_levels_")])
    print("### Gradient reaching each choice node, one backward pass at initialisation\n")
    rows = []
    for key, value in data.items():
        if not key.startswith("gradients_"):
            continue
        _, _, levels, pool, mode = key.split("_")
        for node, entry in value.items():
            if node.startswith("_"):
                continue
            rows.append([f"levels {levels}", pool, mode, node, entry["candidates"],
                         "None" if entry["grad_is_none"] else f"{entry['grad_abs_max']:.2e}"])
    table(["palette", "offset pool", "single-candidate nodes", "choice node", "candidates",
           "max abs gradient"], rows)


def tiebreak():
    data = load("tiebreak")
    if not data:
        return
    print("### Does the lexicographic tie-break bite, and does the validation filter fix it?\n")
    table(["setting", "conforming", "distinct functions", "wrong on held-out",
           "lexicographic pick wrong", "validation survivors", "survivor functions",
           "survivor max held-out error"],
          [[k, v["conforming"], v["distinct_functions"], v["wrong_on_held_out"],
            "yes" if v["lexicographic_pick_wrong"] else "no", v["validation_survivors"],
            v["survivor_functions"],
            "-" if v["survivor_max_held_error"] is None else f(v["survivor_max_held_error"])]
           for k, v in data.items()])


def render():
    template = (OUT.parent / "RESULTS.template.md").read_text()
    import io, contextlib
    out = []
    for line in template.splitlines():
        marker = line.strip()
        if marker.startswith("<!-- TABLE:") and marker.endswith("-->"):
            name = marker[len("<!-- TABLE:"):-3].strip()
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                {"bounds": bounds, "dial": dial, "rung1": rung1, "selections": selections,
                 "surrogate": surrogate, "tiebreak": tiebreak}[name]()
            out.append(buffer.getvalue().rstrip())
        else:
            out.append(line)
    (OUT.parent / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("wrote", OUT.parent / "RESULTS.md")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "render":
        render()
        raise SystemExit
    for name, fn in (("bounds", bounds), ("dial", dial), ("rung1", rung1),
                     ("selections", selections), ("surrogate", surrogate),
                     ("tiebreak", tiebreak)):
        if which in ("all", name):
            fn()
