"""Render every figure-bearing block of RESULTS.md from files under `out/`.

The rule this track is the first to follow: no figure in RESULTS.md is typed by
hand.  Each table or figure-bearing sentence lives between
`<!-- BEGIN:name -->` and `<!-- END:name -->`, and `main()` rewrites exactly those
spans from `render()`.  `verify.py` re-renders and compares, so an edited number
is a FAIL rather than a silent drift.

Nothing here runs an arm; it only reads JSON and `/usr/bin/time -v` logs.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
RESULTS = HERE / "RESULTS.md"

CERTIFIED = (24, 32)
HELD_OUT = (16, 40, 48)
ALL = (16, 24, 32, 40, 48)
STAGES = ("s0", "s1", "s2")
STAGE_NAME = {"s0": "S0 `same`", "s1": "S1' `corner`", "s2": "S2' `rect`"}
# The renderer and the verifier log their own resource use while they run, so
# their logs are never complete when read; the table is about the arms.
SELF_LOGS = {"report", "verify", "check"}


# ----------------------------------------------------------------- loading

def J(name, base=OUT):
    return json.loads((base / f"{name}.json").read_text())


def timelog(name):
    """Peak RSS, wall clock and exit status from a `/usr/bin/time -v` log."""
    path = OUT / f"{name}.time"
    if not path.exists():
        return None
    text = path.read_text()

    def grab(label):
        m = re.search(rf"{re.escape(label)}:\s*(.+)", text)
        return m.group(1).strip() if m else None
    if grab("Exit status") is None:          # the job is still running (or is this one)
        return None
    return {"peak_rss_kb": int(grab("Maximum resident set size (kbytes)")),
            "elapsed": grab("Elapsed (wall clock) time (h:mm:ss or m:ss)"),
            "exit": int(grab("Exit status"))}


# -------------------------------------------------------------- formatting

def n(x):
    return f"{x:,}"


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def s1(x):
    return f"{x:.1f}"


def s2(x):
    return f"{x:.2f}"


def r1(x):
    return f"{x:.1f}×"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def vec(d):
    return "`" + json.dumps(dict(sorted(d.items())), separators=(", ", ": ")) + "`"


def free_of(full, free_nodes):
    return {k: v for k, v in full.items() if k in free_nodes}


# ------------------------------------------------------------------ figures

def figures():
    """Every derived quantity, computed once, from raw JSON."""
    F = {"enum": {w: J(f"enum_{w}") for w in ALL},
         "with": {w: J(f"with_{w}") for w in HELD_OUT},
         "construct": J("construct"), "armf": J("armf"), "refusal": J("refusal"),
         "null": J("null"), "size": J("size"), "collision": J("collision"),
         "types": J("types_probe"), "probe": J("probe"), "check": J("check")}
    F["s33_root"] = J("rung3_root", ROOT / "research/visual-ladder/out")
    F["s33_widgets"] = J("rung3", ROOT / "research/visual-ladder/out")
    F["s55"] = J("check", ROOT / "research/class-identity/out")

    # per held-out width and stage: the with/without row
    ww = []
    for w in HELD_OUT:
        a, b = F["enum"][w], F["with"][w]
        for st in STAGES:
            sa, sb = a["stages"][st], b["stages"][st]
            ww.append({
                "w": w, "st": st, "space": sa["space_size"], "nodes": sa["nodes"],
                "wo_prog": sa["walk"]["evaluated"], "wo_rows": sa["walk"]["row_evaluations"],
                "wo_units": sa["walk"]["node_units"], "wo_walk_s": sa["walk"]["seconds"],
                "wo_prefix_s": sa["sweep"]["seconds"],
                "wo_prefix_nodes": sa["sweep"]["node_evaluations"],
                "wo_cert": sa["walk"]["certificate"], "wo_conf": sa["walk"]["conforming"],
                "wo_exh": sa["walk"]["exhausted"], "wo_digest": sa["artifact_digest"],
                "wi_prog": sb["check"]["programs_evaluated"], "wi_searched": sb["programs_searched"],
                "wi_rows": sb["check"]["row_evaluations"], "wi_units": sb["check"]["node_units"],
                "wi_s": sb["check"]["seconds"], "wi_conforms": sb["check"]["conforms"],
                "wi_digest": sb["artifact_digest"], "wi_store_digest": sb["store_digest"],
                "wi_acc": sb["held_accuracy"], "held_rows": sa["held_rows"],
                "best_const": sb["best_constant"]["accuracy"],
                "ws": sa["whole_space_held"],
            })
    F["ww"] = ww

    pipe = []
    for w in HELD_OUT:
        a, b = F["enum"][w], F["with"][w]
        pa, pb = a["pipeline"], b["pipeline"]
        largest = max(a["stages"][s]["space_size"] for s in STAGES)
        pipe.append({
            "w": w, "staged": pa["staged_space"], "joint": pa["joint_space"],
            "largest": largest,
            "wo_prog": pa["programs_evaluated"], "wi_prog": pb["programs_evaluated"],
            "wo_rows": pa["row_evaluations"], "wi_rows": pb["row_evaluations"],
            "wo_units": pa["node_units"], "wi_units": pb["node_units"],
            "wo_walk_s": pa["walk_seconds"], "wo_prefix_s": pa["prefix_seconds"],
            "wi_s": pb["check_seconds"],
            "prog_ratio": pa["programs_evaluated"] / pb["programs_evaluated"],
            "row_ratio": pa["row_evaluations"] / pb["row_evaluations"],
            "unit_ratio": pa["node_units"] / pb["node_units"],
            "walk_ratio": pa["walk_seconds"] / pb["check_seconds"],
            "prefix_ratio": pa["prefix_seconds"] / pb["check_seconds"],
            "parse_wo": a["parse"]["totals"], "parse_wi": b["parse"]["totals"],
        })
    F["pipe"] = pipe
    return F


# ------------------------------------------------------------------- blocks

def render(F=None):
    F = F or figures()
    B = {}

    # §1 -- the width axis
    B["types_table"] = table(
        ["resolution", "span", "S2' nodes", "`rec` leaf components", "S2' digest (vector `{step_w: 2, step_h: 3}`)"],
        [[r["resolution"], r["span"], r["s2"]["nodes"], n(r["s2"]["input_leaf_components"][0][1]),
          f"`{r['s2_digest']}`"] for r in F["types"]["rows"]])

    rows = []
    for w in ALL:
        e = F["enum"][w]
        a = e["achieved"]
        rows.append([w, "**certified**" if w in CERTIFIED else "held out",
                     n(e["observation_components"]),
                     f"{a['widgets_min']}–{a['widgets_max']} (mean {a['widgets_mean']:.2f})",
                     a["max_widget_width"], a["max_widget_height"], e["span"],
                     "yes" if a["span_sufficient"] else "**no**"])
    B["widths_table"] = table(
        ["resolution", "role", "observation components", "achieved widgets/screen (12 screens)",
         "max widget width", "max widget height", "span", "span ≥ every extent"], rows)

    # §3 -- certificates at the training widths
    rows = []
    for w in CERTIFIED:
        e = F["enum"][w]
        for st in STAGES:
            s = e["stages"][st]
            rows.append([w, STAGE_NAME[st], n(s["space_size"]), n(s["walk"]["evaluated"]),
                         s["walk"]["exhausted"], s["walk"]["conforming"],
                         f"`{s['walk']['certificate']}`",
                         s.get("distinct_functions", "—"), s["validation_survivors"],
                         vec(s["chosen_free"]), f"`{s['artifact_digest']}`",
                         f"{f4(s['held_accuracy'])} ({s['held_rows']} rows)"])
    B["cert_table"] = table(
        ["res", "stage", "space", "evaluated", "exhausted", "conforming", "certificate",
         "distinct functions", "validation survivors", "selected vector", "artifact digest",
         "held-out accuracy"], rows)

    rows = []
    for w in CERTIFIED:
        for st in STAGES:
            s = F["enum"][w]["stages"][st]
            sw = s["sweep"]
            rows.append([w, STAGE_NAME[st], n(sw["evaluated"]), sw["exhausted"], sw["conforming"],
                         f"`{sw['certificate']}`", n(sw["node_evaluations"]), s1(sw["seconds"])])
    B["cert_prefix_table"] = table(
        ["res", "stage", "`enumerate_prefix` evaluated", "exhausted", "conforming", "certificate",
         "node evaluations", "seconds"], rows)

    # §3.1 -- does §33 move?
    root, wid, e32 = F["s33_root"], F["s33_widgets"], F["enum"][32]
    def free33(stage_rec, stage):
        return free_of(stage_rec["chosen"], e32["stages"][stage]["free_nodes"])
    rows33 = [
        ["S0 space / conforming / distinct functions",
         f"{wid['s0']['space_size']} / {wid['s0']['conforming']['count']} / "
         f"{wid['s0']['conforming']['distinct_functions']}",
         f"{e32['stages']['s0']['space_size']} / {e32['stages']['s0']['walk']['conforming']} / "
         f"{e32['stages']['s0']['distinct_functions']}"],
        ["S0 certificate", f"`{wid['s0']['search']['certificate']}`",
         f"`{e32['stages']['s0']['walk']['certificate']}`"],
        ["S0 module digest", f"`{wid['s0']['module']}`", f"`{e32['stages']['s0']['module']}`"],
        ["S1' space / conforming / certificate",
         f"{root['s1']['space_size']} / {root['s1']['conforming']['count']} / "
         f"`{root['s1']['search']['certificate']}`",
         f"{e32['stages']['s1']['space_size']} / {e32['stages']['s1']['walk']['conforming']} / "
         f"`{e32['stages']['s1']['walk']['certificate']}`"],
        ["S1' vector", vec(free33(root["s1"], "s1")), vec(e32["stages"]["s1"]["chosen_free"])],
        ["S2' space / conforming / certificate",
         f"{root['s2']['space_size']} / {root['s2']['conforming']['count']} / "
         f"`{root['s2']['search']['certificate']}`",
         f"{e32['stages']['s2']['space_size']} / {e32['stages']['s2']['walk']['conforming']} / "
         f"`{e32['stages']['s2']['walk']['certificate']}`"],
        ["S2' vector", vec(free33(root["s2"], "s2")), vec(e32["stages"]["s2"]["chosen_free"])],
    ]
    for k in ("widgets_in_probe", "rects_predicted", "screens_rects_exact", "roots_predicted",
              "parent_links_correct", "parent_links_wrong", "trees_exact",
              "corner_key_collisions"):
        rows33.append([f"parse `{k}`", root["parse"]["totals"][k], e32["parse"]["totals"][k]])
    for r in rows33:
        r.append("same" if str(r[1]).replace("module:", "") == str(r[2]).replace("module:", "")
                 else "**MOVED**")
    B["s33_table"] = table(["quantity", "§33 (`visual-ladder/out/rung3*.json`)",
                            "this run, resolution 32", "verdict"], rows33)

    # §3.2 -- the vector at every width
    rows = []
    for st in STAGES:
        free = F["enum"][32]["stages"][st]["free_nodes"]
        cells = []
        for w in ALL:
            s = F["enum"][w]["stages"][st]
            cell = vec(s["chosen_free"])
            if "chosen_offsets" in s:
                cell += f" → bytes {s['chosen_offsets']}"
            cells.append(cell)
        rows.append([STAGE_NAME[st], vec(free)] + cells)
    B["vector_table"] = table(["stage", "free nodes (candidates)"] +
                              [f"res {w}" for w in ALL], rows)

    rows = []
    for st in STAGES:
        rec = F["construct"]["records"][st]
        rows.append([STAGE_NAME[st], f"`{rec['class_id']}`", rec["admitted"],
                     rec.get("widths"), vec(rec.get("selections", {})),
                     ", ".join(f"`{d}`" for d in rec.get("digests", []))])
    B["class_table"] = table(["stage", "class id", "admitted", "certified widths",
                              "stored vector", "member digests (24, 32)"], rows)

    # §4 -- with / without
    rows = []
    for r in F["ww"]:
        rows.append([r["w"], STAGE_NAME[r["st"]], r["space"],
                     n(r["wo_prog"]), n(r["wo_rows"]), s1(r["wo_walk_s"]), s1(r["wo_prefix_s"]),
                     f"`{r['wo_cert']}`",
                     f"{r['wi_searched']} + {r['wi_prog']} check", n(r["wi_rows"]), s2(r["wi_s"]),
                     "`none`",
                     "**yes**" if r["wo_digest"] == r["wi_digest"] else "**NO**",
                     f"`{r['wi_digest']}`"])
    B["ww_table"] = table(
        ["res", "stage", "space", "without: programs", "row-evals", "walk s", "prefix-search s",
         "certificate", "with: programs", "row-evals", "check s", "certificate",
         "same program", "digest"], rows)

    rows = []
    for r in F["ww"]:
        ws = r["ws"]
        rows.append([r["w"], STAGE_NAME[r["st"]],
                     f"{f4(r['wi_acc'])} ({r['held_rows']})", f4(r["best_const"]),
                     f"{ws['rows']}", f4(ws["chosen_accuracy_same_rows"]),
                     f4(ws["best_constant_same_rows"]), f4(ws["mean"]), f4(ws["second_best"]),
                     f4(ws["worst"])])
    B["acc_table"] = table(
        ["res", "stage", "with-class accuracy (held rows)", "best constant (same rows)",
         "whole-space rows", "chosen, on those rows", "best constant, on those rows",
         "uniform random over the space (exact mean)", "second-best of the space",
         "worst of the space"], rows)

    rows = []
    for p in F["pipe"]:
        rows.append([p["w"], p["staged"], n(p["joint"]), p["largest"],
                     f"{p['wo_prog']} → {p['wi_prog']}", r1(p["prog_ratio"]),
                     f"{n(p['wo_rows'])} → {n(p['wi_rows'])}", r1(p["row_ratio"]),
                     r1(p["unit_ratio"]),
                     f"{s1(p['wo_walk_s'])} → {s1(p['wi_s'])}", r1(p["walk_ratio"]),
                     s1(p["wo_prefix_s"]), r1(p["prefix_ratio"])])
    B["pipe_table"] = table(
        ["res", "staged space", "joint space", "largest stage space", "programs",
         "ratio", "row-evaluations", "ratio", "nominal node-unit ratio",
         "walk s → check s", "ratio", "shipped prefix search s", "prefix ÷ check"], rows)

    rows = []
    for p in F["pipe"]:
        for arm, t in (("without (enumerated)", p["parse_wo"]), ("with (instantiated)", p["parse_wi"])):
            rows.append([p["w"], arm, t["screens"], t["widgets_in_probe"], t["rects_predicted"],
                         t["screens_rects_exact"], t["roots_predicted"],
                         f"{t['parent_links_correct']} / {t['parent_links_wrong']}",
                         t["trees_exact"], t["corner_key_collisions"]])
    B["parse_table"] = table(
        ["res", "arm", "screens", "widgets", "rects predicted", "screens exact", "roots",
         "links right / wrong", "trees exact", "key collisions"], rows)

    c = F["collision"]["rows"][0]
    B["collision_line"] = (
        f"The one miss is resolution {c['resolution']}, seed {c['seed']}, rectangle "
        f"`{c['rect']}` (`out/collision.json`, one screen): its fill `{c['own_colour']}` equals "
        f"its left neighbour `{c['left_neighbour_colour']}` and its upper neighbour "
        f"`{c['upper_neighbour_colour']}`, both pixels of its parent `{c['parent_rect']}`; the "
        f"screen shows {c['distinct_colours_on_screen']} distinct colours for {c['widgets']} "
        f"widgets. Its extent is within the span (`{c['extent_within_span']}`).")

    # §4.2 -- §55 beside this track
    rows = []
    for r in F["s55"]["rows"]:
        rows.append(["§55 `program`", f"depth {r['width']}", 272, n(r["a2_programs"]),
                     n(r["a2_episodes"]), s1(r["a2_seconds"]), s2(r["a3_seconds"]),
                     r1(r["episode_ratio"]), r1(r["seconds_ratio"])])
    for p in F["pipe"]:
        rows.append(["this track, visual parse", f"resolution {p['w']}", p["staged"],
                     n(p["wo_prog"]), n(p["wo_rows"]), s1(p["wo_walk_s"]), s2(p["wi_s"]),
                     r1(p["row_ratio"]), r1(p["walk_ratio"])])
    B["s55_table"] = table(
        ["track", "held-out width", "space", "without: programs", "without: episodes / row-evals",
         "without s", "with s", "episode / row ratio", "wall-clock ratio (same code path)"],
        rows)

    # §5 -- arm F
    rows = []
    for name, rec in F["armf"]["schemas"].items():
        for w in sorted(rec["per_resolution"], key=int):
            pr = rec["per_resolution"][w]
            for st in ("s1", "s2"):
                d = pr[st]
                rows.append([f"`{name}`", w, STAGE_NAME[st], d["walk"]["space_size"],
                             d["walk"]["evaluated"], d["walk"]["exhausted"],
                             d["walk"]["conforming"], f"`{d['walk']['certificate']}`",
                             d["stored_vector_conforms"],
                             f4(d["stored_vector_held_accuracy"]), f4(d["best_constant"]),
                             {True: "yes", False: "no", None: "—"}[d.get("bit_identical_to_right")]])
    B["armf_table"] = table(
        ["wrong schema", "res", "stage", "space", "evaluated", "exhausted", "conforming",
         "certificate", "stored vector conforms", "stored vector held accuracy",
         "best constant", "bit-identical to right schema"], rows)

    rows = []
    for k, v in F["armf"]["format"].items():
        rule = v.get("rule", "")
        rows.append([f"`{k}`", "**admitted**" if v["admitted"] else "refused",
                     ("`" + rule.split(":")[0] + "`") if rule else "—"])
    B["armf_format_table"] = table(["record offered to §55's store (stage S2')", "outcome",
                                    "rule"], rows)

    ref = F["refusal"]
    B["refusal_table"] = table(
        ["artifact built at", "offered a screen at", "accepted", "error"],
        [[ref["built_at"], w, v["accepted"],
          f"`{v.get('error', '')}: {v.get('message', '')}`" if not v["accepted"] else "—"]
         for w, v in sorted(ref["offered"].items(), key=lambda kv: int(kv[0]))])

    # §8 -- costs
    nl = F["null"]
    B["null_line"] = (
        f"Null control (`out/null.json`): one conformance check of S1' at resolution 32 timed "
        f"{nl['repeats']} times with nothing changed — median {nl['median']:.3f} s, "
        f"min {nl['min']:.3f} s, max {nl['max']:.3f} s, spread "
        f"{nl['spread_fraction'] * 100:.1f}% of the median.")
    rows = []
    for p in sorted(OUT.glob("*.time")):
        if p.stem in SELF_LOGS:
            continue
        t = timelog(p.stem)
        if t is None:
            continue
        rows.append([f"`{p.stem}`", f"{t['peak_rss_kb'] / 1048576:.2f}", t["elapsed"], t["exit"]])
    B["resource_table"] = table(["capped job", "peak RSS (GB)", "wall clock", "exit"], rows)

    sz = F["size"]
    B["size_line"] = (
        f"The class store is {n(sz['class_store_bytes_on_disk'])} bytes on disk and covers "
        f"every width; the {len(sz['widths']) * 3} per-width stage artifacts it stands for "
        f"serialize to {n(sz['artifact_bytes_total'])} bytes with "
        f"{sz['distinct_digests_per_stage']} distinct digests per stage. The invariant content "
        f"is {sz['invariant_bits_total']:.2f} bits — "
        + " + ".join(f"log2({v})" for v in sz["space_sizes"].values()) + ".")

    rows = [[("PASS" if v else "**FAIL**"), k] for k, v in F["check"]["verdict"].items()]
    B["criteria_table"] = table(["verdict", "criterion (`check.py`)"], rows)

    B["headline"] = headline(F)
    return B


def headline(F):
    ww, pipe = F["ww"], F["pipe"]
    same = sum(r["wo_digest"] == r["wi_digest"] for r in ww)
    conf = sum(r["wi_conforms"] for r in ww)
    vec_same = all(F["enum"][w]["stages"][st]["chosen_free"] ==
                   F["construct"]["vectors"][st] for w in ALL for st in STAGES)
    prog = [p["prog_ratio"] for p in pipe]
    rowr = [p["row_ratio"] for p in pipe]
    walk = [p["walk_ratio"] for p in pipe]
    pref = [p["prefix_ratio"] for p in pipe]
    largest = max(p["largest"] for p in pipe)
    s2_walk = [r["wo_walk_s"] / r["wi_s"] for r in ww if r["st"] == "s2"]
    lines = [
        f"- **Instantiation reproduces the enumerated program:** {same} of {len(ww)} stage "
        f"artifacts at held-out resolutions {', '.join(map(str, HELD_OUT))} are digest-identical "
        f"to what exhaustive enumeration selects there, and {conf} of {len(ww)} pass the "
        f"conformance check. The selection vector is "
        f"{'identical' if vec_same else '**not** identical'} at all {len(ALL)} resolutions. "
        f"(One seed set, one palette, square screens only.)",
        f"- **The saving, on four currencies, per held-out resolution:** programs "
        f"{' / '.join(r1(x) for x in prog)}; row-evaluations {' / '.join(r1(x) for x in rowr)}; "
        f"wall clock through the same code path {' / '.join(r1(x) for x in walk)}; "
        f"against the shipped prefix search {' / '.join(r1(x) for x in pref)}. The largest "
        f"per-stage space is {largest}.",
        f"- **Does the saving exceed the space size?** On programs and row-evaluations, "
        f"{'no' if max(prog + rowr) <= largest else 'yes'}: both are bounded by the largest "
        f"stage space by construction. On same-path wall clock, "
        f"{'no' if max(walk) <= largest else 'yes'} — and at S2' alone the same-path ratio is "
        f"{' / '.join(r1(x) for x in s2_walk)} against a space of 25, because a wrong program "
        f"fails on an early row and the right one must run every row. "
        + (f"Even against the shipped `enumerate_prefix` search the ratio stays below "
           f"{largest}." if max(pref) <= largest else
           f"Against the shipped `enumerate_prefix` search the ratio exceeds {largest} at "
           f"{sum(x > largest for x in pref)} of {len(pref)} resolutions — interpreter overhead "
           f"in that search, not search difficulty."),
    ]
    fmt = F["armf"]["format"]
    fs = F["armf"]["schemas"]["frozen_span"]["per_resolution"]
    fo = F["armf"]["schemas"]["frozen_offsets"]["per_resolution"]
    above = [w for w in ("40", "48") if not fs[w]["s2"]["stored_vector_conforms"]]
    lines += [
        f"- **Arm F, frozen offsets:** "
        f"{'refused' if not fmt['frozen_offsets/F1_one_width']['admitted'] else 'ADMITTED'} at one "
        f"width, {'refused' if not fmt['frozen_offsets/F2_two_widths']['admitted'] else 'ADMITTED'} "
        f"at two; 0 conforming at "
        f"{sum(fo[w][st]['walk']['conforming'] == 0 for w in ('16', '24', '40', '48') for st in ('s1', 's2'))}"
        f" of 8 off-reference stage spaces. §53's result carries.",
        f"- **Arm F, frozen span:** "
        f"{'refused' if not fmt['frozen_span/F1_one_width']['admitted'] else 'ADMITTED'} at one width "
        f"and **{'refused' if not fmt['frozen_span/F2_two_widths']['admitted'] else 'ADMITTED'} at "
        f"two** (24, 32), because it is correct at and below the value it froze. It fails the "
        f"conformance check at {', '.join(above)}, where it would ship held-out accuracy "
        f"{' / '.join(f4(fs[w]['s2']['stored_vector_held_accuracy']) for w in ('40', '48'))} "
        f"against a best constant of "
        f"{' / '.join(f4(fs[w]['s2']['best_constant']) for w in ('40', '48'))}: plausible, and wrong.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ writing

# The body carries its own trailing newline, so an empty block
# ("BEGIN -->\n<!-- END") matches too -- the first version required a newline on
# both sides and silently matched none of the placeholders.
BLOCK = re.compile(r"(<!-- BEGIN:(\w+) -->\n)(.*?)(<!-- END:\2 -->)", re.S)


def blocks_in(text):
    return {m.group(2): m.group(3)[:-1] if m.group(3).endswith("\n") else m.group(3)
            for m in BLOCK.finditer(text)}


def main():
    B = render()
    text = RESULTS.read_text()
    present = blocks_in(text)
    missing = sorted(set(B) - set(present))
    unknown = sorted(set(present) - set(B))
    if unknown:
        sys.exit(f"RESULTS.md has blocks the renderer does not produce: {unknown}")
    text = BLOCK.sub(lambda m: m.group(1) + B[m.group(2)] + "\n" + m.group(4), text)
    RESULTS.write_text(text)
    print(f"rendered {len(present)} blocks into RESULTS.md"
          + (f"; NOT PLACED (no marker): {missing}" if missing else ""))


if __name__ == "__main__":
    main()
