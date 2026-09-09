"""Render every table in RESULTS.md straight from out/*.json.

No number in the report is transcribed by hand: `RESULTS.template.md` carries
`{{name}}` placeholders and this script substitutes the rendered blocks.
"""
from __future__ import annotations

import json
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parent / "out"


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def num(x, digits=4):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if x == 0:
        return "0"
    return f"{x:.{digits}g}"


def blocks():
    b = {}

    # ---- audit -------------------------------------------------------
    a = load("audit")
    if a:
        rows = []
        for r in a["rows"]:
            rows.append([r["resolution"], r["pixels_per_image"],
                         num(r["foreground_fraction_train"], 3),
                         num(r["depth_zero_equals_no_object"]),
                         num(r["foreground"]["held_accuracy"], 4),
                         num(r["object_ids"]["held_accuracy"], 4),
                         num(r["object_ids"]["majority_accuracy"], 4),
                         num(r["depth_q"]["collision_fraction"], 3)])
        b["audit"] = table(["R", "pixels", "foreground fraction",
                            "`depth==0` iff no object", "held-out lookup acc (foreground)",
                            "held-out lookup acc (`object_ids`)", "majority baseline",
                            "colliding RGB (exact depth)"], rows)

    # ---- rung 3 ------------------------------------------------------
    m = load("rung3_mask_combined") or load("rung3_mask")
    if m:
        rows, ident, grad = [], [], []
        for r in m["rows"]:
            e, f = r["enumerate_fit"], r["enumerate_first"]
            ap = r.get("apply", {})
            rows.append([r["resolution"], num(r["observation_width"]), num(r["positions"]),
                         num(r["train_examples"]), num(r["space_size"]),
                         num(e["evaluated"]), num(e["exhausted"]), num(e["unique"]),
                         f"{e['seconds']:.1f}", f"{f['evaluated']:,} / {f['seconds']:.2f}s",
                         num(e.get("held_out_max_error")),
                         num(ap.get("max_error_over_dense_probe")),
                         num(ap.get("max_count_error")),
                         num(r.get("caller_nodes"))])
            for c in r["identification"]["curve"]:
                ident.append([r["resolution"], num(c["supervised_records"]),
                              num(c["conforming"]), num(c["also_exact_on_held_out"]),
                              num(c["unique"]),
                              num(1 - c["also_exact_on_held_out"] / max(1, c["conforming"]), 3)])
            g = r["gradient"]["summary"]
            held = [x.get("held_err") for x in r["gradient"]["rows"] if "held_err" in x]
            grad.append([r["resolution"], f"{g['successes']}/{g['n']}",
                         f"{g['median_seconds']:.1f}", num(r["random_control"]["density"], 3),
                         num(max(held) if held else None),
                         f"{e['seconds']:.1f}"])
        b["rung3"] = table(["R", "obs width", "positions", "supervised records", "space",
                            "programs evaluated", "exhausted", "unique", "enum s",
                            "stop-at-first", "held-out err (records)",
                            "held-out err (whole probe)", "count err", "caller nodes"], rows)
        b["rung3_identification"] = table(
            ["R", "supervised records", "conforming", "also exact on held-out", "unique",
             "fraction that fail held-out"], ident)
        b["rung3_gradient"] = table(
            ["R", "gradient successes (init_noise 0.5)", "median s", "random-draw density",
             "worst held-out error over seeds", "exhaustive enumeration s"], grad)

    # ---- tie-break ---------------------------------------------------
    t = load("tiebreak")
    if t:
        rows = []
        for r in t["rows"]:
            for rule, v in r["rules"].items():
                rows.append([r["resolution"], num(r["space_size"]), num(r["conforming_train"]),
                             num(r["conforming_train_and_validation"]), rule,
                             f"{v['accuracy']:.6f}" if v else "--",
                             f"{v['max_error']:.1f}" if v else "--",
                             f"{v['wrong_slots']}/{v['slots']}" if v else "--"])
        b["tiebreak"] = table(["R", "space", "conforming on train", "+ validation",
                               "selection rule", "test accuracy (whole probe)",
                               "test max error", "wrong slots"], rows)

    # ---- rung 3.5 ----------------------------------------------------
    w = load("rung35_window")
    fl = (load("rung35_flat") or {})
    if w and "staged" in w and fl:
        s = w["staged"]
        e = s["enumerate_fit"]
        rows = [["staged (frozen module called twice)", num(s["space_size"]),
                 num(e["evaluated"]), num(e["exhausted"]), num(e["unique"]),
                 f"{e['seconds']:.3f}", num(e.get("held_out_max_error")),
                 num(e.get("held_out_accuracy"), 5),
                 f"{s['gradient']['summary']['successes']}/{s['gradient']['summary']['n']}"],
                ["flat (nothing frozen)", num(fl["space_size"]),
                 num(fl["enumerate_partial"]["evaluated"]),
                 num(fl["enumerate_partial"]["exhausted"]), "--",
                 f"{fl['enumerate_partial']['seconds']:.1f}", "--", "--",
                 (f"{fl['gradient']['summary']['successes']}/{fl['gradient']['summary']['n']}"
                  if "gradient" in fl else "--")]]
        b["rung35"] = table(["arm", "space", "programs evaluated", "exhausted", "unique",
                             "enum s", "held-out max error", "held-out accuracy",
                             "gradient"], rows)

    # ---- rung 4 ------------------------------------------------------
    o = load("rung4_objects")
    if o:
        o.setdefault("families", [])
        o.setdefault("colour_multiplicity", None)
        rows = [[c["target"], c["classes"], num(c["distinct_rgb"]), num(c["colliding_rgb"]),
                 num(c["train_accuracy"], 4), num(c["held_out_accuracy"], 4),
                 num(c["majority_baseline"], 4), num(c["advantage_over_majority"], 3)]
                for c in o["ceilings"]]
        b["rung4_ceilings"] = table(
            ["target", "classes", "distinct RGB (train)", "colliding RGB",
             "train accuracy", "held-out accuracy", "majority baseline",
             "advantage"], rows)
        rows = []
        for f in o["families"]:
            rows.append([f["family"], num(f["space_size"]), num(f["evaluated"]),
                         num(f["exhausted"]), num(f["solved"]), num(f.get("unique")),
                         f"{f['seconds']:.1f}",
                         num(f.get("held_out_accuracy"), 5) if f.get("solved") else "--"])
        b["rung4_families"] = table(
            ["family / target", "space", "programs evaluated", "exhausted", "solved",
             "unique", "s", "held-out accuracy"], rows)
        c = o["colour_multiplicity"]
        if c: b["rung4_colours"] = (
            f"mean distinct colours per object **{c['mean_distinct_colours_per_object']:.2f}**, "
            f"max **{c['max_distinct_colours_per_object']}**, "
            f"single-coloured objects **{100*c['fraction_single_coloured']:.1f}%** "
            f"({c['objects_measured']} object instances)")
        if not o["families"]: b.pop("rung4_families", None)

    # ---- rung 5 ------------------------------------------------------
    d = load("rung5_depth")
    if d:
        rows = [[k.replace("ceiling_", ""), num(v["distinct_keys"]),
                 num(v["train_accuracy"], 4), num(v["held_out_accuracy"], 4),
                 num(v["majority_baseline"], 4),
                 num(v["held_out_accuracy"] - v["majority_baseline"], 3)]
                for k, v in d.items() if k.startswith("ceiling_")]
        b["rung5_ceilings"] = table(
            ["information source", "distinct keys", "train accuracy", "held-out accuracy",
             "majority baseline", "advantage"], rows)
        rows = []
        for key, label in (("colour_family", "colour constants (rung-3 family)"),
                           ("position_family", "position threshold x colour"),
                           ("position_only_conformance", "position threshold, colour pinned")):
            v = d[key]
            rows.append([label, num(v["space_size"]), num(v["evaluated"]),
                         num(v["exhausted"]), num(v["solved"]), f"{v['seconds']:.1f}"])
        b["rung5_families"] = table(
            ["family", "space", "programs evaluated", "exhausted", "solved", "s"], rows)
        b["rung5_families"] += ("\n\nNo row is solved, so `unique` is undefined for all "
                                "of them; an exhausted, unsolved sweep is a completeness "
                                "certificate that the family contains no exact program.")
        bi = d["position_only_best"]
        b["rung5_best"] = (
            f"best-accuracy program in the 480-program position family: "
            f"train **{bi['best_train_accuracy']:.4f}**, held-out **{bi['held_out_accuracy']:.4f}**, "
            f"against a majority baseline of "
            f"**{d['ceiling_position']['majority_baseline']:.4f}**")

    # ---- full alphabet ------------------------------------------------
    fa = load("full_alphabet")
    if fa:
        br, sc = fa["brute_force"], fa["observed_alphabet"]
        hc, g = fa["hill_climb"], fa["gradient"]
        rows = [["exhaustive enumeration (`enumerate_fit`)", num(fa["space_size"]),
                 num(br["evaluated"]), num(br["exhausted"]), "--", num(br["solved"]),
                 f"{br['seconds']:.1f}", f"{br['projected_exhaustive_seconds']:.3g}"],
                ["observed-alphabet scoping, then enumeration",
                 num(sc["space_size"]), num(sc["enumeration"]["evaluated"]),
                 num(sc["enumeration"]["exhausted"]), "--",
                 num(sc["enumeration"]["solved"]), f"{sc['enumeration']['seconds']:.1f}",
                 f"{sc['enumeration']['projected_exhaustive_seconds']:.3g}"],
                ["coordinate descent, random restarts", num(fa["space_size"]),
                 num(hc["median_evaluations"]), "no", "--",
                 f"{hc['successes']}/{len(hc['restarts'])}",
                 f"{hc['median_seconds']:.1f}", "--"],
                ["gradient descent (init_noise 0.5)", num(fa["space_size"]), "--", "no", "--",
                 f"{g['summary']['successes']}/{g['summary']['n']}",
                 f"{g['summary']['median_seconds']:.1f}", "--"]]
        b["full_alphabet"] = table(
            ["method", "space searched", "programs evaluated", "exhausted", "unique",
             "solved", "s", "projected exhaustive s"], rows)

    # ---- free offsets --------------------------------------------------
    of = load("offsets_free")
    if of:
        e, g = of["enumerate_fit"], of["gradient"]["summary"]
        b["offsets_free"] = table(
            ["space", "programs evaluated", "exhausted", "unique", "enum s",
             "held-out max error", "gradient successes", "gradient median s"],
            [[num(of["space_size"]), num(e["evaluated"]), num(e["exhausted"]),
              num(e["unique"]), f"{e['seconds']:.1f}", num(e["held_out_max_error"]),
              f"{g['successes']}/{g['n']}", f"{g['median_seconds']:.1f}"]])

    # ---- prefix-reusing enumeration --------------------------------------
    inc = load("incremental")
    if inc:
        rows = []
        for r in inc["rows"]:
            i = r["incremental"]
            base = r.get("enumerate_fit_equivalent")
            rows.append([r["resolution"], num(r["observation_width"]), num(r["records"]),
                         num(r["space_size"]), num(i["node_evaluations"]),
                         f"{i['seconds']:.1f}", num(i["count"]),
                         f"{base['seconds']:.1f}" if base else "--",
                         f"{r['speedup']:.0f}x" if base else "--",
                         num(r.get("identical_conforming_set")) if base else "--"])
        b["incremental"] = table(
            ["R", "obs width", "records", "space", "node evaluations",
             "prefix-reusing s", "conforming", "`enumerate_fit` s", "speedup",
             "identical conforming set"], rows)

    # ---- resolution transfer ----------------------------------------------
    rt = load("resolution_transfer")
    if rt:
        rows = []
        for r in rt["rows"]:
            sp = r["search_probe"]
            rows.append([r["resolution"], num(r["observation_width"]), num(r["positions"]),
                         num(r["caller_nodes"]), num(r["space_size"]),
                         num(r["max_error_over_dense_probe"]), num(r["max_count_error"]),
                         f"{r['median_apply_seconds']:.2f}",
                         f"{sp['projected_exhaustive_seconds']:.4g}"])
        b["resolution_transfer"] = table(
            ["R", "obs width", "positions", "caller nodes", "space",
             "max error over dense probe", "count error", "apply s/image",
             "fresh `enumerate_fit` projection (s)"], rows)
        b["resolution_transfer_search"] = (
            f"searched once at R={rt['search_resolution']}: "
            f"**{rt['conforming_train']}** programs conform on training records, "
            f"**{rt['conforming_train_and_validation']}** also on validation, out of "
            f"**{rt['space_size']:,}**, in {rt['search_seconds']:.1f}s")

    # ---- harness check -------------------------------------------------
    c = load("check_local_fit")
    if c:
        b["check_local_fit"] = table(
            ["steps", "polish", "`fit` exact error", "`local_fit` exact error", "identical"],
            [[x["steps"], x["polish"], num(x["fit"]["exact_max_error"]),
              num(x["local_fit"]["exact_max_error"]), num(x["identical"])]
             for x in c["cases"]])
    return b


def main():
    b = blocks()
    template = pathlib.Path(__file__).resolve().parent / "RESULTS.template.md"
    if template.exists():
        text = template.read_text()
        missing = []
        for key in sorted(b, key=len, reverse=True):
            text = text.replace("{{" + key + "}}", b[key])
        for line in text.splitlines():
            if "{{" in line:
                missing.append(line.strip())
        (template.parent / "RESULTS.md").write_text(text)
        print("wrote RESULTS.md;", len(missing), "unfilled placeholders")
        for x in missing:
            print("  ", x)
    else:
        for k, v in b.items():
            print(f"\n### {k}\n{v}")


if __name__ == "__main__":
    main()
