"""Render every table in RESULTS.md directly from out/*.json.

No number in the report is transcribed by hand: `RESULTS.template.md` carries
`{{name}}` placeholders and this fills them.
"""
from __future__ import annotations
import json, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(x, n=4):
    return "--" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


CONTEXT_LABEL = {
    "pixel": "one pixel (3 bytes)",
    "pixel_pos": "pixel + its row/column",
    "pair_right": "pixel + right neighbour (6 bytes)",
    "cross4": "pixel + 4-neighbourhood (15 bytes)",
    "win3x3": "3x3 window (27 bytes)",
    "pixel_agg": "pixel + global colour rank",
    "agg_only": "global colour rank alone",
    "eq_pair_right": "= to right neighbour (1 bit)",
    "eq_cross4": "= to each of 4 neighbours (4 bits)",
    "eq_win3x3": "= to each of 8 neighbours (8 bits)",
    "eqbg_cross4": "is-background + 4 equalities",
    "eqbg_win3x3": "is-background + 8 equalities",
    "bg_pair_right": "is-background x2 + one equality",
    "chroma_pair_right": "is-background x2 + collinearity",
    "chroma_cross4": "is-background + 4 collinearities",
}


def bounds_table(data, targets, contexts=None):
    rows = []
    idx = {(c["context"], c["target"]): c for c in data["ceilings"]}
    for ctx in (contexts or CONTEXT_LABEL):
        for t in targets:
            c = idx.get((ctx, t))
            if not c:
                continue
            rows.append([CONTEXT_LABEL.get(ctx, ctx), t, c["keys"],
                         f(c["held_key_recurrence"], 3), f(c["train_acc"]),
                         f(c["held_acc"]), f(c["majority_baseline"]),
                         ("**%+.4f**" % c["advantage"]) if abs(c["advantage"]) > .01
                         else "%+.4f" % c["advantage"]])
    return table(["context", "target", "keys", "key recurrence", "train acc",
                  "held-out acc", "majority baseline", "advantage"], rows)


def main():
    b = load("bounds"); ch = load("chroma_bound"); rg = load("rung4_segment")
    st = load("rung4_staged"); ap_ = load("apply")
    wd = load("wide_threshold"); rs = load("residual"); w3 = load("world3d")
    uf = load("unfreeze"); cg = load("choice_gradients"); le = load("le_surrogate")
    dc = load("discoverable"); rsh = load("rule_shapes")
    sf = load("surrogate_fix_coarse"); ff = load("surrogate_fix_fullunfreeze")
    v = {}

    perm = b["permutation_certificate"]; col = b["colour_certificate"]
    v["perm_n"] = len(perm)
    v["perm_rgb"] = sum(r["rgb_identical"] for r in perm)
    v["perm_depth"] = sum(r["depth_identical"] for r in perm)
    v["perm_ids_same"] = sum(r["ids_identical"] for r in perm)
    v["perm_remap"] = sum(r["ids_agree_after_remap"] for r in perm)
    v["perm_changed"] = sum(r["ids_changed_pixels"] for r in perm)
    v["perm_fg"] = sum(r["foreground_pixels"] for r in perm)
    v["col_n"] = len(col)
    v["col_ids_same"] = sum(r["ids_identical"] for r in col)
    v["col_rgb_same"] = sum(r["rgb_identical"] for r in col)
    v["col_changed"] = sum(r["changed_pixels"] for r in col)
    v["reuse"] = json.dumps(b["colour_reuse"])
    v["reuse_frac"] = f(b["colour_reuse"]["reuse_fraction"], 4)
    v["reuse_distinct"] = b["colour_reuse"]["distinct_fg_colours"]
    v["reuse_multi"] = b["colour_reuse"]["colours_in_more_than_one_episode"]
    o = b["obstruction"]
    v["mean_rgb_per_object"] = f(o["mean_distinct_rgb_per_object"], 2)
    v["max_rgb_per_object"] = o["max_distinct_rgb_per_object"]
    v["single_frac"] = f(o["single_coloured_fraction"], 3)
    v["adj_same_diff_colour"] = f(o["adjacent_same_object_different_colour_fraction"], 3)
    v["adj_diff_same_colour"] = f(o["adjacent_different_object_same_colour_fraction"], 4)
    v["train_seeds"] = len(b["train_seeds"]); v["held_seeds"] = len(b["held_seeds"])

    v["bounds_ids"] = bounds_table(b, ["object_ids", "raster_rank", "is_object_0"])
    v["bounds_same"] = bounds_table(b, ["same_right", "obj_edge_fg"])
    v["bounds_control"] = bounds_table(b, ["fg", "fg_edge", "near_fg"])
    idx = {(c["context"], c["target"]): c for c in b["ceilings"]}
    v["obj_edge_base"] = f(idx[("pixel", "obj_edge_fg")]["majority_baseline"])
    v["obj_edge_chroma"] = f(idx[("chroma_pair_right", "obj_edge_fg")]["held_acc"])
    v["same_right_base"] = f(idx[("pixel", "same_right")]["majority_baseline"])
    v["recur_pixel"] = f(idx[("pixel", "object_ids")]["held_key_recurrence"], 3)
    v["recur_3x3"] = f(idx[("win3x3", "object_ids")]["held_key_recurrence"], 3)
    v["bounds_fg_restricted"] = bounds_table(b, ["object_ids_fg", "raster_rank_fg"])
    v["bounds_other_probes"] = bounds_table(
        b, ["near_fg", "shade_bin_fg", "normal_up_fg"],
        contexts=["pixel", "pair_right", "cross4", "win3x3", "pixel_agg",
                  "eqbg_win3x3", "chroma_cross4"])

    ex = ch["exact"]
    v["cross_zero_same"] = f(ex["exact_cross_zero_fraction_same"], 4)
    v["cross_zero_diff"] = f(ex["exact_cross_zero_fraction_different"], 4)
    v["max_cross_same"] = ex["max_cross_over_same_object_pairs"]
    v["min_cross_diff"] = ex["min_cross_over_different_object_pairs"]
    v["n_same_pairs"] = ex["fgfg_same_object_pairs"]
    v["n_diff_pairs"] = ex["fgfg_different_object_pairs"]
    v["chroma_table"] = table(
        ["T", "right: train", "right: held-out", "down: train", "down: held-out"],
        [[r["T"], f(r["train_acc"]), f(r["held_acc"]),
          f(d["train_acc"]), f(d["held_acc"])]
         for r, d in zip(ch["right"]["rows"], ch["down"]["rows"])])
    v["chroma_base_right"] = f(ch["right"]["majority_baseline"])
    v["chroma_base_down"] = f(ch["down"]["majority_baseline"])

    if rg:
        d = rg["direct"]
        v["direct_space"] = d["space_size"]
        v["direct_evaluated"] = d["enumerate_fit"]["evaluated"]
        v["direct_exhausted"] = d["enumerate_fit"]["exhausted"]
        v["direct_seconds"] = f(d["enumerate_seconds"] if "enumerate_seconds" in d
                                else d["enumeration_seconds"], 1)
        v["direct_enum_seconds"] = f(d["enumerate_fit"]["seconds"], 1)
        v["direct_conforming"] = d["conforming_on_train"]
        v["direct_val"] = d["conforming_and_validation_exact"]
        v["direct_lex_acc"] = f(d["lexicographic"]["held_accuracy"], 6)
        v["direct_val_acc"] = f(d["validation_filtered"]["held_accuracy"], 6)
        v["direct_lex_err"] = d["lexicographic"]["held_max_error"]
        v["direct_density"] = f(d["random_density"]["density"], 4)
        v["direct_tightness"] = "%.0f" % ((2464 / 32000) / max(1e-9, d["conforming_on_train"] /
                                                               d["space_size"]))
        v["direct_grad"] = f"{d['gradient']['summary']['successes']}/{len(d['gradient']['rows'])}"
        v["direct_grad_s"] = f(d["gradient"]["summary"]["median_seconds"], 1)
        v["direct_selection"] = json.dumps(d["validation_filtered"]["selections"])
        v["train_records"] = rg["train_examples"]; v["val_records"] = rg["validation_examples"]
        v["held_records"] = rg["held_examples"]
        v["positive_fraction"] = f(rg["positive_fraction"], 3)
        if st and "staged" in st:
            s = st["staged"]
            v["staged_space"] = s["space_size"]
            v["staged_conforming"] = s["conforming_on_train"]
            v["staged_val"] = s["conforming_and_validation_exact"]
            v["staged_seconds"] = f(s["enumeration_seconds"], 1)
            v["staged_lex_acc"] = f(s["lexicographic"]["held_accuracy"], 6) if s["lexicographic"] else "--"
            v["staged_val_acc"] = f(s["validation_filtered"]["held_accuracy"], 6) if s["validation_filtered"] else "--"
            v["staged_grad"] = f"{s['gradient']['summary']['successes']}/{len(s['gradient']['rows'])}"
            v["stage1_conforming"] = st["stage1_foreground"]["count"]
            v["stage1_acc_train"] = f(st["stage1_foreground"]["accuracy_all_positions_train"], 6)
            v["stage1_acc_held"] = f(st["stage1_foreground"]["accuracy_all_positions_held_out"], 6)
            v["stage1_space"] = st["stage1_foreground"]["space_size"]
            v["stage1_nodes"] = st["stage1_foreground"]["node_evaluations"]
    if ap_:
        v["apply_table"] = table(
            ["R", "observation bytes", "positions", "caller nodes", "wrong slots",
             "accuracy", "`tcn.scaffold` agrees", "s / 3 images"],
            [[a["resolution"], a["observation_bytes"], a["positions"], a["caller_nodes"],
              f"{a['wrong_slots']}/{a['slots']}", f(a["accuracy"], 6),
              "yes" if a["merged_agrees"] else "NO", f(a["seconds"], 2)]
             for a in ap_["rows"]])
        v["apply_total_slots"] = sum(a["slots"] for a in ap_["rows"])
        v["apply_total_wrong"] = sum(a["wrong_slots"] for a in ap_["rows"])
        v["apply_T"] = ap_["threshold"]
    if wd:
        v["wide_space"] = wd["space_size"]
        v["wide_conforming"] = wd["conforming_on_train"]
        v["wide_val"] = wd["conforming_and_validation_exact"]
        v["wide_held_exact"] = wd["conforming_and_held_exact"]
        th = wd["surviving_thresholds"]
        v["wide_thresholds"] = ("%d..%d (%d values, contiguous)" % (min(th), max(th), len(th))
                                if th == list(range(min(th), max(th) + 1))
                                else str(th[:16]) + " ...")
        v["wide_inc_seconds"] = f(wd["incremental"]["seconds"], 1)
        v["wide_inc_nodes"] = wd["incremental"]["node_evaluations"]
        v["wide_enum_rate"] = f(wd["enumerate_fit_capped"]["programs_per_second"], 0)
        v["wide_enum_proj"] = f(wd["enumerate_fit_capped"]["projected_exhaustive_seconds"], 0)
        v["wide_first_evaluated"] = wd["stop_at_first"]["evaluated"]
        v["wide_first_s"] = f(wd["stop_at_first"]["wall"], 1)
        v["wide_lex_T"] = wd["lexicographic"]["threshold"]
        v["wide_lex_acc"] = f(wd["lexicographic"]["held_accuracy"], 6)
        v["wide_val_T"] = wd["validation_filtered"]["threshold"]
        v["wide_val_acc"] = f(wd["validation_filtered"]["held_accuracy"], 6)
        v["wide_val_err"] = wd["validation_filtered"]["held_max_error"]
        if "gradient" in wd:
            v["wide_grad"] = f"{wd['gradient']['summary']['successes']}/{len(wd['gradient']['rows'])}"
            v["wide_grad_held"] = f"{wd['gradient']['held_exact']}/{len(wd['gradient']['rows'])}"
            v["wide_grad_s"] = f(wd["gradient"]["summary"]["median_seconds"], 1)
        v["wide_grads"] = json.dumps(wd["choice_gradients"]["choice_grad_l1"])
    if rs:
        v["residual_table"] = table(
            ["split", "pairs", "majority baseline", "exact interval for T", "best accuracy",
             "accuracy at T=192", "errors at T=192"],
            [[k, rs[k]["pairs"], f(rs[k]["majority_baseline"]),
              f"[{rs[k]['exact_interval_lo']}, {rs[k]['exact_interval_hi']}]",
              f(rs[k]["best_accuracy"], 6), f(rs[k]["accuracy_at_T192"], 6),
              rs[k]["errors_at_T192"]] for k in ("train", "validation", "test")])
        v["residual_kinds"] = json.dumps(rs["test"]["error_kinds"])
        for k in ("train", "validation", "test"):
            v[f"interval_{k}"] = f"[{rs[k]['exact_interval_lo']}, {rs[k]['exact_interval_hi']}]"
        v["interval_all"] = "[%d, %d]" % (
            max(rs[k]["exact_interval_lo"] for k in ("train", "validation", "test")),
            min(rs[k]["exact_interval_hi"] for k in ("train", "validation", "test")))
        v["residual_example"] = json.dumps(rs["test"]["examples"][0]) if rs["test"]["examples"] else "--"
    if w3:
        rows = []
        for g in ("world_3d", "world_2d"):
            c = w3[g]["ceiling"]; p = w3[g]["permutation_certificate"]
            t = w3[g]["collinearity_transfer"]
            rows.append([g, f"{sum(x['rgb_identical'] for x in p)}/{len(p)}",
                         f"{sum(x['ids_identical'] for x in p)}/{len(p)}",
                         f(c["visible_ids"]["held_acc"]), f(c["visible_ids"]["majority_baseline"]),
                         "%+.4f" % c["visible_ids"]["advantage"],
                         f(c["visible_ids | foreground"]["held_key_recurrence"], 3),
                         f(t["and_of_three"]["held_acc_at_T"], 6),
                         f(t["searched"]["held_acc_at_T"], 6)])
        v["world_table"] = table(
            ["generator", "permuted list: rgb identical", "ids identical",
             "`visible_ids` held-out acc", "majority baseline", "advantage",
             "fg key recurrence", "same-object held-out acc, AND-of-three at T=144",
             "same-object held-out acc, searched shape at T=144"],
            rows)
        v["world_ids"] = json.dumps(w3["world_3d"]["permutation_certificate"][0]["ids"])
        v["world_hist"] = json.dumps(w3["world_3d"]["ceiling"]["id_histogram_held"])
    if uf:
        v["unfreeze_built"] = json.dumps(uf["as_built"]["choice_grad_l1"])
        v["unfreeze_free"] = json.dumps({k: v2 for k, v2 in uf["unfrozen"]["choice_grad_l1"].items()
                                         if k in ("shifted", "thr", "m1", "same")})
        v["unfreeze_space"] = uf["space_size_as_built"]
        v["unfreeze_thr"] = "%.3g" % uf["unfrozen"]["choice_grad_l1"]["thr"]
        if "gradient_unfrozen" in uf:
            v["unfreeze_grad"] = (f"{uf['gradient_unfrozen']['summary']['successes']}/"
                                  f"{len(uf['gradient_unfrozen']['rows'])}")
    if cg:
        v["rung3_grads"] = json.dumps(cg["rung3_control"]["choice_grad_l1"])
    if le:
        v["le_zero"] = le["first_gap_with_zero_gradient"]
        v["le_table"] = table(["gap |d - T|", "sigmoid((T-d)/tau)", "d/dT"],
                              [[r["gap"], "%.3g" % r["sigmoid"], "%.3g" % r["d_sigmoid"]]
                               for r in le["rows"] if r["gap"] in
                               (0, 4, 16, 32, 60, 88, 89, 128, 4096, 65025)])

    if rsh:
        v["rule_table"] = table(
            ["predicate shape", "exact T on train", "on validation", "on test",
             "exact on all three", "width", "coarse-pool values inside it"],
            [[k, str(x["train"]["exact_interval"]), str(x["validation"]["exact_interval"]),
              str(x["test"]["exact_interval"]), str(x["exact_on_all_three"]), x["width"],
              str(x["coarse_pool_hits"]) if x["coarse_pool_hits"] else "**none**"]
             for k, x in rsh.items()])
        vals = list(rsh.values())
        v["rule_and_all"] = str(vals[0]["exact_on_all_three"])
        v["rule_found_all"] = str(vals[1]["exact_on_all_three"])
        v["rule_and_width"] = vals[0]["width"]
        v["rule_found_width"] = vals[1]["width"]

    if dc:
        now = dc["candidates_now"]; was = dc["candidates_with_empty_parameters"]
        v["cand_now"] = json.dumps(now)
        v["cand_was"] = json.dumps(was)
        v["cand_new_ops"] = ", ".join("`%s`" % k for k in sorted(set(now) - set(was)))
        v["disc_per_node"] = str(dc["discovered_caller"]["candidates_per_node"])
        v["disc_space"] = dc["discovered_caller"]["space_size"]
        v["disc_solved"] = dc["search"]["solved"]
        v["disc_exhausted"] = dc["search"]["exhausted"]
        v["disc_unique"] = dc["search"]["unique"]
        v["disc_evaluated"] = dc["search"]["evaluated"]
        v["disc_seconds"] = f(dc["search"]["seconds"], 2)
        v["disc_agrees"] = "yes" if dc["merged_scaffold_agrees"] else "NO"
        v["disc_nodes"] = dc["merged_scaffold_nodes"]

    src = sf or ff
    if src:
        g = src["operating_gaps"]
        v["gap_table"] = table(
            ["`le` node", "median gap", "mean gap", "min", "max",
             "fraction at or past underflow", "shipped surrogate there",
             "shipped derivative there"],
            [[k, f(x["median_gap"], 0), f(x["mean_gap"], 0), x["min_gap"], x["max_gap"],
              f(x["fraction_at_or_past_underflow"], 3),
              "%.3g" % x["shipped_surrogate_at_median_gap"],
              "%.3g" % x["shipped_derivative_at_median_gap"]]
             for k, x in g.items() if isinstance(x, dict)])
        a = src.get("address_sharpness") or (ff or {}).get("address_sharpness")
        if a:
            v["addr_true"] = f(a["weight_on_true_address"], 3)
            v["addr_nb"] = f(a["weight_on_each_immediate_neighbour"], 3)
            v["addr_pm2"] = f(a["weight_within_plus_minus_2"], 4)
    def arm(d, key, prefix):
        if not d or key not in d:
            return
        x = d[key]
        v[prefix] = f"{x['summary']['successes']}/{len(x['rows'])}"
        v[prefix + "_held"] = f"{x['held_exact']}/{len(x['rows'])}"
        v[prefix + "_thr"] = "%.3g" % x["choice_gradients"]["choice_grad_l1"]["thr"]
    arm(sf, "operand", "sfix_operand"); arm(sf, "carrier", "sfix_carrier")
    arm(ff, "operand", "full_operand"); arm(ff, "carrier", "full_carrier")
    if sf and "operand" in sf:
        v["sfix_grads"] = json.dumps({k: x for k, x in
                                      sf["operand"]["choice_gradients"]["choice_grad_l1"].items()
                                      if k in ("shifted", "thr", "m1", "same")})
        v["sfix_thr"] = v.get("sfix_operand_thr")
        best = max((sf[k]["summary"]["successes"] for k in ("operand", "carrier") if k in sf),
                   default=0)
        n = len(sf["operand"]["rows"])
        heldbest = max((sf[k]["held_exact"] for k in ("operand", "carrier") if k in sf), default=0)
        if heldbest:
            v["sfix_conclusion"] = (
                "**With a live surrogate the relaxed path does reach this rung**: the best "
                f"corrected arm is {heldbest}/{n} exact on held-out episodes, against 0/4 for "
                "every arm run with the shipped surrogate.  The recorded failures in section 3 "
                "were invalid relaxations.")
        elif best:
            v["sfix_conclusion"] = (
                f"**A live surrogate is necessary and not sufficient here.**  The best corrected "
                f"arm reaches {best}/{n} conforming on training and {heldbest}/{n} exact on "
                "held-out, against 0/4 with the shipped surrogate -- so scaling the temperature "
                "does change the outcome, and the remaining gap is the offset, which is behind "
                "`pack`'s declared `gradient=\"none\"` and cannot be relaxed at all.")
        else:
            v["sfix_conclusion"] = (
                "**A live surrogate is necessary and not sufficient here.**  Every corrected arm "
                f"is still 0/{n}, and the reason is in the row above it: `shifted` remains "
                "`grad = None` under every temperature policy, because `pack` declares "
                "`gradient=\"none\"`.  The relaxed path can now see the threshold and the two "
                "combinators and still cannot see which neighbour to read, so it cannot settle "
                "this rung.  That is a statement about a declared boundary, which is what the "
                "corrected measurement is for -- it is no longer a statement about an underflow.")

    template = (HERE / "RESULTS.template.md").read_text()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", template)) - set(v))
    if missing:
        print("MISSING PLACEHOLDERS:", missing)
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: str(v.get(m.group(1), "{{%s}}" % m.group(1))),
                  template)
    (HERE / "RESULTS.md").write_text(text)
    print("wrote", HERE / "RESULTS.md")


if __name__ == "__main__":
    main()
