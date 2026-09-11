"""Render every figure in RESULTS.md from `out/`.  No number is typed by hand.

`BLOCKS[name]()` returns the exact text between `<!-- BEGIN:name -->` and
`<!-- END:name -->` in RESULTS.md; `verify.py` re-renders each and compares.

    python report.py            # print every block
    python report.py <name>     # print one
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction

import kit

ARMS = ["N", "N'", "N''", "H1", "H2", "D1", "D2", "ORACLE"]
LABEL = {"N": "N — no library", "N'": "N′ — edit-family class only",
         "N''": "N″ — learned cross-domain edit prior",
         "H1": "H1 — hand: smallest and newest first",
         "H2": "H2 — hand: output-adjacent first",
         "D1": "D1 — distractor, permuted labels",
         "D2": "D2 — distractor, reversed prior",
         "ORACLE": "ORACLE — perfect ordering"}


def A():
    return kit.load("analysis")


def C(domain):
    return kit.load(f"cases_{domain}")


def _n(x, nd=2):
    return "—" if x is None else f"{x:,.{nd}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------- blocks

def corpus():
    a = A()
    rows = []
    for d in a["domains"]:
        c = C(d)
        pd = a["per_domain"].get(d, {})
        rows.append([d, c["note"], c["n_train"], c["n_heldout"],
                     c["base"]["nodes"], f"{c['base']['space']:,}",
                     c["defects_tried"], c["admitted"],
                     c["rejected_solvable_on_train"], c["rejected_no_repair"],
                     c["rejected_invalid"],
                     _n(pd.get("n_edits_mean"), 1), _n(pd.get("n_repairs_mean"), 1),
                     "yes" if c.get("complete") else
                     "**no — stopped early, cases so far**"])
    return table(["domain", "task", "train", "held out", "nodes", "base space",
                  "defects tried", "**admitted**", "rejected: solvable on train",
                  "rejected: no repair", "rejected: invalid",
                  "edits/case", "repairs/case", "defect list exhausted"], rows)


def costs():
    a = A()
    rows = []
    for arm in ARMS:
        r = [LABEL[arm]]
        for d in a["domains"]:
            r.append(_n(a["per_domain"].get(d, {}).get("mean_cost", {}).get(arm)))
        r.append("**" + _n((a["overall"].get("mean_cost_macro")
                            or a["overall"]["mean_cost"])[arm]) + "**")
        r.append(_n(a["overall"]["mean_cost"][arm]))
        r.append(_n((a["overall"].get("mean_cost_macro_optimistic")
                     or a["overall"]["mean_cost_optimistic"])[arm]))
        rows.append(r)
    return table(["arm"] + list(a["domains"]) +
                 ["**pooled (macro)**", "per-case (micro)", "macro, optimistic"],
                 rows)


CRIT = (("C1", "N", "C1_N''_le_half_N", "E(N″) ≤ E(N)/2"),
        ("C2", "N'", "C2_N''_le_half_Nprime", "E(N″) ≤ E(N′)/2"),
        ("C3a", "H1", "C3a_N''_le_half_H1", "E(N″) ≤ E(H1)/2"),
        ("C3b", "H2", "C3b_N''_le_half_H2", "E(N″) ≤ E(H2)/2"))


def _verdict(c, key):
    v = (c or {}).get(key)
    return "**PASS**" if v is True else ("**FAIL**" if v is False else "—")


def _dom_crit(a, dom):
    return ((a.get("criteria_by_domain", {}).get(dom) or {}).get("criteria")) or {}


def criteria():
    """A12: per-domain verdicts first; a pooled figure only beside them."""
    a = A()
    rows = []
    for tag, _other, key, text in CRIT:
        rows.append([tag, text] +
                    [_verdict(_dom_crit(a, d), key) for d in a["domains"]] +
                    [_verdict(a["criteria"], key)])
    rows.append(["C4", "D1 does not meet C1"] +
                [_verdict(_dom_crit(a, d), "C4_D1_fails_C1") for d in a["domains"]] +
                [_verdict(a["criteria"], "C4_D1_fails_C1")])
    rows.append(["**verdict**", "all of C1-C4"] +
                ["**MET**" if _dom_crit(a, d).get("met") else "**NOT MET**"
                 for d in a["domains"]] +
                ["**MET**" if a["criteria"].get("met_all_domains") else "**NOT MET**"])
    head = ["criterion", "as pre-registered"] + list(a["domains"]) + ["pooled (macro)"]

    c = a["criteria"]
    tail = ["", "Pre-registered verdict: "
            + ("**MET**" if c.get("met_all_domains") else "**NOT MET**")
            + ". Per domain: "
            + ", ".join(f"`{k}` {'met' if v else 'not met'}"
                        for k, v in sorted(c.get("met_per_domain", {}).items()))
            + "."]
    if c.get("split_verdict"):
        tail += ["", "**This is a split verdict.** Under amendment A12(4) a split "
                 "is **NOT MET**: the criterion is about transfer to a genuinely "
                 "new domain, and a prior that helps only where it happens to fit "
                 "has not transferred. What it is evidence *for* is read off the "
                 "site structure below, not presented as a qualified success."]
    tail += ["", "Same verdict when undecided edits are counted as repairs: "
             + ("yes" if not c.get("inconclusive") else "NO — inconclusive") + "."]
    return table(head, rows) + "\n".join(tail)


def sites():
    """A12(3): the site structure, disclosed before any verdict is read."""
    rows = []
    for dom in present_domains():
        c = C(dom)
        adm = [x for x in c["cases"] if x.get("admitted")]
        if not adm:
            rows.append([dom, 0, 0, "—", "—", "—"])
            continue
        per_site, per_kind = {}, {}
        for x in adm:
            per_site[x["defect"]["site"]] = per_site.get(x["defect"]["site"], 0) + 1
            per_kind[x["defect"]["kind"]] = per_kind.get(x["defect"]["kind"], 0) + 1
        reps = [x["n_repairs"] for x in adm]
        rows.append([dom, len(adm), f"**{len(per_site)}**",
                     ", ".join(f"`{k}` {v}" for k, v in sorted(per_site.items())),
                     ", ".join(f"{k} {v}" for k, v in sorted(per_kind.items())),
                     f"{min(reps)}–{max(reps)} (mean {sum(reps) / len(reps):.1f})"])
    return table(["domain", "admitted cases", "distinct defect sites",
                  "defects per site", "defect kinds", "repairs per case"], rows)


def ratios():
    a = A()
    m = a["overall"].get("mean_cost_macro") or a["overall"]["mean_cost"]
    rows = [[LABEL[arm], _n(m[arm]),
             _n(m[arm] / m["N''"], 3) if m.get("N''") and m.get(arm) else "—"]
            for arm in ARMS if arm != "N''"]
    return table(["arm", "pooled (macro) mean exact expected edits",
                  "× N″'s cost"], rows)


def deployed():
    a = A()
    n = a["overall"]["deployed_of"]
    rows = [[LABEL[arm], f"{a['overall']['deployed_repairs'][arm]} / {n}"]
            for arm in ARMS]
    return table(["arm", "M2: first training-conforming edit is a repair"], rows)


def validity():
    a = A()
    v = a["validity"]
    rows = [["V1 — repairs that are the defect's syntactic inverse (a "
             "deliberately generous test: for `keep_prefix` any widening at the "
             "same site counts, for `delete_node` any node addition)",
             f"{v['V1_inverse_fraction_overall']:.4f} of all repairs"],
            ["V3 — the distractor's edit space contains the repairs",
             "yes, identical list; " + v["V3_note"]],
            ["undecided edits (space over the declared cap)",
             ", ".join(f"{d}: {a['per_domain'][d]['undecided_fraction']:.4f}"
                       for d in a["domains"] if d in a["per_domain"])]]
    if a["cases"]:
        per = [float(Fraction(r["D1_mean"])) for r in a["cases"]]
        lo = [min(float(Fraction(x)) for x in r["D1_permutations"]) for r in a["cases"]]
        hi = [max(float(Fraction(x)) for x in r["D1_permutations"]) for r in a["cases"]]
        rows.append([
            "D1 over the 20 label permutations (the headline D1 is the seed-0 one)",
            f"mean of per-case means {_n(sum(per) / len(per))}; "
            f"per-case range {_n(sum(lo) / len(lo))} to {_n(sum(hi) / len(hi))}"])
    return table(["check", "result"], rows)


def estimator():
    a = A()
    rows = []
    for held, e in sorted(a["estimators"].items()):
        if "error" in e:
            rows.append([held, "—", "—", "—", e["error"]])
            continue
        top = ", ".join(f"`{k}` {v:+.2f}" for k, v in e["top_features"][:5])
        rows.append([held, "+".join(e["sources"]), e["n_pos"], e["n_neg"],
                     ("flat" if e["flat"] else top)])
    return table(["held-out domain", "fitted on", "repair rows", "non-repair rows",
                  "strongest features"], rows)


def families():
    a = A()
    rows = []
    for held, e in sorted(a["estimators"].items()):
        if "error" in e:
            continue
        rows.append([held, ", ".join(e["family_class"]) or "—"])
    obs = {}
    for r in a["cases"]:
        for f in r["repair_families"]:
            obs[f] = obs.get(f, 0) + 1
    rows.append(["**repairs observed, by family**",
                 ", ".join(f"{k} ({v} cases)" for k, v in sorted(obs.items()))])
    return table(["held-out domain", "inherited edit-family class"], rows)


def s50():
    try:
        d = kit.load("s50_rescore")
    except FileNotFoundError:
        return "S1 was not run."
    s = d["summary"]

    def n_tr(c):
        return [r["variant"] for r in c["rows"] if r["conforming_on_train"]]

    def n_al(c):
        return [r["variant"] for r in c["rows"] if r["conforming_on_all"]]

    variants = sum(len(c["rows"]) for c in d["cases"])
    tr = sum(len(n_tr(c)) for c in d["cases"])
    al = sum(len(n_al(c)) for c in d["cases"])
    lost = [c for c in d["cases"] if len(n_tr(c)) != len(n_al(c))]
    rows = [["failed scaffolds re-decided", s["failed_cases"]],
            ["§50's own count, training conformance (recorded)", "21 / 24"],
            ["cases keeping a repair under the stricter metric",
             f"{s['repair_exists_heldout']} / {s['failed_cases']}"],
            ["the training-stop variant also conforms on all episodes",
             f"{s['training_stop_conforms_heldout']} / {s['failed_cases']}"],
            ["of those, the language cases (the only ones with a held-out split)",
             f"{s['lang_training_stop_conforms_heldout']} / {s['lang_cases']}"],
            ["the Boolean cases, which have no held-out split",
             f"{s['bool_cases']}, both with no conforming variant at all"],
            ["variants enumerated across every case", variants],
            ["variants conforming on training", tr],
            ["variants conforming on training **and** on the held-out episodes", al],
            ["**variants lost to the stricter metric**", tr - al],
            ["cases that lose a variant", f"{len(lost)} / {s['failed_cases']}"]]
    for c in lost:
        a, b = n_tr(c), n_al(c)
        rows.append([f"— `{c['case_id']}`, train-conforming → all-conforming",
                     f"{len(a)} (`{'`, `'.join(a)}`) → "
                     f"{len(b)} (`{'`, `'.join(b)}`), out of {len(c['rows'])} variants"])
        for r in c["rows"]:
            if r["conforming_on_train"]:
                rows.append([
                    f"— `{c['case_id']}` variant `{r['variant']}`: family members "
                    f"conforming on training → on training and held out",
                    f"{r['conforming_on_train']} → {r['conforming_on_all']}"])
    return table(["S1 — §50's corpus on held-out conformance", "count"], rows)


def sweep():
    d = kit.load("episode_sweep")
    rows = []
    for dom in sorted(d["domains"]):
        info = d["domains"][dom]
        for r in info["rows"]:
            mark = " **<- chosen**" if r["budget"] == info["chosen_budget"] else ""
            rows.append([dom, r["budget"], r["n_train"], r["n_heldout"],
                         r["defects"], r["invalid"], r["solvable_on_train"],
                         f"**{r['admissible']}**{mark}"])
    out = table(["domain", "budget", "train", "held out", "defects", "invalid",
                 "solvable on train", "admissible"], rows)
    notes = []
    for dom in sorted(d["domains"]):
        notes.append(f"- `{dom}`: {d['domains'][dom]['reason']}")
    return out + "\n\nRule: " + d["rule"] + ".\n" + "\n".join(notes)


DOMAINS = ("arith", "bool", "rel")


def present_domains():
    return [d for d in DOMAINS if (kit.OUT / f"cases_{d}.json").exists()]


def ceilings():
    rows = []
    for dom in present_domains():
        c = C(dom)
        adm = c["admitted"]
        rows.append([dom, c["defects_tried"], c["rejected_invalid"],
                     c["rejected_solvable_on_train"],
                     c["rejected_invalid"] + c["rejected_solvable_on_train"],
                     c["defects_tried"] - c["rejected_invalid"]
                     - c["rejected_solvable_on_train"],
                     c["rejected_no_repair"], f"**{adm}**"])
    return table(["domain", "defects enumerated", "invalid",
                  "solvable on train", "rejected before the edit sweep",
                  "admissible", "no repair in the edit space",
                  "admitted cases"], rows)


def cost():
    rows = []
    for name, label in (("rate_bool_10", "`bool` `keep_prefix:y:8`"),
                        ("rate_arith_0", "`arith` `delete_node:inner`")):
        try:
            v = kit.load(name)
        except FileNotFoundError:
            continue
        rows.append([label, v["episodes_all"], v["n_edits"], v["over_cap"],
                     v["mean_seconds_per_edit"], v["worst_seconds_per_edit"],
                     v["projected_minutes_per_case"]])
    return table(["case", "episodes", "edits", "over cap",
                  "mean s/edit", "worst s/edit", "**projected min/case**"], rows)


def decider():
    rows = []
    for d in A()["domains"]:
        try:
            v = kit.load(f"validate_{d}")
        except FileNotFoundError:
            rows.append([d, "not run", "—", "—", "—"])
            continue
        rows.append([d, v["fit_checked"], len(v["fit_mismatches"]),
                     f"{v['witnesses_checked']} ({v['episode_evaluations']:,} "
                     f"episode evaluations)", len(v["witness_mismatches"])])
    return table(["domain", "edits re-decided by the flat walk",
                  "disagreements", "witnesses re-executed through `Program.execute`",
                  "witnesses that did not reproduce"], rows)


def percase():
    a = A()
    rows = []
    for r in a["cases"]:
        rows.append([r["case_id"], r["n_edits"], r["n_repairs"], r["n_undecided"]] +
                    [_n(float(Fraction(r["cost"][x])) if r["cost"][x] else None, 1)
                     for x in ARMS])
    return table(["case", "edits", "repairs", "undecided"] +
                 [x for x in ARMS], rows)


def pressure():
    h = kit.load("host_pressure")
    m = kit.load("mem_arith")
    peak = max(m["peak_rss_gb"].values())
    rows = [["measured peak RSS of the heaviest phase (build a domain, enumerate "
             "a case's typed edits, decide them)", f"{peak:.3f} GB"],
            ["candidates held at once across every edited program of one case",
             f"{m['total_candidates_held']:,}"],
            ["this track's live workers, combined", f"{h['this_track_total_gb']:.2f} GB"],
            ["two shards, projected", f"{h['four_shard_projected_total_gb']:.2f} GB"],
            ["`MemAvailable` before the corpus phase",
             f"{h['mem_available_gb_before']:.1f} GB"],
            ["`MemAvailable` when the flat floor refused the launch",
             f"{h['mem_available_gb_at_breach']:.1f} GB"]]
    for k, v in h["this_track_workers_gb"].items():
        rows.append([f"this track: {k}", f"{v:.2f} GB"])
    for k, v in h["other_consumers_gb"].items():
        rows.append([f"other project: {k}", f"{v:.1f} GB"])
    return table(["host memory at the launch decision", "measured"], rows)


def gate():
    log = kit.OUT / "resource_gate.log"
    if not log.exists():
        return "No resource-gate decisions recorded."
    lines = [x for x in log.read_text().splitlines() if x.strip()]
    rows = [["decisions recorded", len(lines)],
            ["phases admitted", sum(1 for x in lines if x.endswith("START"))],
            ["phases refused by a gate", sum(1 for x in lines if x.endswith("REFUSED"))],
            ["launches held by hand, before the gates existed",
             sum(1 for x in lines if "HELD" in x)]]
    for x in lines:
        if x.endswith("REFUSED"):
            rows.append(["a refusal, verbatim", "`" + x.replace("\t", " | ") + "`"])
    return table(["`out/resource_gate.log`", "count"], rows)


def outerloop():
    """The cost of the outer loop itself, so 'moving combinatorics up a level'
    can be checked rather than assumed."""
    rows = []
    for dom in present_domains():
        c = C(dom)
        adm = [x for x in c["cases"] if x.get("admitted")]
        if not adm:
            continue
        proposed = sum(x["n_edits"] for x in adm)
        undec = sum(x["n_undecided"] for x in adm)
        decided = proposed - undec
        space = sum(e["space"] for x in adm for e in x["edits"] if e["decided"])
        reps = [e["space"] for x in adm for e in x["edits"] if e["repair"]]
        eps = c["n_train"] + c["n_admission"]
        rows.append([
            dom, len(adm), f"{proposed:,}", f"{decided:,}", f"{undec:,}",
            f"{space:,}", f"{eps}", f"{space * eps:,}",
            f"{c['seconds'] / 60:.0f} min",
            f"{(sum(reps) / len(reps)):,.0f}" if reps else "—"])
    out = table(["domain", "cases", "edits proposed", "edits decided",
                 "undecided (over cap)",
                 "selection space of the decided edits (upper bound on programs)",
                 "episodes per decision",
                 "episode-evaluations (upper bound)", "corpus wall clock",
                 "mean space of a repaired scaffold (downstream search)"], rows)
    return out + (
        "\n\nThe program and episode-evaluation columns are **upper bounds**, not "
        "counts: `enumerate_prefix` stops at the first conforming member and "
        "prunes a prefix as soon as a probed node misses, so a decided edit "
        "usually costs far less than its selection space. This corpus does not "
        "carry the exact `evaluated` figure — `decide()` did not record it, and "
        "adding the field mid-run would have made shards of one domain "
        "inconsistent with each other. The bound is reported as a bound.")


def resources():
    a = A()
    rows = []
    for d in a["domains"]:
        c = C(d)
        rows.append([d, f"{c['seconds']:.0f}", f"{c['peak_rss_gb']:.2f}"])
    return table(["domain", "corpus seconds", "peak RSS (GB)"], rows)


BLOCKS = {"corpus": corpus, "decider": decider, "sweep": sweep,
          "ceilings": ceilings, "cost": cost, "sites": sites,
          "outerloop": outerloop, "pressure": pressure, "gate": gate, "costs": costs, "criteria": criteria, "ratios": ratios,
          "deployed": deployed, "validity": validity, "estimator": estimator,
          "families": families, "s50": s50, "percase": percase,
          "resources": resources}


def fill(path):
    """Rewrite every block in RESULTS.md in place from `out/`."""
    import re
    text = path.read_text()
    def sub(m):
        name = m.group(1)
        if name not in BLOCKS:
            raise KeyError(f"RESULTS.md asks for an unknown block: {name}")
        return f"<!-- BEGIN:{name} -->\n{BLOCKS[name]()}\n<!-- END:{name} -->"
    out = re.sub(r"<!-- BEGIN:(\w+) -->\n?.*?\n?<!-- END:\1 -->", sub, text, flags=re.S)
    path.write_text(out)
    print(f"filled {len(re.findall(r'<!-- BEGIN:', out))} blocks in {path}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fill"]:
        fill(kit.HERE / "RESULTS.md")
    else:
        for n in (sys.argv[1:] or list(BLOCKS)):
            print(f"<!-- BEGIN:{n} -->")
            print(BLOCKS[n]())
            print(f"<!-- END:{n} -->\n")
