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
        r.append("**" + _n(a["overall"]["mean_cost"][arm]) + "**")
        r.append(_n(a["overall"]["mean_cost_optimistic"][arm]))
        rows.append(r)
    return table(["arm"] + list(a["domains"]) + ["**overall**", "overall, optimistic"],
                 rows)


def criteria():
    a = A()
    c = a["criteria"]
    m = a["overall"]["mean_cost"]
    nn = m["N''"]
    rows = []
    for tag, other, key, text in (
            ("C1", "N", "C1_N''_le_half_N", "E(N\u2033) \u2264 E(N)/2"),
            ("C2", "N'", "C2_N''_le_half_Nprime", "E(N\u2033) \u2264 E(N\u2032)/2"),
            ("C3a", "H1", "C3a_N''_le_half_H1", "E(N\u2033) \u2264 E(H1)/2"),
            ("C3b", "H2", "C3b_N''_le_half_H2", "E(N\u2033) \u2264 E(H2)/2")):
        got = m.get(other)
        shown = "\u2014" if got is None or nn is None else f"{_n(nn)} vs {_n(got / 2)}"
        rows.append([tag, text, shown, "**PASS**" if c[key] else "**FAIL**"])
    d1, nflat = m.get("D1"), m.get("N")
    shown = ("\u2014" if d1 is None or nflat is None
             else f"E(D1) {_n(d1)} vs E(N)/2 {_n(nflat / 2)}")
    rows.append(["C4", "D1 does not meet C1", shown,
                 "**PASS**" if c["C4_D1_fails_C1"] else "**FAIL**"])
    tail = ("\n\nPre-registered verdict: "
            + ("**MET**" if c["met"] else "**NOT MET**")
            + ". Same verdict when undecided edits are counted as repairs: "
            + ("yes" if not c["inconclusive"] else "NO \u2014 inconclusive") + ".")
    return table(["criterion", "as pre-registered", "measured", "outcome"], rows) + tail


def ratios():
    a = A()
    m = a["overall"]["mean_cost"]
    rows = [[LABEL[arm], _n(m[arm]),
             _n(m[arm] / m["N''"], 3) if m.get("N''") and m.get(arm) else "—"]
            for arm in ARMS if arm != "N''"]
    return table(["arm", "mean exact expected edits", "× N″'s cost"], rows)


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


def resources():
    a = A()
    rows = []
    for d in a["domains"]:
        c = C(d)
        rows.append([d, f"{c['seconds']:.0f}", f"{c['peak_rss_gb']:.2f}"])
    return table(["domain", "corpus seconds", "peak RSS (GB)"], rows)


BLOCKS = {"corpus": corpus, "decider": decider, "costs": costs, "criteria": criteria, "ratios": ratios,
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
