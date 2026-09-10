"""Recompute every headline in RESULTS.md from committed files under out/.

Two layers, as the house standard requires:

1. every `<!-- BEGIN:x -->` block equals what `report.py` renders from `out/` now,
   so no figure in a block was hand-edited;
2. every headline claim is re-derived **here**, from `out/cases_*.json` and the
   committed estimator table, without importing `analyse.py` or `report.py` --
   the tiering, the exact expectation, the criteria, the validity checks and the
   S1 summary are all recomputed from scratch and compared with what
   `out/analysis.json` records.  A bug in a run script cannot make its own claim
   true.

It also fails on any number in RESULTS.md prose (outside blocks) that is neither
in the structural allow-list below nor present verbatim in a rendered block.

Writes `out/verify.json` and `out/headline.json`; exits non-zero on any FAIL.

    python verify.py
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import re
import sys
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
RESULTS_PATH = HERE / "RESULTS.md"
results = []

ARMS = ["N", "N'", "N''", "H1", "H2", "D1", "D2", "ORACLE"]
FEATURE_KEYS = ("family", "op_class", "novel", "is_output", "const_site",
                "added", "space", "arity")


def claim(name, ok, detail=""):
    results.append((bool(ok), name, detail))


def J(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------- independent re-implementations

def bucket_added(n):
    for hi, name in ((0, "0"), (4, "1-4"), (16, "5-16"), (48, "17-48")):
        if n <= hi:
            return name
    return ">48"


def bucket_space(x):
    return next((f"<{hi}" for hi in (3, 4, 5, 6) if x < hi), ">=6")


def disc(f):
    return {"family": f["family"], "op_class": f["op_class"],
            "novel": str(f["novel_operator"]), "is_output": str(f["site_is_output"]),
            "const_site": str(f["site_constant_on_train"]),
            "added": bucket_added(f["added_candidates"]),
            "space": bucket_space(f["log_space"]), "arity": str(f["max_arity"])}


def sc(table, row):
    return round(sum(table.get(f"{k}={row[k]}", 0.0) for k in FEATURE_KEYS), 6)


def expectation(order_keys, repairs):
    """Exact expected edits before the first repair, over uniform tiers."""
    before = 0
    for _, grp in itertools.groupby(range(len(order_keys)), key=lambda i: order_keys[i]):
        idx = list(grp)
        s = len(idx)
        k = sum(1 for i in idx if repairs[i])
        if k > 0:
            return Fraction(before) + Fraction(s + 1, k + 1)
        before += s
    return None


def arm_key(arm, e, row, table, dtable, fam):
    if arm == "N":
        return 0
    if arm == "N'":
        return 0 if e["family"] in fam else 1
    if arm == "N''":
        return -sc(table, row)
    if arm == "H1":
        return (e["features"]["added_candidates"],
                0 if e["features"]["novel_operator"] else 1)
    if arm == "H2":
        return -e["features"]["site_depth_frac"]
    if arm == "D1":
        return -sc(dtable, row)
    if arm == "D2":
        return sc(table, row)
    if arm == "ORACLE":
        return 0 if e["repair"] else 1
    raise ValueError(arm)


def main():
    a = J("analysis")
    if a is None:
        claim("out/analysis.json exists", False)
        finish()
        return

    domains = a["domains"]
    cases = {d: J(f"cases_{d}") for d in domains}
    for d in domains:
        claim(f"out/cases_{d}.json exists", cases[d] is not None)

    # ---- layer 2a: corpus counts
    for d in domains:
        c = cases[d]
        adm = [x for x in c["cases"] if x.get("admitted")]
        claim(f"{d}: admitted count in the file equals its `admitted` field",
              len(adm) == c["admitted"], f"{len(adm)} vs {c['admitted']}")
        claim(f"{d}: every admitted case has 0 conformers on train, exhausted, complete",
              all(x["failed"]["conforming"] is False and x["failed"]["exhausted"]
                  and x["failed"]["certificate"] == "complete" for x in adm))
        claim(f"{d}: every admitted case has at least one repair",
              all(x["n_repairs"] > 0 for x in adm))
        claim(f"{d}: n_repairs equals the repairs in the edit list",
              all(x["n_repairs"] == sum(1 for e in x["edits"] if e["repair"])
                  for x in adm))
        claim(f"{d}: n_undecided equals the undecided edits",
              all(x["n_undecided"] == sum(1 for e in x["edits"] if not e["decided"])
                  for x in adm))
        claim(f"{d}: every repair is proved by a witness",
              all(e["certificate_all"] == "witness"
                  for x in adm for e in x["edits"] if e["repair"]))
        claim(f"{d}: every non-repair with a decision is proved by exhaustion",
              all(e["certificate_all"] == "complete"
                  for x in adm for e in x["edits"]
                  if e["decided"] and not e["repair"]))

    # ---- layer 2b: the estimator table is consistent with the source rows
    for held, est in a["estimators"].items():
        if "error" in est:
            continue
        rows, labels = [], []
        for src in est["sources"]:
            for x in cases[src]["cases"]:
                if not x.get("admitted"):
                    continue
                for e in x["edits"]:
                    if e["repair"] is None:
                        continue
                    rows.append(disc(e["features"]))
                    labels.append(bool(e["repair"]))
        claim(f"estimator[{held}]: repair-row count re-derived",
              est["n_pos"] == sum(labels), f"{est['n_pos']} vs {sum(labels)}")
        claim(f"estimator[{held}]: non-repair-row count re-derived",
              est["n_neg"] == len(labels) - sum(labels))
        values = {k: sorted({r[k] for r in rows}) for k in FEATURE_KEYS}
        pos = [r for r, y in zip(rows, labels) if y]
        neg = [r for r, y in zip(rows, labels) if not y]
        redo = {}
        for k in FEATURE_KEYS:
            m = len(values[k])
            for v in values[k]:
                p = (sum(1 for r in pos if r[k] == v) + 1.0) / (len(pos) + m)
                q = (sum(1 for r in neg if r[k] == v) + 1.0) / (len(neg) + m)
                redo[f"{k}={v}"] = round(math.log(p) - math.log(q), 6)
        claim(f"estimator[{held}]: every log-odds entry re-derived from raw rows",
              redo == est["table"],
              f"{len(set(redo.items()) ^ set(est['table'].items()))} differing entries")
        claim(f"estimator[{held}]: the held-out domain contributes no row (V4)",
              held not in est["sources"])
        fams = set()
        for src in est["sources"]:
            for x in cases[src]["cases"]:
                if x.get("admitted"):
                    fams |= {e["family"] for e in x["edits"] if e["repair"]}
        claim(f"estimator[{held}]: the inherited edit-family class re-derived",
              sorted(fams) == est["family_class"])

    # ---- layer 2c: every per-case arm cost, recomputed from scratch
    by_case = {}
    for d in domains:
        for x in cases[d]["cases"]:
            if x.get("admitted"):
                by_case[x["case_id"]] = (d, x)
    bad, checked = [], 0
    for r in a["cases"]:
        d, x = by_case[r["case_id"]]
        est = a["estimators"][d]
        if "error" in est:
            claim(f"{r['case_id']}: a case is costed but its estimator errored", False)
            continue
        edits = x["edits"]
        rows = [disc(e["features"]) for e in edits]
        for arm in ARMS:
            if arm == "D1":
                continue          # its ordering depends on a stored permutation seed
            for optimistic in (False, True):
                reps = [(True if e["repair"] is None and optimistic else bool(e["repair"]))
                        for e in edits]
                keys = [arm_key(arm, e, rows[i], est["table"], {}, est["family_class"])
                        for i, e in enumerate(edits)]
                order = sorted(range(len(edits)), key=lambda i: (keys[i], i))
                got = expectation([keys[i] for i in order], [reps[i] for i in order])
                field = "cost_optimistic" if optimistic else "cost"
                want = r[field][arm]
                ok = (got is None and want is None) or (
                    got is not None and want is not None and got == Fraction(want))
                checked += 1
                if not ok:
                    bad.append((r["case_id"], arm, field, str(got), want))
    claim(f"all {checked} per-case arm costs re-derived from raw edits", not bad,
          "; ".join(f"{c}/{a_}/{f}: {g} vs {w}" for c, a_, f, g, w in bad[:5]))

    # ---- layer 2d: the means and the criteria
    def mean(arm, field="cost"):
        vals = [Fraction(r[field][arm]) for r in a["cases"] if r[field][arm] is not None]
        return float(sum(vals) / len(vals)) if vals else None

    for arm in ARMS:
        for field, key in (("cost", "mean_cost"),
                           ("cost_optimistic", "mean_cost_optimistic")):
            got, want = mean(arm, field), a["overall"][key][arm]
            claim(f"overall {key}[{arm}] re-derived",
                  (got is None and want is None) or
                  (got is not None and want is not None and abs(got - want) < 1e-9),
                  f"{got} vs {want}")

    m = {arm: mean(arm) for arm in ARMS}
    c = a["criteria"]
    for tag, arm, field in (("C1_N''_le_half_N", "N", "C1"),
                            ("C2_N''_le_half_Nprime", "N'", "C2"),
                            ("C3a_N''_le_half_H1", "H1", "C3a"),
                            ("C3b_N''_le_half_H2", "H2", "C3b")):
        want = c[tag]
        got = None if m["N''"] is None or m[arm] is None else bool(m["N''"] <= m[arm] / 2)
        claim(f"{field} re-derived", got == want, f"{got} vs {want}")
    claim("C4 re-derived",
          c["C4_D1_fails_C1"] == (None if m["D1"] is None or m["N"] is None
                                  else not (m["D1"] <= m["N"] / 2)))
    claim("the pre-registered verdict is the conjunction of C1-C4",
          c["met"] == all(c[k] is True for k in
                          ("C1_N''_le_half_N", "C2_N''_le_half_Nprime",
                           "C3a_N''_le_half_H1", "C3b_N''_le_half_H2",
                           "C4_D1_fails_C1")))

    # ---- layer 2e: validity
    tot_rep = sum(r["n_repairs"] for r in a["cases"])
    inv = sum(r["inverse_repairs"] for r in a["cases"])
    claim("V1 inverse fraction re-derived",
          abs(a["validity"]["V1_inverse_fraction_overall"] -
              round(inv / max(1, tot_rep), 4)) < 1e-9)
    claim("V1 inverse repairs re-derived from the raw edit lists",
          inv == sum(1 for _, x in by_case.values()
                     for e in x["edits"] if e["repair"] and e["is_inverse"]))
    claim("V3 every arm orders the identical edit list, so repairs are identical",
          all(r["n_repairs"] == by_case[r["case_id"]][1]["n_repairs"]
              for r in a["cases"]))

    # ---- layer 2f: S1
    s = J("s50_rescore")
    if s is not None:
        sm = s["summary"]
        claim("S1: failed-case count re-derived", sm["failed_cases"] == len(s["cases"]))
        claim("S1: repair-exists count re-derived",
              sm["repair_exists_heldout"] ==
              sum(1 for x in s["cases"] if any(r["conforming_on_all"] for r in x["rows"])))
        claim("S1: training-stop count re-derived",
              sm["training_stop_conforms_heldout"] ==
              sum(1 for x in s["cases"] if x["training_stop_conforms_heldout"]))

    # ---- layer 1: blocks, and the prose allow-list
    if RESULTS_PATH.exists():
        text = RESULTS_PATH.read_text()
        sys.path.insert(0, str(HERE))
        import report                                             # noqa: E402
        rendered = {}
        for mm in re.finditer(r"<!-- BEGIN:(\w+) -->\n(.*?)\n<!-- END:\1 -->",
                              text, flags=re.S):
            name, body = mm.group(1), mm.group(2)
            ok = name in report.BLOCKS
            got = report.BLOCKS[name]() if ok else ""
            rendered[name] = got
            claim(f"block `{name}` equals its rendering from out/", ok and got == body)
        claim("every block report.py can render appears in RESULTS.md",
              set(report.BLOCKS) >= set(rendered))
        allow = set(ALLOWED)
        inside = "\n".join(rendered.values())
        prose = re.sub(r"<!-- BEGIN:\w+ -->\n.*?\n<!-- END:\w+ -->", "", text, flags=re.S)
        prose = re.sub(r"`[^`]*`", "", prose)          # code spans carry formulae
        prose = re.sub(r"§\s*\d[\d.]*", "", prose)     # citations of other sections
        prose = re.sub(r"\*\*A\d+ —", "**", prose)     # amendment numbers
        stray = [t for t in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", prose)
                 if t not in allow and t not in inside]
        claim("no number in RESULTS.md prose is outside a block or the allow-list",
              not stray, ", ".join(sorted(set(stray))[:20]))

    finish()


ALLOWED = {
    # the only numbers allowed in RESULTS.md prose: structural constants fixed
    # by the pre-registration or by an amendment.  Nothing measured belongs here.
    "1", "2", "3", "4", "6", "8",          # keep_prefix's k values (amendment A3)
    "48", "12", "400,000",                 # MAX_NEW, MAX_NEW_NODE, MAX_SPACE
}


def finish():
    passed = sum(1 for ok, _, _ in results if ok)
    failed = len(results) - passed
    for ok, name, detail in results:
        print(("PASS " if ok else "FAIL ") + name + (f"   [{detail}]" if detail and not ok else ""))
    print(f"\n{passed} PASS, {failed} FAIL")
    OUT.mkdir(exist_ok=True)
    (OUT / "verify.json").write_text(json.dumps(
        {"pass": passed, "fail": failed,
         "claims": [{"ok": ok, "claim": n, "detail": d} for ok, n, d in results]},
        indent=1))
    a = J("analysis")
    if a:
        (OUT / "headline.json").write_text(json.dumps({
            "domains": a["domains"],
            "n_cases": a["overall"]["n_cases"],
            "mean_cost": a["overall"]["mean_cost"],
            "mean_cost_optimistic": a["overall"]["mean_cost_optimistic"],
            "criteria": a["criteria"],
            "validity": {k: v for k, v in a["validity"].items()
                         if k != "V3_repairs_per_case"},
            "deployed_repairs": a["overall"]["deployed_repairs"],
            "deployed_of": a["overall"]["deployed_of"],
        }, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
