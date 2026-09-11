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
NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"


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


def check_expectation():
    """The closed form, checked against a route that shares no algebra with it.

    Re-implementing `(s + 1) / (k + 1)` here would only be the same derivation
    typed twice -- and the pre-registration's first draft of it was wrong, so
    that is exactly the check that failed.  Instead the tier is *simulated*:
    uniformly random permutations, averaging the position of the first repair.
    """
    import random as _r
    rng = _r.Random(12345)
    trials = 200_000
    for s, k in ((1, 1), (3, 1), (5, 1), (6, 2), (10, 3), (8, 8)):
        idx = list(range(s))
        total = 0
        for _ in range(trials):
            rng.shuffle(idx)
            total += next(i for i, v in enumerate(idx, start=1) if v < k)
        empirical = total / trials
        closed = Fraction(s + 1, k + 1)
        claim(f"tier (s={s}, k={k}): the closed form matches a simulated draw",
              abs(empirical - float(closed)) < 0.02,
              f"simulated {empirical:.4f} vs closed form {float(closed):.4f}")
    # degenerate cases, asserted outright
    for keys, reps, want in (
            ([0], [True], Fraction(1)),
            ([0, 0], [False, True], Fraction(3, 2)),
            ([0, 0, 0], [True, True, True], Fraction(1)),
            ([0, 1], [False, True], Fraction(2)),
            ([0], [False], None)):
        got = expectation(keys, reps)
        claim(f"degenerate tier {keys}/{reps} costs {want}", got == want,
              f"{got} vs {want}")


def main():
    check_expectation()
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
        # A corpus that was killed mid-run is a partial corpus.  It is allowed to
        # exist -- it is written deliberately so a kill costs one defect -- but it
        # must never silently become the corpus an arm is costed on.
        claim(f"{d}: the corpus was not stopped by a signal",
              c.get("stopped_by_signal") is None,
              f"stopped_by_signal={c.get('stopped_by_signal')}, "
              f"last defect started {c.get('last_defect_started')}")
        claim(f"{d}: the corpus records whether its defect list was exhausted",
              "complete" in c, "no `complete` field")

    # ---- layer 2a1: A13's blind-split guarantees.  A19 records that these
    # assertions were PROMISED by A13 before they were implemented; the claim
    # stood for several hours with nothing behind it.
    FEATURE_FIELDS = {"family", "added_candidates", "log_added", "log_space",
                      "op_class", "novel_operator", "site_depth_frac",
                      "site_is_output", "site_constant_on_train", "max_arity"}
    fps = {}
    for d in domains:
        c = cases[d]
        claim(f"A13[{d}]: the corpus declares the final split untouched",
              c.get("final_untouched") is True,
              f"final_untouched={c.get('final_untouched')}")
        claim(f"A13[{d}]: the corpus records a final-split fingerprint and size",
              bool(c.get("final_digest")) and isinstance(c.get("n_final"), int),
              f"digest={c.get('final_digest')} n_final={c.get('n_final')}")
        fps[d] = c.get("final_digest")
        # No ordering may be computed off anything but `train`: every feature an
        # arm orders by is stored per edit, and the stored field set is exactly
        # the declared one -- nothing derived from `admission` or `final`.
        adm = [x for x in c["cases"] if x.get("admitted")]
        extra = set()
        for x in adm:
            for e in x["edits"]:
                extra |= set(e["features"]) - FEATURE_FIELDS
        claim(f"A13[{d}]: no stored feature is derived from admission or final",
              not extra, f"unexpected feature fields: {sorted(extra)}")
        claim(f"A13[{d}]: the only episode-derived feature names the train split",
              all("site_constant_on_train" in e["features"]
                  for x in adm for e in x["edits"]))
    sc = J("final_scores")
    if sc is not None:
        for d in domains:
            rec = (sc.get("domains", {}).get(d) or {}).get("final_digest")
            claim(f"A13[{d}]: final scoring used the split the corpus withheld",
                  rec == fps[d], f"{rec} vs {fps[d]}")

    # ---- A21: every evidence artifact carries the five provenance identities,
    # and derived artifacts agree with what they were derived from.
    FIVE = ("base_digest", "split_digest", "grammar_digest", "git", "prereg")
    prov = {}
    for name in ("analysis", "final_scores", *[f"validate_{d}" for d in domains],
                 *[f"cases_{d}" for d in domains]):
        art = J(name)
        if art is None:
            continue
        got = art.get("provenance")
        if got is None and isinstance(art.get("domains"), dict):
            got = {k: v.get("provenance") for k, v in art["domains"].items()}
        if isinstance(got, dict) and set(got) & set(domains):
            for dom, pv in got.items():
                claim(f"A21[{name}/{dom}]: carries all five provenance identities",
                      isinstance(pv, dict) and all(f in pv for f in FIVE),
                      f"missing {sorted(set(FIVE) - set(pv or {}))}")
                prov.setdefault(dom, []).append((name, pv))
        else:
            rec = art.get("provenance_reconstructed")
            if got is None and isinstance(rec, dict):
                # A20/A21: a corpus built before A21 is not back-filled.  It
                # carries a reconstruction whose evidence is git, and the
                # verifier treats that as weaker than a build-time stamp and
                # says so rather than accepting it silently.
                claim(f"A21[{name}]: provenance is RECONSTRUCTED, not stamped "
                      f"at build time",
                      all(k in rec for k in
                          ("kind", "producing_files", "corpus_recorded_at",
                           "diff_since_corpus", "digests_now")),
                      "reconstruction is missing its evidence fields")
                changed = [f for f, v in (rec.get("diff_since_corpus") or {}).items()
                           if not isinstance(v, str)]
                claim(f"A21[{name}]: every changed producer since the corpus is "
                      f"disclosed with its diff",
                      all(isinstance((rec['diff_since_corpus'] or {})[f], dict)
                          and "diff" in rec["diff_since_corpus"][f]
                          for f in changed),
                      f"changed without a recorded diff: {changed}")
                prov.setdefault(name.replace("cases_", ""), []).append(
                    (name, rec.get("digests_now") or {}))
            else:
                claim(f"A21[{name}]: carries all five provenance identities",
                      isinstance(got, dict) and all(f in got for f in FIVE),
                      f"missing {sorted(set(FIVE) - set(got or {}))}"
                      if got is not None else "no provenance block "
                      "(artifact predates A21 and cannot be shown current)")
    for dom, entries in prov.items():
        grammars = {p.get("grammar_digest") for _, p in entries}
        claim(f"A21[{dom}]: every artifact used one mutation-grammar version",
              len(grammars) == 1, f"grammar digests differ: {grammars}")
        splits = {p.get("split_digest") for _, p in entries if p.get("split_digest")}
        claim(f"A21[{dom}]: every artifact refers to one blind split",
              len(splits) <= 1, f"split digests differ: {splits}")
        bases = {p.get("base_digest") for _, p in entries if p.get("base_digest")}
        claim(f"A21[{dom}]: every artifact refers to one base scaffold",
              len(bases) <= 1, f"base digests differ: {bases}")

    # ---- layer 2a2: the episode budgets are the ones the rule chose (A8)
    sw = J("episode_sweep")
    if sw is None:
        claim("out/episode_sweep.json exists (A8's evidence)", False)
    else:
        for d in domains:
            info = sw["domains"].get(d)
            if info is None:
                claim(f"A8[{d}]: the domain appears in the sweep", False)
                continue
            row = next((r for r in info["rows"]
                        if r["budget"] == info["chosen_budget"]), None)
            claim(f"A8[{d}]: the corpus was built at the budget the rule chose",
                  row is not None and row["n_train"] == cases[d]["n_train"],
                  f"sweep chose train={row['n_train'] if row else '?'}, "
                  f"corpus used train={cases[d]['n_train']}")
            # and the rule's own arithmetic: stability across two doublings
            by = {r["budget"]: r for r in info["rows"]}
            b = info["chosen_budget"]
            two, four = by.get(b * 2), by.get(b * 4)
            stable = (two is not None and four is not None and
                      row["admissible_set"] == two["admissible_set"]
                      == four["admissible_set"])
            # Compare against the explicit flag, never the prose: the sentence
            # "no budget on the ladder is stable" contains the word "stable",
            # and a substring test on it passed a domain that was not stable.
            declared = info.get("stable")
            claim(f"A8[{d}]: the sweep's stability verdict is re-derived",
                  declared is not None and stable == declared,
                  f"recomputed stable={stable}, recorded stable={declared}")
        claim("A11: no edit's candidate list was truncated",
              all(not e.get("truncated") for d in domains
                  for x in cases[d]["cases"] if x.get("admitted")
                  for e in x["edits"]))

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
            claim(f"per-case (micro) {key}[{arm}] re-derived — reported only "
                  f"beside the macro figure, never used for a criterion",
                  (got is None and want is None) or
                  (got is not None and want is not None and abs(got - want) < 1e-9),
                  f"{got} vs {want}")

    # ---- A12: criteria are re-derived PER DOMAIN, and the pooled figure is the
    # unweighted mean of the per-domain means, never the per-case mean.
    KEYS = (("C1_N''_le_half_N", "N"), ("C2_N''_le_half_Nprime", "N'"),
            ("C3a_N''_le_half_H1", "H1"), ("C3b_N''_le_half_H2", "H2"))

    def dmean(dom, arm, field="cost"):
        vals = [Fraction(r[field][arm]) for r in a["cases"]
                if r["domain"] == dom and r[field][arm] is not None]
        return float(sum(vals) / len(vals)) if vals else None

    for dom in domains:
        got_m = {arm: dmean(dom, arm) for arm in ARMS}
        rec = (a.get("criteria_by_domain", {}).get(dom) or {}).get("criteria")
        if rec is None:
            claim(f"A12[{dom}]: per-domain criteria are recorded", False)
            continue
        for key, arm in KEYS:
            want = rec[key]
            got = (None if got_m["N''"] is None or got_m[arm] is None
                   else bool(got_m["N''"] <= got_m[arm] / 2))
            claim(f"A12[{dom}]: {key} re-derived", got == want, f"{got} vs {want}")
        want4 = rec["C4_D1_fails_C1"]
        got4 = (None if got_m["D1"] is None or got_m["N"] is None
                else not (got_m["D1"] <= got_m["N"] / 2))
        claim(f"A12[{dom}]: C4 re-derived", got4 == want4, f"{got4} vs {want4}")
        claim(f"A12[{dom}]: the domain verdict is the conjunction of C1-C4",
              rec["met"] == all(rec[k] is True for k, _ in KEYS) and
              rec["met"] == (all(rec[k] is True for k, _ in KEYS)
                             and rec["C4_D1_fails_C1"] is True))

    c = a["criteria"]
    macro = {arm: (lambda v: (sum(v) / len(v)) if v else None)(
                 [x for x in (dmean(d, arm) for d in domains) if x is not None])
             for arm in ARMS}
    for arm in ARMS:
        rec = a["overall"].get("mean_cost_macro", {}).get(arm)
        got = macro[arm]
        claim(f"A12: pooled (macro) cost for {arm} is the unweighted mean of the "
              f"per-domain means",
              (got is None and rec is None) or
              (got is not None and rec is not None and abs(got - rec) < 1e-9),
              f"{got} vs {rec}")
    for key, arm in KEYS:
        want = c.get(key)
        got = (None if macro["N''"] is None or macro[arm] is None
               else bool(macro["N''"] <= macro[arm] / 2))
        claim(f"pooled {key} re-derived from the macro means", got == want,
              f"{got} vs {want}")
    per = {d: ((a.get("criteria_by_domain", {}).get(d) or {}).get("criteria") or {})
              .get("met") for d in domains}
    claim("A12(4): the headline verdict is met on EVERY domain, not pooled",
          c.get("met_all_domains") == (bool(per) and all(per.values())),
          f"per-domain {per}")
    claim("A12(4): a split verdict is flagged as split",
          c.get("split_verdict") == (len({v for v in per.values()}) > 1))
    claim("A12(4): a split verdict is never reported as MET",
          not (c.get("split_verdict") and c.get("met_all_domains")))

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
    claim("D1's per-case mean equals the mean of its 20 stored permutations",
          all(len(r["D1_permutations"]) == 20 and
              Fraction(r["D1_mean"]) ==
              sum(Fraction(x) for x in r["D1_permutations"]) / 20
              for r in a["cases"]))
    claim("D1's headline ordering is the seed-0 permutation, which is one of the 20",
          all(r["cost"]["D1"] in r["D1_permutations"] for r in a["cases"]))

    # ---- layer 2e2: V2, the decider's own cross-checks
    for d in domains:
        v = J(f"validate_{d}")
        if v is None:
            claim(f"V2[{d}]: the decider cross-check was run", False,
                  "out/validate_<domain>.json missing")
            continue
        claim(f"V2[{d}]: the flat walk agrees with the prefix walk on every sample",
              not v["fit_mismatches"], f"{len(v['fit_mismatches'])} disagreements")
        claim(f"V2[{d}]: every recorded witness re-executes to the target",
              not v["witness_mismatches"],
              f"{len(v['witness_mismatches'])} witnesses did not reproduce")
        adm = [x for x in cases[d]["cases"] if x.get("admitted")]
        want = sum(1 for x in adm for e in x["edits"] if e["repair"] and e["witness"])
        claim(f"V2[{d}]: every repair's witness was checked, not a sample",
              v["witnesses_checked"] == want,
              f"{v['witnesses_checked']} vs {want}")
        # A20: the validation must refer to THIS corpus, not a superseded one.
        claim(f"V2[{d}]: the validation stamps the corpus it validated",
              v.get("corpus_final_digest") is not None,
              "no corpus stamp: the artifact predates A20 and may be stale")
        claim(f"V2[{d}]: the validation refers to the current corpus",
              v.get("corpus_final_digest") == cases[d].get("final_digest")
              and v.get("corpus_n_train") == cases[d].get("n_train")
              and v.get("corpus_admitted") == cases[d].get("admitted"),
              f"stamp {v.get('corpus_final_digest')}/{v.get('corpus_n_train')}/"
              f"{v.get('corpus_admitted')} vs corpus "
              f"{cases[d].get('final_digest')}/{cases[d].get('n_train')}/"
              f"{cases[d].get('admitted')}")

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
        # Tokenise the blocks rather than substring-matching them: otherwise a
        # stray `7` in prose passes merely because some block contains `527`.
        inside = set(re.findall(NUMBER, "\n".join(rendered.values())))
        # The block strip must be anchored by a backreference and must tolerate an
        # empty block, or it spans from one BEGIN to a later END and silently
        # deletes the prose in between -- which would let unchecked numbers
        # through.  The code-span strip must not cross a newline, for the same
        # reason: one unbalanced backtick would otherwise pair across paragraphs.
        before = len(text)
        prose = re.sub(r"<!-- BEGIN:(\w+) -->\n?.*?\n?<!-- END:\1 -->", "",
                       text, flags=re.S)
        prose = re.sub(r"`[^`\n]*`", "", prose)        # code spans carry formulae
        prose = re.sub(r"§\s*\d[\d.]*", "", prose)     # citations of other sections
        prose = re.sub(r"\bA\d+\b", "", prose)         # amendment identifiers
        # A guard on the guard: the strips together may not remove most of the
        # document.  If they do, the prose check is not checking anything.
        claim("the prose strip leaves most of RESULTS.md to be checked",
              len(prose) > 0.35 * before,
              f"{len(prose)} of {before} characters survived the strip")
        # A13 left one superseded formulation of the split in the opening
        # paragraph while the amendment stated another.  Two live formulations in
        # an audited document is what this infrastructure exists to prevent, so
        # the check is mechanical: the two-way wording may appear only inside a
        # struck passage (~~...~~) or an amendment section.
        body = re.sub(r"<!-- BEGIN:(\w+) -->\n?.*?\n?<!-- END:\1 -->", "",
                      text, flags=re.S)
        body = re.sub(r"~~.*?~~", "", body, flags=re.S)
        amend = body.find("## Amendments to the pre-registration")
        aend = body.find("## ", amend + 10) if amend >= 0 else -1
        scanned = (body[:amend] + body[aend:]) if amend >= 0 and aend > amend else body
        # A retrospective mention ("under the two-way split ...", "superseded
        # by A13") is fine and is part of the record; a LIVE claim that the split
        # is two-way is not.  An occurrence counts as retrospective only if a
        # marker sits within 200 characters before it.
        MARK = re.compile(r"superseded|under the|the old|previously|was |had been"
                          r"|objection to|A1[0-9]\b", re.I)
        bad = []
        for m in re.finditer(r"train/held-out|held-out split|two-way split", scanned):
            before = scanned[max(0, m.start() - 200):m.start()]
            if not MARK.search(before):
                bad.append(scanned[max(0, m.start() - 60):m.end() + 20].replace("\n", " "))
        claim("no superseded two-way split formulation is live in RESULTS.md",
              not bad, f"{len(bad)} live occurrence(s): {bad[:2]}")

        stray = [t for t in re.findall(NUMBER, prose)
                 if t not in allow and t not in inside]
        claim("no number in RESULTS.md prose is outside a block or the allow-list",
              not stray, ", ".join(sorted(set(stray))[:20]))

    finish()


ALLOWED = {
    # the only numbers allowed in RESULTS.md prose: structural constants fixed
    # by the pre-registration or by an amendment.  Nothing measured belongs here.
    "1", "2", "3", "4", "6", "8",          # keep_prefix's k values (amendment A3)
    "48", "12", "400,000",                 # MAX_NEW, MAX_NEW_NODE, MAX_SPACE
    "2.6", "89",                           # cited: section 65's gap-1 conformer
                                           # count (2.6e11) and arith's measured
                                           # ~89 min/case, both named in prose
    "20", "25", "80",                      # the resource policy: 20x the measured
                                           # peak, the 25 GB unmeasured-phase
                                           # floor, ~80x for MemoryMax (R1, R2)
}


REQUIRED_CLAIMS = (
    # A19: a claim about the verifier belongs inside the verifier.  If an
    # amendment promises a mechanical check, the absence of that check must
    # itself fail -- A13 promised three and shipped none.
    "the corpus declares the final split untouched",
    "the corpus records a final-split fingerprint",
    "no stored feature is derived from admission or final",
    "the corpus was not stopped by a signal",
    "no superseded two-way split formulation is live",
    "the closed form matches a simulated draw",
    "the validation refers to the current corpus",
    "carries all five provenance identities",
    "every artifact used one mutation-grammar version",
)


def check_claims_exist():
    names = " || ".join(n for _, n, _ in results)
    for required in REQUIRED_CLAIMS:
        claim(f"A19: the verifier contains the check '{required}'",
              required in names,
              "an amendment promised this check and nothing implements it")


def finish():
    check_claims_exist()
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
