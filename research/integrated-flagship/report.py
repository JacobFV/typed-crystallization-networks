"""Render every figure-bearing block of RESULTS.md from files under `out/`.

Blocks sit between `<!-- BEGIN:name -->` and `<!-- END:name -->`. `report.py`
rewrites them in place; `verify.py` re-renders them and fails if the committed
text differs, and separately re-derives each headline claim from raw JSON.

    python report.py
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
RESULTS = HERE / "RESULTS.md"
ARMS = ["N", "N'", "N''", "P", "D'", "A_noobs"] + [f"D''_{s}" for s in range(5)] + \
       [f"R_{s}" for s in range(5)] + ["U_feat"] + [f"OCC_{r}" for r in (1, 3, 10, 100, 3500)] + \
       [f"V_{r}" for r in (1, 3, 10, 100)]
LABEL = {"U_feat": "U — N″'s features, UNFITTED weights (post-hoc control)",
         **{f"OCC_{r}": f"N″ with occurs ratio pinned at {r} (post-hoc sweep)" for r in (1, 3, 10, 100, 3500)},
         **{f"V_{r}": f"N″ with V ratio pinned at {r} (post-hoc sweep)" for r in (1, 3, 10, 100)},"N": "N — flat, uniform", "N'": "N′ — schema, uniform", "N''": "N″ — schema + learned prior",
         "P": "P — flat + learned prior", "D'": "D′ — distractor schemas, uniform",
         "A_noobs": "N″ without observation features",
         **{f"D''_{s}": f"D″ — distractor schemas + permuted prior (seed {s})" for s in range(5)},
         **{f"R_{s}": f"R — right schemas + permuted prior (seed {s})" for s in range(5)}}


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def arm(gap, a):
    return load(f"arm_gap{gap}_{a.replace(chr(39), 'p')}.json")


def sci(x):
    if x is None:
        return "—"
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        e = math.floor(math.log10(abs(x)))
        return f"{x / 10 ** e:.2f}×10^{e}"
    if abs(x) < 1e5:
        return f"{x:,.0f}" if x == int(x) else f"{x:,.2f}"
    e = math.floor(math.log10(abs(x)))
    return f"{x / 10 ** e:.2f}×10^{e}"


def big(s):
    return sci(int(s)) if s is not None else "—"


def cost(d):
    return Fraction(int(d["expected_programs"]["num"]), int(d["expected_programs"]["den"])) \
        if d and d.get("expected_programs") else None


def ratio_str(a, b):
    if a is None or b is None:
        return "—"
    r = a / b
    return sci(float(r))


# ----------------------------------------------------------------- blocks
def block_arms(gap):
    n = arm(gap, "N")
    cn = cost(n)
    rows = ["| arm | full space S | conformers K | certificate | first solvable tier | expected programs to first solution | expected episodes | saving vs N | first solution generalizes (of sampled) | held-out accuracy of first solution |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        d = arm(gap, a)
        if d is None:
            continue
        c = cost(d)
        s = d.get("first_solution_samples") or {}
        gen = (f"{s['generalize_count']}/{s['n']} (95% {s['generalize_wilson95'][0]:.3f}–{s['generalize_wilson95'][1]:.3f})"
               if s else "—")
        rows.append(f"| {LABEL[a]} | {big(d['full_space']['S'])} | {big(d['full_space']['K'])} | "
                    f"`{d['full_space']['certificate']}` | "
                    f"{d['first_solvable_tier'] if d['first_solvable_tier'] is not None else 'none'} | "
                    f"{sci(float(c)) if c is not None else 'no solution in the space'} | "
                    f"{sci(float(c * 12)) if c is not None else '—'} | "
                    f"{ratio_str(cn, c) + '×' if c is not None and cn is not None else '—'} | {gen} | "
                    f"{s['held_out_accuracy_mean']:.4f} |" if s else
                    f"| {LABEL[a]} | {big(d['full_space']['S'])} | {big(d['full_space']['K'])} | "
                    f"`{d['full_space']['certificate']}` | none | no solution in the space | — | — | — | — |")
    return "\n".join(rows)


def block_tiers(gap, a):
    d = arm(gap, a)
    if d is None:
        return "(not run)"
    rows = ["| tier | S_t | K_t | shell S | shell K | ADDR kept | MATCH kept | STEP kept | absent letters kept |",
            "|---|---|---|---|---|---|---|---|---|"]
    for t in d["tiers"]:
        rows.append(f"| {t['tier']} | {big(t['S'])} | {big(t['K'])} | {big(t['shell_S'])} | "
                    f"{big(t['shell_K'])} | {t['sizes']['cpos']} | {t['sizes']['M']} | "
                    f"{t['sizes']['X1']} | {'yes' if t['lit_absent_allowed']['lc1'] else 'no'} |")
    return "\n".join(rows)


def block_criteria(gap):
    n, n1, n2 = arm(gap, "N"), arm(gap, "N'"), arm(gap, "N''")
    dist = arm(gap, "D''_0")
    ex = load(f"extras_gap{gap}.json")
    lines = []
    cn, c1, c2 = cost(n), cost(n1), cost(n2)
    if cn is not None and c2 is not None:
        lines.append(f"- **C1 (material saving, ≥10× in programs and episodes):** N″/N = {sci(float(c2 / cn))} "
                     f"→ **{'PASS' if c2 * 10 <= cn else 'FAIL'}** (episodes are programs × 12, so the same ratio).")
    ds = [arm(gap, f"D''_{s}") for s in range(5)] + [arm(gap, "D'")]
    ds = [d for d in ds if d is not None]
    if ds and cn is not None:
        ok = all((not d["solution_exists"]) or cost(d) * 10 > cn for d in ds)
        lines.append(f"- **C2 (the matched distractor does not reach 10×):** "
                     + "; ".join(f"{d['arm']}: " + ("no solution in its space (`complete`, 0)" if not d["solution_exists"]
                                                    else f"cost/N = {sci(float(cost(d) / cn))}") for d in ds)
                     + f" → **{'PASS' if ok else 'FAIL'}**.")
    if n2 and ex:
        s = n2["first_solution_samples"]
        b = ex["baselines"]
        best = max(b["best_constant_click_held_accuracy"], b["oracle_constant_click_held_accuracy"],
                   b["random_pixel_held_accuracy"], b["random_child_held_accuracy"])
        ok = s["held_out_accuracy_min"] >= .9 and s["held_out_accuracy_min"] > best
        lines.append(f"- **C3 (a solution, not a fit):** the worst of {s['n']} sampled N″ first solutions scores "
                     f"{s['held_out_accuracy_min']:.4f} held out (evaluator; live check in the instrumentation table) "
                     f"against the best baseline {best:.4f} → **{'PASS' if ok else 'FAIL'}**.")
    if cn is not None and c1 is not None:
        lines.append(f"- **N′ vs N:** N′/N = {sci(float(c1 / cn))} → schema "
                     f"{'is organisational reuse (within 10×)' if c1 * 10 > cn else 'saves ≥10×'}.")
    if c1 is not None and c2 is not None:
        lines.append(f"- **N″ vs N′ (does the prior earn its name, ≤ N′/10?):** N″/N′ = {sci(float(c2 / c1))} → "
                     f"**{'yes' if c2 * 10 <= c1 else 'no'}**.")
    return "\n".join(lines)


def block_f3(gap):
    n = arm(gap, "N")
    if n is None:
        return "(N not run)"
    SN, KN, cn = int(n["full_space"]["S"]), int(n["full_space"]["K"]), cost(n)
    rows = ["| arm | space ratio S_N / S_arm | solution retention K_arm / K_N | saving N / arm (expected programs) | saving vs space ratio |",
            "|---|---|---|---|---|"]
    for a in ("N'", "N''", "P"):
        d = arm(gap, a)
        if d is None or cost(d) is None:
            continue
        S, K = int(d["full_space"]["S"]), int(d["full_space"]["K"])
        sav = cn / cost(d)
        rows.append(f"| {LABEL[a]} | {sci(SN / S)} | {sci(K / KN) if KN else '—'} | {sci(float(sav))} | "
                    f"{sci(float(sav / Fraction(SN, S)))} |")
    return "\n".join(rows)


def block_prior():
    d = arm(0, "N''")
    if d is None:
        return "(N″ not run)"
    rows = ["| kind | feature tuple | survived / rows | score |", "|---|---|---|---|"]
    for kind, e in d["prior_estimators"].items():
        for k, c in e["cells"].items():
            if kind == "ADDR" and k.count(",") < 2:
                continue
            rows.append(f"| {kind} | `{k}` | {c['survive']}/{c['n']} | {c['score']:.4f} |")
    return "\n".join(rows)


def block_sources():
    s = load("sources.json")
    rows = ["| source | space | conforming | certificate | recorded |", "|---|---|---|---|---|"]
    for c in s["certificates"]:
        if "stages" in c:
            for k, v in c["stages"].items():
                rows.append(f"| §33 {k} (res {c['resolution']}, committed) | {v['space']} | {v['conforming']} | `{v['certificate']}` | same file |")
        else:
            rows.append(f"| {c['source']} | {c['space']} | {c['conforming']} | `{c['certificate']}` | {c['recorded']} |")
    kinds = {k: (len(v), sum(r["survive"] for r in v)) for k, v in s.items() if isinstance(v, list) and k != "certificates"}
    rows.append("")
    rows.append("Rows per kind (rows / survivors): " + ", ".join(f"{k} {n}/{p}" for k, (n, p) in kinds.items()))
    return "\n".join(rows)


def block_validation():
    rows = ["| check | result |", "|---|---|"]
    for gap in (0, 1):
        b = load(f"v2_brute_gap{gap}.json")
        if b:
            rows.append(f"| V2 counter = brute force, gap {gap} | {b['n_agree']}/{b['n']} sub-spaces agree on S and K; {b['n_with_conformers']} contain conformers |")
    t = load("v2_tcn_gap0.json")
    if t:
        rows.append("| V2 counter = `tcn.search.enumerate_prefix` | " + "; ".join(
            f"{r['design']}: S {r['space_counter']}, K {r['K_counter']}, tcn `{r['tcn']['certificate']}` {r['tcn']['conforming']}"
            for r in t["designs"]) + f" — {t['n_agree']}/{len(t['designs'])} agree |")
    v1 = [load(f"v1_shard{i}.json") for i in range(3)]
    v1 = [x for x in v1 if x]
    if v1:
        n = sum(x["n"] for x in v1)
        ok = sum(x["n_agree"] for x in v1)
        arms = {}
        for x in v1:
            for r in x["rows"]:
                arms[r["arm"]] = arms.get(r["arm"], 0) + 1
        rows.append(f"| V1 evaluator = `Program.execute` (click node, 48 episodes each) | {ok}/{n} programs agree ({', '.join(f'{k} {v}' for k, v in arms.items())}) |")
    v3 = load("v3_live.json")
    if v3:
        e = v3["enumerate_environment"]
        rows.append(f"| V3 live kernel rewards = evaluator hits (12 training episodes each) | {v3['n_agree']}/{v3['n']} programs |")
        rows.append(f"| V3 `tcn.search.enumerate_environment` through the integrated generator | space {e['space']}, "
                    f"conforming {e['tcn']['conforming']} (`{e['tcn']['certificate']}`), counter {e['K_counter']}, "
                    f"episodes {e['tcn']['episodes']} → {'agree' if e['agree'] else 'DISAGREE'} |")
    return "\n".join(rows)


def block_extras(gap):
    ex = load(f"extras_gap{gap}.json")
    if ex is None:
        return "(not run)"
    b, h, a = ex["baselines"], ex["hard_vector"], ex["achieved"]
    rows = ["| quantity | value |", "|---|---|",
            f"| hard-transferred vector H: training accuracy | {h['train_accuracy']:.4f} |",
            f"| H: held-out accuracy | {h['held_accuracy']:.4f} |",
            f"| best constant click (chosen on training) held-out | {b['best_constant_click_held_accuracy']:.4f} |",
            f"| oracle constant click (chosen on held-out) | {b['oracle_constant_click_held_accuracy']:.4f} |",
            f"| uniform random pixel, held-out (exact) | {b['random_pixel_held_accuracy']:.4f} |",
            f"| uniform random child widget, held-out (exact) | {b['random_child_held_accuracy']:.4f} |",
            f"| widgets per screen (48 episodes) | {a['widgets_per_screen']} |",
            f"| rejection draws per episode (min / median / max) | {min(a['draws'])} / {sorted(a['draws'])[len(a['draws']) // 2]} / {max(a['draws'])} |",
            f"| instruction lengths (bytes) | {a['text_lengths']} |",
            f"| target area in pixels (min / max) | {min(a['target_areas'])} / {max(a['target_areas'])} |",
            f"| H's selection | {', '.join(f'{k} {v}' for k, v in h['specs'].items())} |"]
    return "\n".join(rows)


def block_instrument(gap):
    rows = load(f"instrument_gap{gap}.json")
    if not rows:
        return "(not run)"
    out = ["| arm | found program (slot specs) | slots on the execution path | live train / held-out (kernel) | live = evaluator | interpreted s/episode | compiled s/episode | compiled speed-up |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("specs"):
            continue
        spec = "; ".join(f"{k}={v}" for k, v in r["specs"].items() if k in ("cpos", "ra", "M", "X1", "X2", "X3"))
        out.append(f"| {r['arm']} | {spec} | {len(r['slots_on_execution_path'])}/15 | "
                   f"{r['live_train_accuracy']:.4f} / {r['live_held_accuracy']:.4f} | {r['live_agrees_with_engine']} | "
                   f"{r['interpreted_median_s']:.3f} | {r['compiled_median_s']:.5f} | {r['compiled_speedup']:.0f}× |")
    return "\n".join(out)


def block_resources():
    rows = ["| capped job | peak RSS (GB) | wall clock | exit |", "|---|---|---|---|"]
    for p in sorted(OUT.glob("*.time")):
        txt = p.read_text()
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
        w = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", txt)
        e = re.search(r"Exit status: (\d+)", txt)
        if m:
            rows.append(f"| `{p.stem}` | {int(m.group(1)) / 1024 / 1024:.2f} | {w.group(1) if w else '?'} | {e.group(1) if e else '0'} |")
    log = (OUT / "memory_floor.log").read_text().splitlines() if (OUT / "memory_floor.log").exists() else []
    vals = [int(l.split("MemAvailable_kB=")[1]) for l in log if "MemAvailable_kB=" in l]
    refused = sum("REFUSED" in l for l in log)
    rows.append("")
    rows.append(f"Memory floor: {len(vals)} capped starts logged in `out/memory_floor.log`; MemAvailable at start "
                f"ranged {min(vals) / 1024 / 1024:.1f}–{max(vals) / 1024 / 1024:.1f} GB; phases refused by the 25 GB floor: {refused}."
                if vals else "Memory floor log empty.")
    return "\n".join(rows)


def block_leakage():
    s = load("sources.json")
    kinds = {k: v for k, v in s.items() if isinstance(v, list) and k != "certificates"}
    by = {}
    for k, v in kinds.items():
        for r in v:
            by.setdefault((k, r["source"]), [0, 0])
            by[(k, r["source"])][0] += 1
            by[(k, r["source"])][1] += int(r["survive"])
    rows = ["| prior kind | rows | survivors | from which earlier search |", "|---|---|---|---|"]
    for k in ("ADDR", "LIT", "TRUTH", "STEP", "GROUND"):
        srcs = [(src, n, p) for (kk, src), (n, p) in sorted(by.items()) if kk == k]
        rows.append(f"| {k} | {sum(n for _, n, _ in srcs)} | {sum(p for _, _, p in srcs)} | "
                    + (", ".join(f"{src} ({n})" for src, n, _ in srcs) or "none") + " |")
    rows.append("")
    rows.append("No row comes from the integrated task, its training episodes or its held-out episodes; "
                "`verify.py` checks this and that the prior refitted from `sources.json` alone reproduces "
                "every stored estimator cell and N″'s tiers from the training episodes' unlabelled text. "
                "The integrated task contributes only unlabelled features of its training observations "
                "(V: does the addressed byte vary; occurs: does the literal appear there). The held-out "
                "episodes enter nothing but the generalization score.")
    return "\n".join(rows)


def block_generalize(gap):
    g = load(f"generalize_gap{gap}.json")
    if g is None:
        return "(not run)"
    d, s = g["distractor"], g["schema"]
    kd = int(d["K_generalizing"])
    n2 = g.get("N''_expected_programs_to_first_generalizing")
    ok = all(v["transform"] == v["direct"] for v in g["direct_equals_transform_on_train"].values())
    lines = [
        "| space | programs | conform on all 48 episodes (train + held-out) | certificate | expected programs to first generalizing, uniform order |",
        "|---|---|---|---|---|",
        f"| distractor pools (D′, D″) | {big(d['S_all_episodes'])} | {big(d['K_generalizing'])} | `{d['certificate']}` | "
        f"{'no generalizing program exists' if kd == 0 else sci(10 ** d['expected_programs_to_first_generalizing_uniform']['log10'])} |",
        f"| schema pools (N′) | {big(s['S_all_episodes'])} | {big(s['K_generalizing'])} | `{s['certificate']}` | "
        f"{sci(10 ** s['expected_programs_to_first_generalizing_uniform']['log10']) if s['expected_programs_to_first_generalizing_uniform'] else '—'} |",
        "",
        f"N″'s tiered order, to its first generalizing program: {sci(10 ** n2['log10']) if n2 else 'none found in the tiers counted'} programs.",
        f"Distractor address expressions equal to `length−5` on every episode: "
        f"{len(g['distractor_addresses_equal_length_minus_5'])} of {g['distractor_address_pool_size']}.",
        f"Direct 48-episode counting equals the transform on the 12 training episodes: {ok}.",
        "",
        ("**The pre-registered distractor could not succeed**: its space contains no program that conforms on all 48 "
         "episodes, so D′/D″'s failure to generalize is true by construction and says nothing about the library. "
         "The matched-distractor role is carried by R (right schemas, permuted prior) and U (unfitted weights)."
         if kd == 0 else
         "The distractor space does contain generalizing programs, so D″'s failure to find them first is informative."),
    ]
    return "\n".join(lines)


BLOCKS = {
    "leakage": block_leakage,
    "generalize_gap0": lambda: block_generalize(0),
    "criteria_gap0": lambda: block_criteria(0), "criteria_gap1": lambda: block_criteria(1),
    "arms_gap0": lambda: block_arms(0), "arms_gap1": lambda: block_arms(1),
    "tiers_npp_gap0": lambda: block_tiers(0, "N''"), "tiers_p_gap0": lambda: block_tiers(0, "P"),
    "f3_gap0": lambda: block_f3(0), "f3_gap1": lambda: block_f3(1),
    "prior": block_prior, "sources": block_sources, "validation": block_validation,
    "extras_gap0": lambda: block_extras(0), "extras_gap1": lambda: block_extras(1),
    "instrument_gap0": lambda: block_instrument(0), "instrument_gap1": lambda: block_instrument(1),
    "resources": block_resources,
}


def render(text):
    def sub(m):
        name = m.group(1)
        body = BLOCKS[name]() if name in BLOCKS else f"(unknown block {name})"
        return f"<!-- BEGIN:{name} -->\n{body}\n<!-- END:{name} -->"
    return re.sub(r"<!-- BEGIN:(\w+) -->.*?<!-- END:\1 -->", sub, text, flags=re.S)


if __name__ == "__main__":
    RESULTS.write_text(render(RESULTS.read_text()))
    print("rendered", RESULTS)
