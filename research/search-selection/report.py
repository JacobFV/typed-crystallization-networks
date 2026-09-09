"""Render the tables in RESULTS.md from the raw JSON, so nothing is transcribed.

Run:  PYTHONPATH=. .venv/bin/python research/search-selection/report.py
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def g(x, n=3):
    if x is None: return "--"
    if isinstance(x, bool): return "yes" if x else "no"
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")): return "inf"
        return f"{x:.{n}g}"
    return str(x)


def eq_table(d):
    rows = d["eq_profile"]["rows"]
    keep = [0, 1, 4, 8, 10, 11, 12, 25, 32, 64, 128, 255]
    out = ["| \\|a-b\\| | `eq` at tau=1 | d/da at tau=1 | `eq` at tau=2^8 | d/da at tau=2^8 |",
           "|---|---|---|---|---|"]
    for r in rows:
        if r["spread"] in keep:
            out.append(f"| {r['spread']} | {g(r['value_tau1'])} | {g(r['grad_tau1'])} | "
                       f"{g(r['value_tau256'])} | {g(r['grad_tau256'])} |")
    return "\n".join(out)


def liveness_table(d):
    out = ["| case | space | reachable, shipped | reachable, carrier-scaled | dead nodes | candidates dead on every example |",
           "|---|---|---|---|---|---|"]
    for c in d["cases"]:
        a, b = c["arms"][0], c["arms"][1]
        out.append(f"| `{c['case']}` | {c['space']:,} | {g(a['reachable_fraction'],4)} | "
                   f"{g(b['reachable_fraction'],4)} | {', '.join(a['dead_nodes']) or '--'} | "
                   f"{len(a['dead_candidates'])} -> {len(b['dead_candidates'])} |")
    return "\n".join(out)


def arm(block, backend):
    for a in block["arms"]:
        if a["backend"] == backend: return a
    return None


def decisions_table(d):
    out = ["| case | family | space | chose | right | agrees | why |", "|---|---|---|---|---|---|---|"]
    for c in d["cases"]:
        out.append(f"| `{c['case']}` | {c['family']} | {c['space']:,} | **{c['chosen']}** | {c['right']} | "
                   f"{'yes' if c['chosen'] == c['right'] else '**no**'} | {c['reason']} |")
    return "\n".join(out)


def outcomes_table(d):
    out = ["| case | enumerate | hybrid | relax (carrier-scaled where chosen) | relax, shipped surrogate |",
           "|---|---|---|---|---|"]
    for c in d["cases"]:
        cells = []
        for b in ("enumerate", "hybrid", "relax", "relax_shipped_surrogate"):
            a = arm(c, b)
            if a is None: cells.append("--"); continue
            if not a.get("ran", True):
                cells.append(f"not run, projected {g(a['projected_seconds'],4)} s"); continue
            s = f"{a['successes']}/{a['seeds']} in {g(a.get('seconds'),3)} s"
            if b == "enumerate":
                s += f", {a['conforming']} conforming"
                s += ", certificate" if a["certificate"] else ", no certificate"
            cells.append(s)
        out.append(f"| `{c['case']}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def cost_table(d):
    """The asymmetry: what each mistake would have cost on each case."""
    out = ["| case | chose | cost of choosing enumerate | cost of choosing relax | ratio |",
           "|---|---|---|---|---|"]
    for c in d["cases"]:
        e, r = arm(c, "enumerate"), arm(c, "relax")
        h = arm(c, "hybrid")
        disc = h if c["chosen"] == "hybrid" and h else e
        ecost = (disc.get("seconds") if disc and disc.get("ran", True)
                 else (e or {}).get("projected_seconds"))
        eok = disc.get("successes", 0) if disc and disc.get("ran", True) else None
        rcost = r.get("seconds_per_run") if r else None
        rok = r.get("successes") if r else None
        ratio = "--"
        if ecost and rcost: ratio = f"{rcost/ecost:.3g}x" if ecost else "--"
        out.append(f"| `{c['case']}` | {c['chosen']} | {g(ecost,3)} s"
                   f"{'' if eok is None else f', {eok} solved'}"
                   f"{' (projected)' if disc and not disc.get('ran', True) else ''} | "
                   f"{g(rcost,3)} s/run, {rok}/{r['seeds'] if r else '--'} exact | {ratio} |")
    return "\n".join(out)


def fixtures_table(d):
    out = ["| fixture | space | chose | right | agrees | exact error | certified |", "|---|---|---|---|---|---|---|"]
    for r in d["auto"]:
        if "error" in r:
            out.append(f"| `{r['fixture']}` | -- | error | -- | -- | {r['error']} | -- |"); continue
        out.append(f"| `{r['fixture']}` | {r['space']:,} | **{r['chose'] if 'chose' in r else r['chosen']}** | "
                   f"{r['right']} | {'yes' if r['agrees'] else '**no**'} | {g(r['exact_max_error'])} | "
                   f"{g(r['certified'])} |")
    return "\n".join(out)


def main():
    lv, su, fx = load("liveness"), load("suite"), load("fixtures")
    if lv:
        print("### eq surrogate profile\n"); print(eq_table(lv)); print()
        print("### liveness per case\n"); print(liveness_table(lv)); print()
        print("shipped forward bit-identical:",
              {r["fixture"]: r["bit_identical"] for r in lv["shipped_forward_bit_identical"]}); print()
    if su:
        print("### decisions\n"); print(decisions_table(su)); print()
        print("### outcomes\n"); print(outcomes_table(su)); print()
        print("### asymmetric costs\n"); print(cost_table(su)); print()
        n = len(su["cases"]); ok = sum(c["chosen"] == c["right"] for c in su["cases"])
        print(f"agreement {ok}/{n}; suite wall {su['wall_seconds']} s\n")
    if fx:
        print("### fixtures through synthesis.fit(mode='auto')\n"); print(fixtures_table(fx)); print()
        print("default path:", fx["default"])


if __name__ == "__main__":
    main()
