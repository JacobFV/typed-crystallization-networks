"""Render every table in RESULTS.md from out/*.json, so no number is transcribed."""
from __future__ import annotations
import json, statistics
from pathlib import Path
O = Path(__file__).resolve().parent / "out"


def load(name):
    p = O / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, n=3):
    if x is None: return "--"
    if isinstance(x, bool): return "yes" if x else "no"
    if isinstance(x, (int,)): return f"{x:,}"
    try: return f"{x:.{n}g}"
    except Exception: return str(x)


def rung1_ladder():
    d = load("rung1_ladder")
    if not d: return "(rung1_ladder.json missing)"
    out = ["| scaffold | free nodes | programs | gradient (16 seeds) | median exact error | enumeration | unique | random-draw density |",
           "|---|---|---|---|---|---|---|---|"]
    for k, v in d.items():
        s, e, r = v["summary"], v["enumeration"], v["random"]
        out.append(f"| `{k}` | {len(v['free'])} | {v['space']:,} | **{s['train_success']}/{s['n']}** "
                   f"({s['rate']:.0%}) | {fmt(s['median_test_err'])} | {e['seconds']:.3f} s, solved | "
                   f"{fmt(e['unique'])} | {fmt(r['density'],2)} |")
    return "\n".join(out)


def rung1_future():
    d = load("rung1_future")
    if not d: return "(rung1_future.json missing)"
    out = ["| arm | supervision | programs | gradient (12 seeds) | median held-out max error |", "|---|---|---|---|---|"]
    for k, v in d.items():
        s = v["summary"]
        out.append(f"| `{k}` | {' + '.join(v['signals'])} | {v['space']:,} | {s['train_success']}/{s['n']} | {fmt(s['median_test_err'],4)} |")
    return "\n".join(out)


def rung1_gap():
    d = load("rung1_relaxation_gap")
    if not d: return "(missing)"
    out = ["| scaffold | relaxed loss at the one-hot reference | median relaxed loss the optimiser reaches | best of 8 |", "|---|---|---|---|"]
    for k, v in d.items():
        ls = [x[0] for x in v["optimiser"] if isinstance(x[0], float) and x[0] == x[0]]
        out.append(f"| `{k}` | {v['onehot_reference_relaxed_loss']:.3g} | {statistics.median(ls):.3g} | {min(ls):.3g} |")
    return "\n".join(out)


def rung2():
    d = load("rung2_arms")
    if not d: return "(rung2_arms.json missing)"
    out = ["| arm | programs | supervision | exact on train (8 seeds) | mean held-out accuracy | enumeration | random density |",
           "|---|---|---|---|---|---|---|"]
    for k, v in d.items():
        e = v.get("enumeration"); r = v.get("random")
        for arm in ("dense", "output_only"):
            a = v.get(arm)
            if not a: continue
            s = a["summary"]
            out.append(f"| `{k}` | {v['space']:,} | {arm} | **{s['train_success']}/{s['n']}** | "
                       f"{a.get('mean_test_accuracy', float('nan')):.4f} | "
                       f"{('%.2f s, solved=%s' % (e['seconds'], e['solved'])) if e else 'space too large'} | "
                       f"{fmt(r['density'],2) if r else '--'} |")
    return "\n".join(out)


def rung3_width():
    d = load("rung3_width")
    if not d: return "(missing)"
    out = ["| resolution | observation width | byte addresses | programs | gradient (12 seeds) | enumeration | held-out error of the enumerated program | random density |",
           "|---|---|---|---|---|---|---|---|"]
    for k, v in d.items():
        s, e, r = v["summary"], v["enumeration"], v["random"]
        out.append(f"| {v['resolution']}x{v['resolution']} | {v['image_width']} | {v['addresses']} | {v['space']:,} | "
                   f"**{s['train_success']}/{s['n']}** ({s['rate']:.0%}) | {e['seconds']:.2f} s, solved | "
                   f"{fmt(e.get('held_out_error'))} | {fmt(r['density'],2)} |")
    return "\n".join(out)


def rung3_dense(name="rung3_dense_pinned"):
    d = load(name)
    if not d: return f"({name}.json missing)"
    out = ["| head | resolution | programs | P(all background) | supervision | exact on train (12 seeds) | **generalises to 48 held-out episodes** |",
           "|---|---|---|---|---|---|---|"]
    for k, v in d.items():
        s = v["summary"]; head, res, arm = k.split("_", 2)
        out.append(f"| `{head}` | {res} | {v['space']:,} | {v['positive_rate_allbg']:.3f} | {arm} | "
                   f"{s['train_success']}/{s['n']} | **{s['generalises']}/{s['n']}** |")
    return "\n".join(out)


def rung3_dense_free():
    return rung3_dense("rung3_dense")


def rung3_colour():
    d = load("rung3_colour")
    if not d: return "(rung3_colour.json missing)"
    out = ["| resolution | programs (address pinned, colour searched over all 256 byte values) | gradient (12 seeds) | enumeration |", "|---|---|---|---|"]
    for k, v in d.items():
        s, e = v["summary"], v["enumeration"]
        out.append(f"| {k} | {v['space']:,} | **{s['train_success']}/{s['n']}** | {e['seconds']:.2f} s, solved={e['solved']}, unique={fmt(e['unique'])} |")
    return "\n".join(out)


def rung4():
    parts = []
    e = load("rung4_eq")
    if e:
        s = e["summary"]; en = e["enumeration"]
        parts.append(f"`eq` route (the only differentiable predicate on bytes): {e['space']:,} programs. "
                     f"Enumeration **exhausted the space in {en['seconds']:.1f} s and found no conforming program** "
                     f"(solved={en['solved']}, exhausted={en['exhausted']}). Gradient synthesis: {s['train_success']}/{s['n']}.")
    t = load("rung4_threshold")
    if t:
        parts.append("Gradient reach in the `pack -> decode -> lt` program: "
                     + ", ".join(f"`{k}` = {v}" for k, v in t["choice_grad_l1"].items())
                     + f"; trainable threshold gradient = {t['threshold_grad_l1']}.")
        rows = ["| arm | exact on train | median held-out error |", "|---|---|---|"]
        for k, v in t.get("gradient", {}).items():
            s = v["summary"]; rows.append(f"| {k} | {s['train_success']}/{s['n']} | {fmt(s['median_test_err'])} |")
        parts.append("\n".join(rows))
        ep = t.get("enumeration_threshold_pool")
        if ep:
            en = ep["enumeration"]
            parts.append(f"Discrete backend over address x comparison x 256 thresholds ({ep['space']:,} programs): "
                         f"solved={en['solved']} in {en['seconds']:.1f} s, unique={fmt(en['unique'])}, "
                         f"held-out accuracy {en.get('held_out_accuracy', float('nan')):.4f}; "
                         f"random search density {fmt(ep['random']['density'],2)}.")
    return "\n\n".join(parts) if parts else "(rung4 json missing)"


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["rung1_ladder", "rung1_future", "rung1_gap", "rung2", "rung3_width",
                             "rung3_dense", "rung3_colour", "rung4"]
    for name in which:
        print(f"\n### {name}\n"); print(globals()[name]())
