"""Render the arm-comparison markdown tables from the raw result JSON."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def med(xs):
    return round(statistics.median(xs), 4) if xs else None


def reachable_ops(sel, outputs=("y",)):
    """Operators on the path from the outputs, using recorded selections."""
    keep, stack = set(), list(outputs)
    while stack:
        k = stack.pop()
        if k in sel and k not in keep:
            keep.add(k)
            stack += sel[k]["sources"]
    return [sel[k]["operator"] for k in keep]


def summarize(runs, label, budget, post_freeze=False):
    succ = [r for r in runs if r["final_conformant"]]
    st = [r["first_conformant_step"] for r in succ if r["first_conformant_step"] is not None]
    row = {
        "arm": label,
        "n": len(runs),
        "success": f"{len(succ)}/{len(runs)}",
        "steps_median": med(st),
        "steps_range": f"{min(st)}-{max(st)}" if st else "-",
        "loss_median": med([r["final_loss"] for r in runs]),
        "loss_median_failures": med([r["final_loss"] for r in runs if not r["final_conformant"]]),
        "candidates": runs[0]["scaffold_candidates"],
        "sec_per_step": med([r["seconds_per_step"] for r in runs]),
        "bits_median": med([r["final_description_bits"] for r in runs]),
        "cost_median": med([r["final_execution_cost"] for r in runs]),
        "budget": budget,
    }
    if post_freeze:
        pf = [r for r in runs if r.get("post_freeze_conformant")]
        row["post_freeze_success"] = f"{len(pf)}/{len(runs)}"
        cr = [r["crystallization"] for r in runs if "crystallization" in r]
        if cr:
            row["freeze_attempts_median"] = med([len(c["events"]) for c in cr])
            row["freezes_accepted_median"] = med([c["accepted"] for c in cr])
            row["fully_frozen"] = f"{sum(1 for c in cr if c['fully_frozen'])}/{len(cr)}"
            reasons = {}
            for c in cr:
                for e in c["events"]:
                    k = e["reason"].split(":")[0]
                    reasons[k] = reasons.get(k, 0) + 1
            row["freeze_reasons"] = reasons
    outs = ("y",) if "y" in runs[0].get("selections", {}) else ("sum_out", "carry_out")
    row["module_anywhere_in_argmax"] = sum(
        1 for r in runs
        if any(v["operator"].startswith("module:") for v in r.get("selections", {}).values()))
    row["module_on_output_path"] = sum(
        1 for r in runs
        if any(o.startswith("module:") for o in reachable_ops(r.get("selections", {}), outs)))
    row["module_on_output_path_when_successful"] = sum(
        1 for r in runs if r["final_conformant"]
        and any(o.startswith("module:") for o in reachable_ops(r.get("selections", {}), outs)))
    return row


def main():
    out = {}
    e1 = load("e1_results.json")
    ctl = load("control_results.json")
    e2 = load("e2_results.json")
    e1l = load("e1_long_results.json")

    if e1:
        rows = []
        for arm, label in (("A", "A flat"), ("B", "B module")):
            runs = [r for r in e1["runs"] if r["arm"] == arm]
            if runs:
                rows.append(summarize(runs, label, e1["config"]["steps"], post_freeze=True))
        if ctl and ctl["runs"]:
            rows.append(summarize(ctl["runs"], "C useless module", ctl["config"]["steps"]))
        out["e1"] = rows
        ms = e1["module_stage"]
        out["e1_module_stage"] = dict(
            n=len(ms),
            success=sum(1 for m in ms if m.get("module_digest")),
            steps_median=med([m["steps_run"] for m in ms]),
            steps_range=f"{min(m['steps_run'] for m in ms)}-{max(m['steps_run'] for m in ms)}",
            bits_median=med([m["module_description_bits"] for m in ms if "module_description_bits" in m]),
            cost_median=med([m["module_execution_cost"] for m in ms if "module_execution_cost" in m]),
            distinct_digests=len({m.get("module_digest") for m in ms}),
        )
    if e1l and e1l.get("runs"):
        out["e1_long"] = [
            summarize([r for r in e1l["runs"] if r["arm"] == arm], f"{arm} ({e1l['config']['steps']} steps)",
                      e1l["config"]["steps"])
            for arm in ("A", "B") if [r for r in e1l["runs"] if r["arm"] == arm]
        ]
    if e2 and e2.get("runs"):
        out["e2"] = [
            summarize([r for r in e2["runs"] if r["arm"] == arm], f"{arm} {'flat' if arm=='A' else 'module'}",
                      e2["config"]["steps"])
            for arm in ("A", "B") if [r for r in e2["runs"] if r["arm"] == arm]
        ]
        ms = e2["module_stage"]
        out["e2_module_stage"] = dict(
            n=len(ms),
            success=sum(1 for m in ms if m.get("digest")),
            attempts_median=med([m["attempts"] for m in ms]),
            acquisition_steps_median=med([m["acquisition_steps"] for m in ms]),
            acquisition_steps_range=f"{min(m['acquisition_steps'] for m in ms)}-{max(m['acquisition_steps'] for m in ms)}",
            bits_median=med([m["description_bits"] for m in ms if "description_bits" in m]),
            cost_median=med([m["execution_cost"] for m in ms if "execution_cost" in m]),
        )
    print(json.dumps(out, indent=2, default=str))
    (HERE / "summary.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
