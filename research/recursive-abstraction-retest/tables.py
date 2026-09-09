"""Summary tables and a two-sided Fisher exact test over the arm results."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent


def fisher(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]]."""
    def logp(x, y, z, w):
        n = x + y + z + w
        return (math.lgamma(x + y + 1) + math.lgamma(z + w + 1) + math.lgamma(x + z + 1)
                + math.lgamma(y + w + 1) - math.lgamma(n + 1) - math.lgamma(x + 1)
                - math.lgamma(y + 1) - math.lgamma(z + 1) - math.lgamma(w + 1))
    obs = logp(a, b, c, d)
    total = 0.
    for x in range(a + b + 1):
        y = a + b - x
        z = a + c - x
        w = d - a + x
        if min(y, z, w) < 0:
            continue
        p = logp(x, y, z, w)
        if p <= obs + 1e-9:
            total += math.exp(p)
    return min(1., total)


def med(xs):
    return statistics.median(xs) if xs else None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    data = json.loads((HERE / src).read_text())
    out = {"acquisition": {}, "arms": {}, "fisher": {}}

    mods = data.get("modules", [])
    for arm in ("B", "C"):
        rows = [m for m in mods if m["arm"] == arm]
        if not rows:
            continue
        ok = [m for m in rows if m.get("acquired")]
        out["acquisition"][arm] = {
            "acquired": f"{len(ok)}/{len(rows)}",
            "median_attempts": med([m["attempts"] for m in ok]),
            "median_steps": med([m["acquisition_steps"] for m in ok]),
            "all_steps": [m["acquisition_steps"] for m in rows],
            "median_live_nodes": med([m["pruned_nodes"] for m in ok]),
            "minimum_nodes": 4,
            "median_execution_cost": med([m["execution_cost"] for m in ok]),
            "median_wall_seconds": med([m["wall_seconds"] for m in rows]),
        }

    for scaffold in ("wide", "tight"):
        for arm in ("A", "B", "C"):
            rs = [r for r in data["runs"] if r["arm"] == arm and r["scaffold"] == scaffold]
            if not rs:
                continue
            succ = [r for r in rs if r["final_conformant"]]
            fail = [r for r in rs if not r["final_conformant"]]
            out["arms"][f"{scaffold}_{arm}"] = {
                "n": len(rs),
                "success": len(succ),
                "module_on_output_path": sum(r["module_on_output_path"] for r in rs),
                "module_on_output_path_of_successes": sum(r["module_on_output_path"] for r in succ),
                "module_selected_somewhere": sum(r["module_selected_somewhere"] for r in rs),
                "median_steps_to_conformance": med([r["first_conformant_step"] for r in succ]),
                "steps_range": [min([r["first_conformant_step"] for r in succ]),
                                max([r["first_conformant_step"] for r in succ])] if succ else None,
                "median_final_loss": round(med([r["final_loss"] for r in rs]), 5),
                "median_final_loss_failures": round(med([r["final_loss"] for r in fail]), 5) if fail else None,
                "median_seconds_per_step": round(med([r["seconds_per_step"] for r in rs]), 4),
                "median_wall_seconds": round(med([r["total_wall_seconds"] for r in rs]), 1),
                "median_wall_seconds_to_success": round(med([r["total_wall_seconds"] for r in succ]), 1) if succ else None,
                "min_final_loss": round(min([r["final_loss"] for r in rs]), 5),
                "scaffold_candidates": rs[0]["scaffold_candidates"],
                "median_live_nodes": med([r["live_nodes"] for r in rs]),
                "median_live_nodes_successes": med([r["live_nodes"] for r in succ]),
                "median_description_bits_pruned": med([r["description_bits_pruned"] for r in rs]),
                "median_description_bits_pruned_successes": med([r["description_bits_pruned"] for r in succ]),
                "median_execution_cost_pruned_successes": med([r["execution_cost_pruned"] for r in succ]),
            }

    for scaffold in ("wide", "tight"):
        a = out["arms"].get(f"{scaffold}_A")
        for arm in ("B", "C"):
            o = out["arms"].get(f"{scaffold}_{arm}")
            if not a or not o:
                continue
            out["fisher"][f"{scaffold}_A_vs_{arm}"] = round(
                fisher(a["success"], a["n"] - a["success"], o["success"], o["n"] - o["success"]), 4)

    # trajectory of softmax mass on module candidates, arm B and C
    traj = {}
    for scaffold in ("wide", "tight"):
        for arm in ("B", "C"):
            rs = [r for r in data["runs"] if r["arm"] == arm and r["scaffold"] == scaffold and r.get("probes")]
            if not rs:
                continue
            steps = sorted({p["step"] for r in rs for p in r["probes"]})
            rows = []
            for st in steps:
                mods, corrects = [], []
                for r in rs:
                    p = next((x for x in r["probes"] if x["step"] == st), None)
                    if not p:
                        continue
                    mods.append(max(v["module_mass"] for v in p["nodes"].values()))
                    corrects.append(max(v["correct_binding_mass"] for v in p["nodes"].values()))
                if mods:
                    rows.append({"step": st, "n": len(mods),
                                 "median_max_module_mass": round(med(mods), 4),
                                 "median_max_correct_binding_mass": round(med(corrects), 5)})
            uni = rs[0]["probes"][0]["nodes"]
            traj[f"{scaffold}_{arm}"] = {
                "uniform_module_mass_per_node": round(med([v["uniform_module_mass"] for v in uni.values()]), 4),
                "uniform_mass_per_single_binding": round(med([1 / v["n_candidates"] for v in uni.values()]), 6),
                "rows": rows[:16]}
    out["module_mass_trajectory"] = traj

    dest = sys.argv[2] if len(sys.argv) > 2 else "tables.json"
    (HERE / dest).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
