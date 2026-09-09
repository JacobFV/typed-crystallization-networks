"""Emit the section 4 markdown from tables.json, so the report's numbers come
straight from the recorded runs rather than being transcribed by hand."""
import json, sys
from pathlib import Path

wide = json.loads(Path("tables_wide.json").read_text())
tight = json.loads(Path("tables_tight.json").read_text())

def block(t, scaffold, seeds, chance):
    a = t["arms"].get(f"{scaffold}_A"); b = t["arms"].get(f"{scaffold}_B"); c = t["arms"].get(f"{scaffold}_C")
    f = t["fisher"]
    def row(name, key, fmt=lambda v: v):
        return f"| {name} | {fmt(a[key])} | {fmt(b[key])} | {fmt(c[key])} |"
    def frac(x, n): return f"{x} / {n}"
    lines = [
        "| | arm A (flat only) | arm B (MAJ3 offered) | arm C (useless module) |",
        "|---|---|---|---|",
        row("candidates in scaffold", "scaffold_candidates"),
        f"| **success (exact conformance)** | **{frac(a['success'],a['n'])}** | **{frac(b['success'],b['n'])}** | **{frac(c['success'],c['n'])}** |",
        f"| Fisher exact vs arm A | — | **p = {f.get(f'{scaffold}_A_vs_B')}** | p = {f.get(f'{scaffold}_A_vs_C')} |",
        f"| module on the output path (chance {chance}) | — | {frac(b['module_on_output_path'],b['n'])} | {frac(c['module_on_output_path'],c['n'])} |",
        f"| **module on the output path, of successes** | — | **{frac(b['module_on_output_path_of_successes'],b['success'])}** | {frac(c['module_on_output_path_of_successes'],c['success'])} |",
        row("median steps to conformance", "median_steps_to_conformance"),
        row("median final relaxed loss", "median_final_loss"),
        row("lowest relaxed loss reached", "min_final_loss"),
        row("median seconds / optimizer step", "median_seconds_per_step"),
        row("median live nodes (all runs)", "median_live_nodes"),
        row("median live nodes (successes)", "median_live_nodes_successes"),
        row("median description bits, pruned (successes)", "median_description_bits_pruned_successes"),
        row("median execution cost, pruned (successes)", "median_execution_cost_pruned_successes"),
    ]
    return "\n".join(lines)

print("### WIDE\n")
print(block(wide, "wide", 8, "68.9%"))
print("\n### TIGHT\n")
print(block(tight, "tight", 24, "81.6%"))
print("\n### module mass trajectory (wide B)\n")
print(json.dumps(wide.get("module_mass_trajectory", {}).get("wide_B", {}), indent=1)[:1200])
print("\n### acquisition\n")
print(json.dumps(wide.get("acquisition", {}), indent=1))
