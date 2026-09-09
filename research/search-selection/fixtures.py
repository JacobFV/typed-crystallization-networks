"""What `synthesis.fit(mode="auto")` chooses on every fixture in the repository.

Through the public entry point, not through `select_backend` directly, so this
also checks the wiring: that the default is untouched, that the report shape is
the same whichever backend runs, and that the returned model exports a frozen
program either way.

The repository's shipped fixtures are `examples/mixed.py` and `examples/joint.py`.
The remaining rows are the scaffolds `research/FINDINGS.md` records outcomes for,
reconstructed in `common.py`, so that "every fixture" means every scaffold there
is a measured right answer for and not only the two that ship.

Run:  PYTHONPATH=. .venv/bin/python research/search-selection/fixtures.py
"""
from __future__ import annotations
import time

from common import CASES, dump
from tcn.synthesis import fit


def default_is_untouched():
    """`mode` defaults to the shipped gradient path and reports no selection."""
    from examples.mixed import problem
    p, signals, examples = problem()
    t = time.perf_counter()
    _, report = fit(p, examples, signals, steps=300, tolerance=.005)
    return {"fixture": "examples/mixed.py", "mode": "default",
            "selection": report["selection"], "exact_max_error": report["exact_max_error"],
            "relaxed_loss": report["relaxed_loss"], "fully_frozen": report["fully_frozen"],
            "seconds": round(time.perf_counter() - t, 3),
            "matches_FINDINGS_section_10": abs(report["relaxed_loss"] - 1.007e-06) < 5e-9
                                           and report["exact_max_error"] == 0.}


def auto(name, case, steps=200):
    t = time.perf_counter()
    model, report = fit(case.program, case.train, case.signals, steps=steps, tolerance=case.tolerance,
                        mode="auto", validation=case.validation, rollout_cost=case.rollout_cost,
                        freeze=False)
    d = report["selection"]
    return {"fixture": name, "family": case.family, "space": case.space,
            "chosen": d["mode"], "right": case.right, "agrees": d["mode"] == case.right,
            "reason": d["reason"], "certified": d["certified"], "feasible": d["feasible"],
            "reachable_fraction": d["reachable_fraction"], "dead_nodes": d["dead_nodes"],
            "scale_surrogates": d["scale_surrogates"],
            "exact_max_error": report["exact_max_error"],
            "exact_conformance": report["exact_conformance"],
            "fully_frozen": report["fully_frozen"],
            "report_keys": sorted(report),
            "seconds": round(time.perf_counter() - t, 3)}


def main():
    t0 = time.perf_counter()
    out = {"default": default_is_untouched(), "auto": []}
    print("default path:", out["default"], flush=True)
    for name, make in CASES.items():
        case = make()
        if case.space > 1 << 20 and case.right != "relax":
            continue
        try:
            row = auto(name, case)
        except Exception as e:
            row = {"fixture": name, "error": f"{type(e).__name__}: {e}"}
        out["auto"].append(row)
        print(f"  {name:20s} chose {row.get('chosen'):10s} right={row.get('right'):10s} "
              f"agrees={row.get('agrees')} exact={row.get('exact_max_error')} "
              f"in {row.get('seconds')}s", flush=True)
    agree = sum(bool(r.get("agrees")) for r in out["auto"])
    out["agreement"] = f"{agree}/{len(out['auto'])}"
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    print("agreement with the tracks:", out["agreement"], flush=True)
    dump("fixtures", out)


if __name__ == "__main__":
    main()
