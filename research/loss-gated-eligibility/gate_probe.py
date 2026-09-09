"""What the gate changes about *when* the measurement is taken, on `joint`.

`research/perturbation-selection/RESULTS.md` section 6 measured the perturbation
score and the argmax at one instant -- the end of base training, with nothing
frozen -- and found them to disagree badly at 40 episodes (argmax picks the
reference truth table 0/16, perturbation 6/16). The scheduler acts on that
instant. This probe asks the follow-on question: when the plateau gate defers the
commitment, what do the two rules say at the moment the gate actually opens?

For each seed it runs the gated scheduler and records, at every gate opening, the
round, the task loss, and for each of the two real decision nodes still unfrozen
the argmax candidate and the perturbation-selected candidate. Table 6 is the
reference program the probe objective identifies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from examples.joint import trainer  # noqa: E402
from tcn.crystallize import Objective  # noqa: E402

from arms import (PLATEAU_PATIENCE, PLATEAU_TOLERANCE, PLATEAU_WINDOW,  # noqa: E402
                  Instrumented, count_steps)

DECISIONS = ("relation", "goal_relation")
# The incumbent scheduler, and the fully gated one -- the two arms section 3
# compares. `gate` (plateau eligibility, fixed-clock anneal) is left out because
# section 3 shows it is a broken configuration, not a candidate.
SCHEDULES = (("immediate", "round"), ("plateau", "plateau"))
REFERENCE = 6
TOLERANCE = 0.05
RETRAIN_STEPS = 2
ROUNDS = 200


class Probing(Instrumented):
    """Records what each rule says at every gate opening, before any freeze runs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openings = []

    def candidates(self, loss=None):
        order = super().candidates(loss)
        argmax = self.model.selections()
        self.openings.append({
            "round": len(self.gate_log),
            "task_loss": self.progress[-1] if self.progress else None,
            "frozen": sorted(self.model.frozen),
            "decisions": {n: {"argmax": argmax[n], "perturbation": self.selected.get(n)}
                          for n in DECISIONS if n not in self.model.frozen},
        })
        return order


def run(seed, episodes, eligibility, anneal):
    torch.set_num_threads(1)
    t = trainer(episodes, seed=seed)
    counter = count_steps(t.optimizer)
    t.run(None)

    def validation(regularized=True):
        return sum(t.episode(20000 + i, False, "validation", loss_only=True, regularized=regularized)
                   for i in range(2)) / 2

    scheduler = Probing(t.model, t.optimizer, selection="perturbation", tolerance=TOLERANCE,
                        eligibility=eligibility, anneal=anneal, plateau_window=PLATEAU_WINDOW,
                        plateau_tolerance=PLATEAU_TOLERANCE, plateau_patience=PLATEAU_PATIENCE,
                        counter=counter)
    scheduler.run(Objective(validation, lambda: validation(False)), rounds=ROUNDS,
                  retrain_steps=RETRAIN_STEPS)
    return {"seed": seed, "episodes": episodes, "eligibility": eligibility, "anneal": anneal,
            "openings": scheduler.openings, "selections": t.model.selections(),
            "rounds": len(scheduler.gate_log), "total_steps": counter["n"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[40, 80, 160])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "gate-probe.json"))
    args = ap.parse_args()

    records = []
    for episodes in args.episodes:
        for eligibility, anneal in SCHEDULES:
            for seed in range(args.seeds):
                records.append(run(seed, episodes, eligibility, anneal))
                print(f"{episodes} {eligibility}/{anneal} seed {seed} done", flush=True)

    print("\n| episodes | schedule | first-commitment decisions | argmax = table 6 | "
          "perturbation = table 6 | agree | final program = table 6 |")
    print("|---|---|---|---|---|---|---|")
    for episodes in args.episodes:
        for eligibility, anneal in SCHEDULES:
            rs = [r for r in records if r["episodes"] == episodes
                  and r["eligibility"] == eligibility and r["anneal"] == anneal]
            # The first opening at which each decision node is still unfrozen: the
            # instant the scheduler's own commitment for that node is measured.
            seen = []
            for r in rs:
                first = {}
                for opening in r["openings"]:
                    for name, d in opening["decisions"].items():
                        first.setdefault(name, d)
                seen.extend(first.values())
            n = len(seen)
            final = sum(1 for r in rs if all(r["selections"].get(k) == REFERENCE for k in DECISIONS))
            print("| {} | {}/{} | {} | {}/{} | {}/{} | {}/{} | {}/{} |".format(
                episodes, eligibility, anneal, n,
                sum(1 for d in seen if d["argmax"] == REFERENCE), n,
                sum(1 for d in seen if d["perturbation"] == REFERENCE), n,
                sum(1 for d in seen if d["argmax"] == d["perturbation"]), n,
                final, len(rs)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
