"""A population of candidate programs per problem, on the mixed fixture.

The proposal: rather than one program per problem, carry several concurrently
with selection pressure toward both the LIGHTER and the HIGHER-PERFORMING ones.
`research/abstraction-preference/RESULTS.md` is the motivation: a single scalar
description-cost weight is aligned with the target on an over-provisioned
scaffold (7/8 -> 8/8 minimal programs) and opposed to it on an exactly-sized one
(19/24 -> 2/24 conformance).  A population does not have to choose a weight.

**The selection rule, stated.** At the end of each generation every member is
scored on two numbers: `L`, the probe loss it is being trained on, and `D`,
`SoftProgram.description_cost()` -- the expected description length in bits of
the pruned hardened program, which at any one-hot distribution is exactly
`export().pruned().description_bits()`.  Member i DOMINATES member j when
`L_i <= L_j` and `D_i <= D_j` with at least one strict.  The non-dominated front
survives unchanged.  Every dominated member is replaced by a copy of a uniformly
chosen front member with fresh Gaussian noise (std `sigma`) added to its choice
logits and a fresh optimizer.  No weight is chosen between the two objectives and
neither is ever traded away: a member that is lightest survives even if it is the
worst performer, and vice versa.  That is the whole point -- it keeps the light
candidate AND the conforming one instead of picking a weight.

**The final answer.** Among members that conform exactly, the one with the fewest
pruned description bits; if none conform, the lowest-loss member.  `any_conformant`
is reported separately, because "some member conforms" is what P independent
restarts also buy and must not be confused with what selection buys.

**The control.** `--select none` runs the same P members with no selection at
all: P independent restarts.  With `--arm argmax` that is `argmax-restarts`, the
best-of-P baseline the population has to beat.  Compute is summed over members
and reported as one total, so a population of 4 is charged 4x.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcn.crystallize import Objective  # noqa: E402

from arms import (count_forwards, count_steps, freeze_all_argmax,  # noqa: E402
                  make_scheduler, reason_histogram, rounds_for, scheduler_record)
from run_mixed import (LR, RETRAIN_STEPS, STEPS, TOLERANCE, build,  # noqa: E402
                       closures)


class Member:
    def __init__(self, index, seed, noise):
        self.index = index
        (self.program, self.signals, self.examples, self.model, self.optimizer,
         self.inputs, self.targets) = build(seed, noise)
        self.evals = {"loss": 0, "conform": 0}
        self.loss_fn, self.exact_error = closures(self.model, self.signals, self.examples,
                                                  self.inputs, self.targets, self.evals)
        self.steps = count_steps(self.optimizer)
        self.forwards = count_forwards(self.model)
        self.ancestry = [index]

    def score(self):
        with torch.no_grad():
            return float(self.loss_fn().detach()), float(self.model.description_cost().detach())

    def reseed_from(self, other, sigma):
        """Replace this member's parameters with a noised copy of `other`'s."""
        self.model.load_state_dict(copy.deepcopy(other.model.state_dict()))
        self.model.temperatures = dict(other.model.temperatures)
        self.model.surrogate_scale = dict(other.model.surrogate_scale)
        with torch.no_grad():
            for logits in self.model.choices:
                logits.add_(torch.randn(logits.shape) * sigma)
        # A fresh optimizer: Adam's moments belong to the trajectory that was
        # culled, and carrying them across a reseed would drag the copy back.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.optimizer.step = _rebind(self.optimizer.step, self.steps)
        self.ancestry = other.ancestry + [self.index]


def _rebind(original, counter):
    def step(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)
    return step


def dominated(scores):
    """Indices of members dominated by at least one other on (loss, description bits)."""
    out = []
    for i, (li, di) in enumerate(scores):
        for j, (lj, dj) in enumerate(scores):
            if i == j:
                continue
            if lj <= li and dj <= di and (lj < li or dj < di):
                out.append(i)
                break
    return out


def run(arm, seed, members=4, generations=4, steps=STEPS, select="pareto", sigma=.5,
        noise=.5, tolerance=TOLERANCE):
    torch.set_num_threads(1)
    pop = [Member(m, seed * 1000 + m, noise) for m in range(members)]
    generator = torch.Generator().manual_seed(seed * 7919 + 13)
    per = max(1, steps // max(1, generations))
    log = []
    start = time.perf_counter()

    step_index = 0
    for gen in range(generations):
        for member in pop:
            for _ in range(per):
                member.optimizer.zero_grad()
                loss = (member.loss_fn()
                        + 0.001 * ((step_index + _) / max(1, steps)) * member.model.entropy())
                if loss.requires_grad:
                    loss.backward()
                    member.optimizer.step()
        step_index += per
        scores = [m.score() for m in pop]
        entry = {"generation": gen, "scores": scores}
        if select == "pareto" and gen < generations - 1:
            losers = dominated(scores)
            front = [i for i in range(len(pop)) if i not in losers]
            for i in losers:
                parent = front[int(torch.randint(len(front), (1,), generator=generator))]
                pop[i].reseed_from(pop[parent], sigma)
            entry["front"] = front
            entry["reseeded"] = losers
        log.append(entry)

    rows = []
    for member in pop:
        row = {"member": member.index, "ancestry": member.ancestry}
        if arm == "argmax":
            freeze_all_argmax(member.model)
            row["reasons"] = {}
        else:
            scheduler = make_scheduler(arm, member.model, member.optimizer, member.steps, tolerance)
            try:
                scheduler.run(Objective(member.loss_fn, member.loss_fn), rounds=rounds_for(arm),
                              retrain_steps=RETRAIN_STEPS,
                              conformance=lambda exact, m=member: m.exact_error(exact) <= tolerance)
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"scheduler: {exc}"
            row["reasons"] = reason_histogram(scheduler.events)
            row.update(scheduler_record(scheduler))
        n = len(member.program.nodes)
        row["frozen"] = len(member.model.frozen)
        row["fully_frozen"] = len(member.model.frozen) == n
        try:
            err = member.exact_error(member.model.export())
            row["exact_max_error"] = err
            row["exact_conformance"] = err <= tolerance
            row["description_bits"] = member.model.export().pruned().description_bits(member.model.registry)
        except Exception as exc:  # noqa: BLE001
            row["exact_max_error"] = None
            row["exact_conformance"] = False
            row["description_bits"] = None
            row["export_error"] = str(exc)
        row["loss"] = float(member.loss_fn().detach())
        row["steps"] = member.steps["n"]
        row["forwards"] = member.forwards["n"]
        row["loss_evaluations"] = member.evals["loss"]
        row["selections"] = member.model.selections()
        rows.append(row)

    conforming = [r for r in rows if r["exact_conformance"]]
    # The population's answer: lightest among the conformant, else lowest loss.
    chosen = (min(conforming, key=lambda r: (r["description_bits"], r["member"])) if conforming
              else min(rows, key=lambda r: (r["loss"], r["member"])))
    record = {"task": "mixed", "arm": f"population/{arm}/{select}", "seed": seed,
              "members": members, "generations": generations, "steps": steps,
              "select": select, "sigma": sigma, "init_noise": noise, "member_arm": arm,
              "generation_log": log, "member_rows": rows,
              "chosen_member": chosen["member"],
              "exact_conformance": chosen["exact_conformance"],
              "exact_max_error": chosen["exact_max_error"],
              "description_bits": chosen["description_bits"],
              "loss": chosen["loss"],
              "fully_frozen": chosen["fully_frozen"],
              "frozen": chosen["frozen"],
              "any_conformant": bool(conforming),
              "conformant_members": len(conforming),
              "min_description_bits": min([r["description_bits"] for r in rows
                                           if r["description_bits"] is not None], default=None),
              "total_steps": sum(r["steps"] for r in rows),
              "total_forwards": sum(r["forwards"] for r in rows),
              "loss_evaluations": sum(r["loss_evaluations"] for r in rows),
              "wall": time.perf_counter() - start}
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="seasons", help="crystallizer arm per member, or 'argmax'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--select", default="pareto", choices=("pareto", "none"))
    ap.add_argument("--sigma", type=float, default=.5)
    ap.add_argument("--noise", type=float, default=.5)
    ap.add_argument("--out")
    args = ap.parse_args()
    record = run(args.arm, args.seed, members=args.members, generations=args.generations,
                 steps=args.steps, select=args.select, sigma=args.sigma, noise=args.noise)
    text = json.dumps(record)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
