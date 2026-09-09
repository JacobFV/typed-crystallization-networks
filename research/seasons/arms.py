"""Arms and compute accounting for the seasons comparison.

Every crystallizing arm is the `tcn.crystallize.Crystallizer` on this branch with
`selection="perturbation"`; they differ only in whether commitment is reversible.
Nothing here re-implements the scheduler.

    shipped     the incumbent: seasons=0, eligibility=immediate, anneal=round,
                rounds 24.  Monotone freezing, rollback only inside a rejected
                trial.  This is what ARCHITECTURE.md section 5 specifies today.
    seasons     seasons=3, winters of 8 rounds, prune on.  Between winters a
                summer releases part of the frozen set, warms it and retrains.
    argmax      no crystallizer at all -- train, then freeze every node at its
                argmax.  Padded with extra optimizer steps on the crystallizer's
                own objective until its total matches a named arm's realised
                total.  This is the thing to beat.
    argmax0     the same with no padding: strictly less compute than any arm.

The population arms live in `population.py`; they carry several members and are
accounted by the SUM over members, with `argmax-restarts` -- P independent argmax
runs of the same per-member budget, scored best-of-P -- as their control.  A
population that beats a single argmax but not best-of-P restarts has bought
parallel restarts, not selection.

Compute is counted four ways, all exact:

* `optimizer.step()` invocations, including steps inside trials later discarded
  by rollback, and including every summer's retraining (track 1's standard);
* `SoftProgram.forward` invocations -- the honest cost of perturbation scoring;
* objective evaluations.  On `joint` one objective evaluation is two environment
  rollouts, so this is the environment budget;
* thaws, seasons and pruned nodes, so the seasonal machinery's price is visible.
"""
from __future__ import annotations

from tcn.crystallize import Crystallizer

# ARCHITECTURE.md section 5: "Temperature, thresholds, stability windows, block
# size, and precision are experiment configuration, not settled constants."
# These are the values every number in RESULTS.md was taken at.  They were not
# tuned per fixture.
SEASONS = 3           # summers; the run always ends in a winter
SEASON_ROUNDS = 8     # scheduler rounds per winter
THAW_FRACTION = .5    # of the eligible frozen set, released at the first summer
THAW_DECAY = .5       # geometric decay of that fraction each summer
THAW_LIMIT = 1        # releases allowed per node before its commitment is final
WARM_TEMPERATURE = 1. # released nodes go back to the initial choice temperature
SUMMER_STEPS = None   # None -> 10 * retrain_steps

BASE_ROUNDS = 24      # the shipped cap

CONFIG = {
    # arm: (seasons, season_rounds, prune, close_block, rounds)
    "shipped":         (0, None, False, False, BASE_ROUNDS),
    "shipped-cap":     (0, None, False, False, SEASONS * SEASON_ROUNDS + BASE_ROUNDS),
    "seasons":         (SEASONS, SEASON_ROUNDS, True, True, BASE_ROUNDS),
    "seasons-noprune": (SEASONS, SEASON_ROUNDS, False, True, BASE_ROUNDS),
    "seasons-noblock": (SEASONS, SEASON_ROUNDS, True, False, BASE_ROUNDS),
    "prune":           (0, None, True, False, BASE_ROUNDS),
}


def count_steps(optimizer):
    """Shadow ``optimizer.step`` with a counting wrapper. Returns the counter."""
    counter = {"n": 0}
    original = optimizer.step

    def step(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    optimizer.step = step
    return counter


def count_forwards(model):
    """Shadow ``SoftProgram.forward`` on this instance. Returns the counter."""
    counter = {"n": 0}
    original = model.forward

    def forward(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    model.forward = forward
    return counter


class Instrumented(Crystallizer):
    """Records the optimizer steps each freeze trial and each summer consumed."""

    def __init__(self, *args, counter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = counter if counter is not None else {"n": 0}
        self.trial_steps = []
        self.summer_steps_spent = []

    def try_freeze(self, name, loss_fn, retrain_steps=20, conformance=None, index=None):
        before = self.counter["n"]
        event = super().try_freeze(name, loss_fn, retrain_steps, conformance, index)
        self.trial_steps.append(
            {"node": name, "accepted": event.accepted, "steps": self.counter["n"] - before}
        )
        return event

    def summer(self, objective, names, steps):
        before = self.counter["n"]
        out = super().summer(objective, names, steps)
        self.summer_steps_spent.append(self.counter["n"] - before)
        return out

    def retained_steps(self):
        return sum(t["steps"] for t in self.trial_steps if t["accepted"])


def rounds_for(arm):
    return CONFIG[arm][4]


def make_scheduler(arm, model, optimizer, counter, tolerance):
    if arm not in CONFIG:
        raise ValueError(f"not a crystallizing arm: {arm}")
    seasons, season_rounds, prune, close_block, _ = CONFIG[arm]
    return Instrumented(model, optimizer, selection="perturbation", tolerance=tolerance,
                        seasons=seasons, season_rounds=season_rounds, close_block=close_block,
                        thaw_fraction=THAW_FRACTION, thaw_decay=THAW_DECAY,
                        thaw_limit=THAW_LIMIT, warm_temperature=WARM_TEMPERATURE,
                        summer_steps=SUMMER_STEPS, prune=prune, counter=counter)


def scheduler_record(scheduler):
    return {"trial_steps": scheduler.trial_steps,
            "retained_trial_steps": scheduler.retained_steps(),
            "sweep_evaluations": scheduler.sweep_evaluations,
            "summer_steps": scheduler.summer_steps_spent,
            "seasons_run": len(scheduler.season_log),
            "thawed": [list(s.thawed) for s in scheduler.season_log],
            "thaw_count": sum(len(s.thawed) for s in scheduler.season_log),
            "settled": sorted(scheduler.settled),
            "pruned": list(scheduler.pruned),
            "block_events": [(e.node, e.accepted, e.reason) for e in scheduler.block_events]}


def freeze_all_argmax(model):
    """Round every choice logit to its argmax and declare the graph frozen."""
    for name, index in model.selections().items():
        if name not in model.frozen:
            model.freeze(name, index)
    return model.selections()


def reason_histogram(events):
    counts = {}
    for e in events:
        key = "accepted" if e.accepted else e.reason.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


SCHEDULERS = ("shipped", "seasons")
PADDED = ("shipped", "seasons")

LABELS = {
    "shipped": "the shipped crystallizer (monotone freezing, rollback within a trial)",
    "shipped-cap": "the shipped crystallizer at the seasonal round cap (cap control)",
    "seasons": "seasons: winters of freezing and pruning, summers that release and rewarm",
    "seasons-noprune": "seasons without pruning",
    "seasons-noblock": "seasons without the closing block trial",
    "prune": "monotone freezing plus dead-node pruning (prune control)",
    "population/seasons/pareto": "population of P, Pareto selection on (loss, description bits), seasons per member",
    "population/seasons/none": "the same P members with NO selection: P independent seasonal restarts",
    "population/argmax/pareto": "population of P under Pareto selection, each member frozen at argmax",
    "argmax@shipped": "plain argmax, padded to `shipped`'s optimizer steps",
    "argmax@seasons": "plain argmax, padded to `seasons`'s optimizer steps",
    "argmax@population": "plain argmax, padded to the population's TOTAL optimizer steps",
    "argmax-restarts": "best of P independent argmax runs (the population's compute control)",
    "argmax0": "plain argmax, no padding (less compute than any arm)",
}

ORDER = ("shipped", "shipped-cap", "prune", "seasons", "seasons-noprune", "seasons-noblock",
         "population/seasons/pareto", "population/seasons/none", "population/argmax/pareto",
         "argmax@shipped", "argmax@seasons", "argmax@population",
         "argmax-restarts", "argmax0")
