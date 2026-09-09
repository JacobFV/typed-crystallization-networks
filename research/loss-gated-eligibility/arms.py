"""Arms and compute accounting for the loss-gated-eligibility comparison.

Every crystallizing arm is the *shipped* `tcn.crystallize.Crystallizer` with
`selection="perturbation"`; they differ only in what puts the commitment on a
clock and what puts it on the task loss, so nothing here re-implements the
scheduler.

    perturbation       eligibility=immediate, anneal=round.  The scheduler as it
                       stands on branch `perturbation-selection` (commit 90dd08a).
                       rounds=24, its shipped cap.
    perturbation-cap   the same, run at the raised round cap the gated arms need.
                       Controls for the cap itself rather than the gate.
    gate               eligibility=plateau, anneal=round.  Loss-gated eligibility
                       alone.
    anneal             eligibility=immediate, anneal=plateau.  Loss-gated
                       temperature/quantization annealing alone.
    noanneal           eligibility=immediate, anneal=never.  The control that
                       separates "concentrate later" from "never concentrate".
    gate+anneal        both.
    gate+noanneal      eligibility=plateau, anneal=never.  Separates the gated
                       anneal from no anneal at all, on top of the gated eligibility.
    argmax             no crystallizer at all -- train, then freeze every node at
                       its argmax.  Padded with extra optimizer steps on the
                       crystallizer's own retraining objective until its total
                       matches a named scheduler arm's realised total.
    argmax0            the same with no padding: strictly less compute than any arm.

Compute is counted four ways, all of them exact:

* `optimizer.step()` invocations, including steps taken inside freeze trials that
  are later discarded by rollback, and including the residual training the gate
  spends on closed rounds (track 1's standard);
* `SoftProgram.forward` invocations -- the honest cost of perturbation scoring
  and of the gate's own loss readings;
* objective evaluations (calls to the loss closure).  On `joint` one objective
  evaluation is two environment rollouts, so this is the environment budget;
* the gate's own loss readings (`gate_evaluations`) separately from the
  perturbation sweep's (`sweep_evaluations`), so the gate's price is visible.
"""
from __future__ import annotations

from tcn.crystallize import Crystallizer

# The gate's parameters.  ARCHITECTURE.md section 5: "Temperature, thresholds,
# stability windows, block size, and precision are experiment configuration, not
# settled constants."  These are the values every number in RESULTS.md was taken
# at, and they were not tuned per fixture.
PLATEAU_WINDOW = 3        # rounds of task-loss readings compared against
PLATEAU_TOLERANCE = 1e-3  # relative improvement below which the loss has stopped
PLATEAU_PATIENCE = 12     # closed rounds after which the gate opens regardless

BASE_ROUNDS = 24          # the shipped cap, and what `perturbation` runs at
GATED_ROUNDS = 200        # gated arms spend rounds waiting, so they need more cap

CONFIG = {
    # arm: (eligibility, anneal, rounds)
    "perturbation":     ("immediate", "round",   BASE_ROUNDS),
    "perturbation-cap": ("immediate", "round",   GATED_ROUNDS),
    "gate":             ("plateau",   "round",   GATED_ROUNDS),
    "anneal":           ("immediate", "plateau", GATED_ROUNDS),
    "gate+anneal":      ("plateau",   "plateau", GATED_ROUNDS),
    # Control for the anneal gate: on these fixtures a gated anneal mostly does
    # not fire, so "concentrate later" has to be separated from "never
    # concentrate" before the gate can be credited with anything.
    "noanneal":         ("immediate", "never",   GATED_ROUNDS),
    "gate+noanneal":    ("plateau",   "never",   GATED_ROUNDS),
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
    """Records the optimizer steps each freeze trial consumed."""

    def __init__(self, *args, counter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = counter if counter is not None else {"n": 0}
        self.trial_steps = []

    def try_freeze(self, name, loss_fn, retrain_steps=20, conformance=None, index=None):
        before = self.counter["n"]
        event = super().try_freeze(name, loss_fn, retrain_steps, conformance, index)
        self.trial_steps.append(
            {"node": name, "accepted": event.accepted, "steps": self.counter["n"] - before}
        )
        return event

    def retained_steps(self):
        return sum(t["steps"] for t in self.trial_steps if t["accepted"])


def rounds_for(arm):
    return CONFIG[arm.replace("-total-guard", "")][2]


def make_scheduler(arm, model, optimizer, counter, tolerance):
    key = arm.replace("-total-guard", "")
    if key not in CONFIG:
        raise ValueError(f"not a crystallizing arm: {arm}")
    eligibility, anneal, _ = CONFIG[key]
    return Instrumented(model, optimizer, selection="perturbation", tolerance=tolerance,
                        eligibility=eligibility, anneal=anneal,
                        plateau_window=PLATEAU_WINDOW, plateau_tolerance=PLATEAU_TOLERANCE,
                        plateau_patience=PLATEAU_PATIENCE, counter=counter)


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


MIXED_SCHEDULERS = ("perturbation", "perturbation-cap", "gate", "anneal", "gate+anneal")
JOINT_SCHEDULERS = ("perturbation", "gate", "anneal", "gate+anneal")
# Which scheduler arms get a step-matched plain-argmax partner.
MIXED_PADDED = MIXED_SCHEDULERS
JOINT_PADDED = ("perturbation", "gate", "gate+anneal")

LABELS = {
    "perturbation": "perturbation, immediate eligibility + round anneal (branch 90dd08a)",
    "perturbation-cap": "the same at the gated arms' round cap (cap control)",
    "gate": "loss-gated eligibility, round anneal",
    "anneal": "immediate eligibility, loss-gated anneal",
    "gate+anneal": "loss-gated eligibility + loss-gated anneal",
    "noanneal": "immediate eligibility, no anneal at all (anneal-gate control)",
    "gate+noanneal": "loss-gated eligibility, no anneal at all (anneal-gate control)",
    "argmax@perturbation": "plain argmax, padded to `perturbation`'s steps",
    "argmax@perturbation-cap": "plain argmax, padded to `perturbation-cap`'s steps",
    "argmax@gate": "plain argmax, padded to `gate`'s steps",
    "argmax@anneal": "plain argmax, padded to `anneal`'s steps",
    "argmax@gate+anneal": "plain argmax, padded to `gate+anneal`'s steps",
    "argmax@noanneal": "plain argmax, padded to `noanneal`'s steps",
    "argmax@gate+noanneal": "plain argmax, padded to `gate+noanneal`'s steps",
    "argmax=fwd@noanneal": "plain argmax, matched to `noanneal`'s FORWARD passes",
    "argmax0": "plain argmax, no padding (less compute than any arm)",
}

ORDER = ("perturbation", "perturbation-cap", "gate", "anneal", "noanneal",
         "gate+anneal", "gate+noanneal",
         "argmax@perturbation", "argmax@perturbation-cap", "argmax@gate",
         "argmax@anneal", "argmax@noanneal", "argmax@gate+anneal",
         "argmax@gate+noanneal", "argmax=fwd@noanneal", "argmax0")
