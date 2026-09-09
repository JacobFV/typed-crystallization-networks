"""Arms and compute accounting for the perturbation-selection comparison.

Both scheduler arms are the *shipped* `tcn.crystallize.Crystallizer`; they differ
only in the `selection` argument, so nothing here re-implements the scheduler.

    perturbation   the new default: DARTS-PT removal scoring selects the frozen
                   candidate and orders the nodes
    entropy        the superseded rule: readiness by softmax entropy plus argmax
                   stability, frozen candidate = argmax
    argmax         no crystallizer at all -- train, then freeze every node at its
                   argmax. Padded with extra optimizer steps on the crystallizer's
                   own retraining objective until its total matches a named
                   scheduler arm's realised total.
    argmax0        the same with no padding: strictly less compute than any arm.

Compute is counted three ways, all of them exact:

* `optimizer.step()` invocations, including steps taken inside freeze trials that
  are later discarded by rollback (track 1's standard);
* `SoftProgram.forward` invocations -- the honest cost of perturbation scoring,
  which buys its selection with extra forward passes and no extra gradients;
* objective evaluations (calls to the loss closure) and conformance evaluations
  (exact program executions), which are cheaper to compare across fixtures.
"""
from __future__ import annotations

from tcn.crystallize import Crystallizer


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


def make_scheduler(arm, model, optimizer, counter, tolerance):
    if arm.startswith("perturbation"):
        selection = "perturbation"
    elif arm.startswith("entropy"):
        selection = "entropy"
    else:
        raise ValueError(f"not a crystallizing arm: {arm}")
    return Instrumented(model, optimizer, selection=selection, tolerance=tolerance,
                        entropy_limit=0.9, counter=counter)


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


MIXED_ARMS = ("entropy", "perturbation", "argmax@entropy", "argmax@perturbation", "argmax0")
JOINT_ARMS = ("entropy", "perturbation", "perturbation-total-guard",
              "argmax@entropy", "argmax@perturbation", "argmax0")
LABELS = {
    "entropy": "entropy+stability selection (shipped rule)",
    "perturbation": "perturbation selection (new default)",
    "perturbation-total-guard": "perturbation, guard probes the regularized total",
    "argmax@entropy": "plain argmax, padded to the entropy arm's steps",
    "argmax@perturbation": "plain argmax, padded to the perturbation arm's steps",
    "argmax0": "plain argmax, no padding (less compute than any arm)",
}
