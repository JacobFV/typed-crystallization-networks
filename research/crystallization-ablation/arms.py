"""Ablation arms for progressive crystallization (research track 1).

Nothing in `tcn/` or `generators/` is modified. Every variant is expressed as a
subclass of `tcn.crystallize.Crystallizer` or as a local re-implementation of
`tcn.synthesis.fit` that keeps the original control flow but adds
instrumentation (optimizer-step counting, per-trial step accounting).

Arms
----
A  full        the Crystallizer exactly as shipped
B  argmax      soft training only, then argmax every choice logit and freeze
C  shuffled    Crystallizer machinery, candidate order shuffled per round
D  no-retrain  Crystallizer machinery, retrain_steps=0
E  no-rollback Crystallizer machinery, tolerance=inf
F  perturb     Crystallizer machinery, DARTS-PT perturbation-based selection
               replacing the entropy/stability readiness ordering
G  accept-all  Crystallizer machinery, every trial accepted unconditionally
"""
from __future__ import annotations

import random

import torch

from tcn.crystallize import Crystallizer, FreezeEvent


def count_steps(optimizer):
    """Shadow ``optimizer.step`` with a counting wrapper. Returns the counter."""
    counter = {"n": 0}
    original = optimizer.step

    def step(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    optimizer.step = step
    return counter


class Instrumented(Crystallizer):
    """Records optimizer steps consumed by each freeze trial."""

    def __init__(self, *args, counter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = counter if counter is not None else {"n": 0}
        self.trial_steps = []

    def try_freeze(self, name, loss_fn, retrain_steps=20, conformance=None):
        before = self.counter["n"]
        event = super().try_freeze(name, loss_fn, retrain_steps, conformance)
        self.trial_steps.append(
            {"node": name, "accepted": event.accepted, "steps": self.counter["n"] - before}
        )
        return event

    def retained_steps(self):
        return sum(t["steps"] for t in self.trial_steps if t["accepted"])

    def trial_step_total(self):
        return sum(t["steps"] for t in self.trial_steps)


class ShuffledOrder(Instrumented):
    """Arm C: identical readiness test, randomized ordering instead of outside-in."""

    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = rng or random.Random(0)

    def candidates(self):
        names = list(super().candidates())
        self.rng.shuffle(names)
        return names


class PerturbationSelection(Instrumented):
    """Arm F: DARTS-PT-style selection.

    Rationale (Wang et al., ICLR 2021, "Rethinking Architecture Selection in
    Differentiable NAS"): the magnitude/concentration of a learned architecture
    distribution does not indicate an operation's contribution. The shipped
    `Crystallizer.candidates()` gates readiness on softmax entropy plus argmax
    stability -- exactly the falsified rule. This arm replaces it:

    * readiness/order: every unfrozen node is eligible; nodes are ordered by
      descending perturbation gap (second-best forced loss minus best forced
      loss), i.e. the node whose choice is most clearly determined goes first;
    * selection: the frozen index is the candidate whose forced hard execution
      yields the lowest objective, not the candidate with the largest logit.

    Residual retraining, the degradation tolerance, the gradient-connectivity
    guard, conformance and transactional rollback are all left intact.
    """

    def __init__(self, *args, decisions_first=True, mode="force", **kwargs):
        super().__init__(*args, **kwargs)
        self._loss = None
        self._best = {}
        self.sweep_evaluations = 0
        # DARTS-PT only ever selects at genuine decision points. A
        # single-candidate node carries no architecture decision, so with
        # decisions_first=True it is ordered last: hardening it early would
        # detach gradient paths before any real choice has been made. Arm F2
        # sets this False to isolate exactly that ordering effect.
        self.decisions_first = decisions_first
        # "force": score a candidate by the objective when it alone is executed
        #   (hard forward through the existing trial mechanism).
        # "remove": the literal DARTS-PT rule -- score a candidate by how much
        #   the objective DEGRADES when that candidate is deleted from the
        #   mixture; the most important candidate is the one whose removal
        #   hurts most.
        if mode not in ("force", "remove"):
            raise ValueError("unknown perturbation mode")
        self.mode = mode

    def run(self, loss_fn, rounds=12, retrain_steps=20, conformance=None):
        self._loss = loss_fn
        return super().run(loss_fn, rounds, retrain_steps, conformance)

    def _forced_loss(self, name, index):
        model = self.model
        saved = model.trials.get(name)
        model.trials[name] = index
        try:
            with torch.no_grad():
                value = float(self._loss().detach())
            self.sweep_evaluations += 1
        except (ValueError, OverflowError, RuntimeError):
            value = float("inf")
        finally:
            if saved is None:
                model.trials.pop(name, None)
            else:
                model.trials[name] = saved
        return value if value == value else float("inf")

    def _removal_loss(self, node_index, i):
        param = self.model.choices[node_index]
        saved = param.detach().clone()
        with torch.no_grad():
            param[i] = -1e9
        try:
            with torch.no_grad():
                value = float(self._loss().detach())
            self.sweep_evaluations += 1
        except (ValueError, OverflowError, RuntimeError):
            value = float("inf")
        finally:
            with torch.no_grad():
                param.copy_(saved)
        return value if value == value else float("inf")

    def candidates(self):
        model = self.model
        scored = []
        for index, node in enumerate(model.program.nodes):
            if node.name in model.frozen:
                continue
            if len(node.candidates) == 1:
                self._best[node.name] = 0
                scored.append((float("-inf") if self.decisions_first else float("inf"), node.name))
                continue
            if self.mode == "force":
                # lower forced loss is better
                scores = [-self._forced_loss(node.name, i) for i in range(len(node.candidates))]
            else:
                # higher loss on removal means the candidate mattered more
                scores = [self._removal_loss(index, i) for i in range(len(node.candidates))]
                scores = [s if s != float("inf") else float("-inf") for s in scores]
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            best = order[0]
            if scores[best] == float("-inf"):
                continue
            gap = scores[best] - scores[order[1]]
            self._best[node.name] = best
            scored.append((gap, node.name))
        return [name for _, name in sorted(scored, key=lambda x: -x[0])]

    def try_freeze(self, name, loss_fn, retrain_steps=20, conformance=None):
        index = self._best.get(name)
        if index is None:
            return super().try_freeze(name, loss_fn, retrain_steps, conformance)
        model = self.model
        original = model.selections
        model.selections = lambda: {**original(), name: index}
        try:
            return super().try_freeze(name, loss_fn, retrain_steps, conformance)
        finally:
            model.selections = original


class AcceptAll(Instrumented):
    """Arm G: no transaction at all. Every trial is committed."""

    def try_freeze(self, name, loss_fn, retrain_steps=20, conformance=None):
        before_count = self.counter["n"]
        model = self.model
        index = model.selections()[name]
        try:
            before = float(loss_fn().detach())
        except Exception:  # noqa: BLE001 - measuring failure modes is the point
            before = float("nan")
        reason = "forced"
        after = float("nan")
        try:
            model.trials[name] = index
            for _ in range(retrain_steps):
                self.optimizer.zero_grad()
                loss = loss_fn()
                if loss.requires_grad:
                    loss.backward()
                    self.optimizer.step()
            model.freeze(name, index)
            after = float(loss_fn().detach())
        except (ValueError, OverflowError, RuntimeError) as exc:
            reason = f"forced after error: {exc}"
            model.trials.pop(name, None)
            model.frozen[name] = index
        event = FreezeEvent(name, True, before, after, reason)
        self.events.append(event)
        self.trial_steps.append(
            {"node": name, "accepted": True, "steps": self.counter["n"] - before_count}
        )
        return event


def freeze_all_argmax(model):
    """Arm B: round every choice logit to its argmax and declare the graph frozen."""
    selections = model.selections()
    for name, index in selections.items():
        if name not in model.frozen:
            model.freeze(name, index)
    return selections


def make_scheduler(arm, model, optimizer, counter, seed, tolerance, entropy_limit):
    """Return (scheduler, retrain_steps_override) for a crystallizing arm."""
    if arm in ("A", "H"):
        # H is A with the conformance callback disabled by the caller.
        return Instrumented(model, optimizer, tolerance=tolerance,
                            entropy_limit=entropy_limit, counter=counter), None
    if arm == "C":
        return ShuffledOrder(model, optimizer, tolerance=tolerance,
                             entropy_limit=entropy_limit, counter=counter,
                             rng=random.Random(10_000 + seed)), None
    if arm == "D":
        return Instrumented(model, optimizer, tolerance=tolerance,
                            entropy_limit=entropy_limit, counter=counter), 0
    if arm == "E":
        return Instrumented(model, optimizer, tolerance=float("inf"),
                            entropy_limit=entropy_limit, counter=counter), None
    if arm in ("F", "F2", "FR"):
        return PerturbationSelection(model, optimizer, tolerance=tolerance,
                                     entropy_limit=entropy_limit, counter=counter,
                                     decisions_first=(arm != "F2"),
                                     mode="remove" if arm == "FR" else "force"), None
    if arm == "G":
        return AcceptAll(model, optimizer, tolerance=tolerance,
                         entropy_limit=entropy_limit, counter=counter), None
    raise ValueError(f"unknown crystallizing arm {arm}")


ARMS = ("A", "B", "B0", "C", "D", "E", "F", "F2", "FR", "G", "H")
ARM_LABELS = {
    "A": "A full crystallizer",
    "B": "B naive argmax (budget-matched)",
    "B0": "B0 naive argmax (unpadded control)",
    "C": "C shuffled freeze order",
    "D": "D no residual retraining",
    "E": "E no rollback (tolerance=inf)",
    "F": "F perturbation-based selection (decisions first)",
    "F2": "F2 perturbation-based selection (plumbing first)",
    "FR": "FR perturbation-by-removal (literal DARTS-PT rule)",
    "G": "G accept-all (no transaction)",
    "H": "H full crystallizer, conformance check disabled",
}
