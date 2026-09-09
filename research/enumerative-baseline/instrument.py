"""Count the work the gradient path actually does, without touching tcn/.

Everything here is monkey-patching applied to *instances* and to
``torch.autograd`` entry points inside this process only.  ``tcn/`` and
``generators/`` are never modified.
"""
from __future__ import annotations

import torch

from tcn.learning import SoftProgram
from tcn.graph import Program


class Counter:
    def __init__(self):
        self.forward = 0
        self.forward_examples = 0
        self.backward = 0
        self.autograd_grad = 0
        self.exact_executions = 0
        self.optimizer_steps = 0


def instrument(model, counter, batch=1):
    """Wrap one SoftProgram instance's forward so every call is counted."""
    original = model.forward
    ops = sum(len(n.candidates) for n in model.program.nodes)

    def wrapped(inputs, state=None, return_trace=False):
        counter.forward += 1
        n = 1
        for v in inputs.values():
            if isinstance(v, torch.Tensor) and v.dim() > 1:
                n = max(n, v.shape[0])
        counter.forward_examples += n
        return original(inputs, state, return_trace)

    model.forward = wrapped
    model._ops_per_example = ops
    return model


class autograd_counting:
    """Context manager counting backward passes and torch.autograd.grad calls."""

    def __init__(self, counter):
        self.counter = counter

    def __enter__(self):
        self._backward = torch.autograd.backward
        self._grad = torch.autograd.grad
        self._execute = Program.execute
        c = self.counter

        def backward(*a, **k):
            c.backward += 1
            return self._backward(*a, **k)

        def grad(*a, **k):
            c.autograd_grad += 1
            return self._grad(*a, **k)

        def execute(self_, *a, **k):
            c.exact_executions += 1
            return self._execute(self_, *a, **k)

        torch.autograd.backward = backward
        torch.autograd.grad = grad
        Program.execute = execute
        return self.counter

    def __exit__(self, *a):
        torch.autograd.backward = self._backward
        torch.autograd.grad = self._grad
        Program.execute = self._execute
