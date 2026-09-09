"""Shared accounting for the enumerative / random / SAT baselines.

Two evaluation counters are reported everywhere, because the two sides of the
comparison do different units of work:

- ``programs``: complete discrete candidate programs evaluated.  One
  enumeration step is one program.  A gradient step is *not* one program, so
  this number is not directly comparable across methods and is reported for the
  discrete methods only.
- ``op_applications``: elementwise applications of one candidate operator to one
  example.  This *is* comparable.  Enumerating a P-node program over N examples
  with C combinations costs ``C * P * N``.  One forward pass of the soft graph
  over a batch of N examples costs ``sum_v |K_v| * N`` because every node
  evaluates every legal candidate (``tcn/learning.py`` line 100).  A backward
  pass is charged at 2x the forward (standard reverse-mode accounting).

Wall clock is measured with ``time.perf_counter`` around the search itself,
excluding problem construction, on a single torch thread so that the gradient
side and the enumerative side get the same CPU.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


@dataclass
class Budget:
    programs: int = 0
    op_applications: int = 0
    env_steps: int = 0
    gradient_steps: int = 0

    def add(self, programs=0, op_applications=0, env_steps=0, gradient_steps=0):
        self.programs += programs
        self.op_applications += op_applications
        self.env_steps += env_steps
        self.gradient_steps += gradient_steps


@dataclass
class Result:
    method: str
    task: str
    solved: bool
    seconds: float
    budget: Budget = field(default_factory=Budget)
    detail: dict = field(default_factory=dict)

    def row(self):
        d = asdict(self)
        d["budget"] = asdict(self.budget)
        return d


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t


def soft_forward_ops(program, examples):
    """Candidate-operator applications in ONE forward pass of the soft graph."""
    return sum(len(n.candidates) for n in program.nodes) * examples


def write(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def machine():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
