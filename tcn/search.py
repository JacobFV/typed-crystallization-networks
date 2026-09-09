"""Exhaustive discrete search over the same candidate space relaxation searches.

Enumeration is the reference every synthesis claim is measured against. It uses
only `Program.execute` over the node candidates already declared, so it searches
exactly the space `SoftProgram` relaxes -- no separate encoding, no extra
operators, no domain knowledge. When it exhausts the space it also reports
whether a solution is unique, which a gradient run cannot establish.

It applies to the discrete choice only. A program with trainable constants has a
continuous part that enumeration does not address; `enumerate_fit` reports that
rather than silently searching a subspace.
"""
from dataclasses import dataclass
import itertools
import math
import time
from .operators import Registry

@dataclass(frozen=True)
class SearchResult:
    solved: bool
    selections: dict | None
    exact_max_error: float
    evaluated: int
    space_size: int
    exhausted: bool
    unique: bool | None
    seconds: float
    continuous: tuple[str, ...] = ()
    def to_dict(self):
        return {'solved':self.solved,'selections':self.selections,'exact_max_error':self.exact_max_error,
                'evaluated':self.evaluated,'space_size':self.space_size,'exhausted':self.exhausted,
                'unique':self.unique,'seconds':self.seconds,'continuous':list(self.continuous)}

def candidate_counts(program):
    return tuple(len(n.candidates) for n in program.nodes)

def space_size(program):
    """Number of distinct discrete programs the scaffold admits."""
    return math.prod(candidate_counts(program))

def evaluate(program, selections, examples, signals, registry, tolerance=None):
    """Largest absolute error of one discrete selection, or None if it cannot run.

    An illegal numeric domain (a division by zero, a logarithm of a
    non-positive value) is a property of that candidate, not an error in the
    search: report it as unusable and continue.
    """
    worst = 0.
    for example in examples:
        try:
            _, _, trace = program.execute(example['inputs'], registry=registry, selections=selections)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError):
            return None
        for signal in signals:
            a = trace[signal.source].flat(); b = example['targets'][signal.target].flat()
            worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
        if tolerance is not None and worst > tolerance:
            return worst
    return worst

def enumerate_fit(program, examples, signals, registry=None, tolerance=.001, max_programs=1 << 20, stop_at_first=False):
    """Search every discrete program in the scaffold, exactly.

    `stop_at_first` returns as soon as a conforming program is found, which
    forfeits the uniqueness certificate.
    """
    if not examples: raise ValueError('training examples required')
    r = registry or Registry()
    program.validate(r); program.validate_signals(signals)
    counts = candidate_counts(program); total = math.prod(counts)
    names = [n.name for n in program.nodes]
    started = time.perf_counter(); evaluated = 0; found = []; best = float('inf')
    for combination in itertools.product(*(range(c) for c in counts)):
        if evaluated >= max_programs: break
        selections = dict(zip(names, combination)); evaluated += 1
        error = evaluate(program, selections, examples, signals, r, tolerance)
        if error is None: continue
        best = min(best, error)
        if error <= tolerance:
            found.append(selections)
            if stop_at_first: break
    exhausted = evaluated >= total
    return SearchResult(bool(found), found[0] if found else None,
                        0. if not found else min(best, tolerance), evaluated, total, exhausted,
                        (len(found) == 1) if exhausted and not stop_at_first else None,
                        time.perf_counter() - started, tuple(program.trainable_constants))
