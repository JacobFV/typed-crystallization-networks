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
    conforming: int = 0
    ranked_by: str = 'order'
    description_bits: int | None = None
    execution_cost: float | None = None
    def to_dict(self):
        return {'solved':self.solved,'selections':self.selections,'exact_max_error':self.exact_max_error,
                'evaluated':self.evaluated,'space_size':self.space_size,'exhausted':self.exhausted,
                'unique':self.unique,'seconds':self.seconds,'continuous':list(self.continuous),
                'conforming':self.conforming,'ranked_by':self.ranked_by,
                'description_bits':self.description_bits,'execution_cost':self.execution_cost}

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
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError, IndexError):
            # IndexError belongs here too: an address that runs off the end of a
            # tuple is the normal case for a window operator at an image border,
            # and is a property of that candidate. Without it one such candidate
            # aborted the whole sweep.
            return None
        for signal in signals:
            a = trace[signal.source].flat(); b = example['targets'][signal.target].flat()
            worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
        if tolerance is not None and worst > tolerance:
            return worst
    return worst

def program_cost(program, selections, registry):
    """Description bits and execution cost of one selection, after pruning.

    Hardening keeps the whole scaffold, so an unpruned program's size and cost
    describe the search space rather than the chosen program. Description bits
    charge each distinct frozen module definition once and every call site
    individually, exactly as `Program.description_bits` does.
    """
    chosen = program.harden(selections).pruned()
    return chosen.description_bits(registry), chosen.execution_cost(registry)

def enumerate_fit(program, examples, signals, registry=None, tolerance=.001, max_programs=1 << 20, stop_at_first=False, rank='order'):
    """Search every discrete program in the scaffold, exactly.

    `stop_at_first` returns as soon as a conforming program is found, which
    forfeits the uniqueness certificate.

    `rank` decides *which* conforming program is returned, which is a separate
    question from whether one exists. `'order'` returns the first in enumeration
    order, so it answers "is there a program"; `'description'` returns the one
    with the fewest description bits and `'cost'` the one with the lowest
    execution cost, each tie-broken by the other, so they answer "what is the
    best program". Ranking needs the whole conforming set and is therefore
    incompatible with `stop_at_first`. `conforming` reports how many were found.
    """
    if rank not in {'order','description','cost'}: raise ValueError('unknown ranking')
    if rank != 'order' and stop_at_first: raise ValueError('ranking requires the full conforming set')
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
    chosen = found[0] if found else None
    bits = cost = None
    if found:
        if rank == 'order':
            bits, cost = program_cost(program, chosen, r)
        else:
            scored = [(s,) + program_cost(program, s, r) for s in found]
            key = (lambda x: (x[1], x[2])) if rank == 'description' else (lambda x: (x[2], x[1]))
            chosen, bits, cost = min(scored, key=key)
    return SearchResult(bool(found), chosen,
                        0. if not found else min(best, tolerance), evaluated, total, exhausted,
                        (len(found) == 1) if exhausted and not stop_at_first else None,
                        time.perf_counter() - started, tuple(program.trainable_constants),
                        len(found), rank, bits, cost)
