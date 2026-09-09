"""Exhaustive discrete search over the same candidate space relaxation searches.

Enumeration is the reference every synthesis claim is measured against. It uses
only `Program.execute` over the node candidates already declared, so it searches
exactly the space `SoftProgram` relaxes -- no separate encoding, no extra
operators, no domain knowledge. When it exhausts the space it also reports
whether a solution is unique, which a gradient run cannot establish.

It applies to the discrete choice only. A program with trainable constants has a
continuous part that enumeration does not address; `enumerate_fit` reports that
rather than silently searching a subspace.

Four scoring modes share one `SearchResult` contract:

* `enumerate_fit`         -- feed-forward, one execution per example, probe targets.
* `enumerate_recurrent`   -- `Program.state` driven for a declared tick budget,
                             scored on per-tick targets or a trailing settle window.
* `enumerate_environment` -- actual rollouts of the frozen candidate through a
                             generator, scored by summed reward components, with
                             environment episodes reported as a first-class cost.
* `enumerate_prefix`      -- the same conforming set as `enumerate_fit`, walked as
                             a DFS over the choice tree so a node value is computed
                             once per prefix rather than once per leaf, with an
                             optional beam width that bounds the walk and, when it
                             actually discards anything, forfeits the certificate.

Every result carries an explicit `certificate` field, because the uniqueness
certificate is this backend's main advantage over a gradient run and losing it
silently would be worse than not having it. `'unique'` means the space was
exhausted and exactly one program conforms; `'complete'` means the space was
exhausted and the reported conforming set is the whole of it; `'none'` means the
space was not exhausted -- a budget, an early exit, or a beam that discarded
candidates -- so neither existence-completeness nor uniqueness is established.
"""
from dataclasses import dataclass, field, replace
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
    # `certificate` states in one word what the sweep established: 'unique' (the
    # space was exhausted and exactly one program conforms), 'complete' (the space
    # was exhausted and the conforming set reported is the whole of it, whether it
    # has zero members or many), or 'none' (a budget, an early exit or a beam left
    # part of the space unexamined, so nothing is established about what was not
    # looked at). It exists so that forfeiting the certificate cannot be silent.
    certificate: str = 'none'
    mode: str = 'fit'
    # Costs. `evaluated` counts discrete programs *decided* -- a prefix walk that
    # rejects a subtree decides every leaf under it -- so `evaluated == space_size`
    # remains the exhaustion test in every mode. `episodes`/`steps` are the
    # environment resource, which is the one axis on which relaxation was measured
    # to win. `node_evaluations` is operator work: counted exactly in the prefix
    # and beam modes, and reported as the sweep's nominal size
    # `programs * nodes * examples (* ticks)` in the flat modes, which is an upper
    # bound because `evaluate` abandons an example once the tolerance is exceeded.
    node_evaluations: int = 0
    episodes: int = 0
    steps: int = 0
    # Recurrent scoring configuration, echoed so a result is self-describing.
    ticks: int = 0
    settle_window: int = 0
    # Environment scoring: the best mean return observed, and the acceptance
    # threshold it was compared against (None = "best in the enumerated set").
    best_return: float | None = None
    threshold: float | None = None
    # Beam bookkeeping. `discarded` counts programs the beam dropped without
    # deciding them; it is nonzero exactly when the certificate was forfeited by
    # the beam rather than by a budget.
    beam: int | None = None
    discarded: int = 0
    def to_dict(self):
        return {'solved':self.solved,'selections':self.selections,'exact_max_error':self.exact_max_error,
                'evaluated':self.evaluated,'space_size':self.space_size,'exhausted':self.exhausted,
                'unique':self.unique,'seconds':self.seconds,'continuous':list(self.continuous),
                'conforming':self.conforming,'ranked_by':self.ranked_by,
                'description_bits':self.description_bits,'execution_cost':self.execution_cost,
                'certificate':self.certificate,'mode':self.mode,
                'node_evaluations':self.node_evaluations,'episodes':self.episodes,'steps':self.steps,
                'ticks':self.ticks,'settle_window':self.settle_window,
                'best_return':self.best_return,'threshold':self.threshold,
                'beam':self.beam,'discarded':self.discarded}

def certificate_of(exhausted, complete, conforming):
    """One word for what a sweep established. See `SearchResult.certificate`."""
    if not exhausted or not complete: return 'none'
    return 'unique' if conforming == 1 else 'complete'

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
    complete = not stop_at_first
    return SearchResult(bool(found), chosen,
                        0. if not found else min(best, tolerance), evaluated, total, exhausted,
                        (len(found) == 1) if exhausted and complete else None,
                        time.perf_counter() - started, tuple(program.trainable_constants),
                        len(found), rank, bits, cost,
                        certificate_of(exhausted, complete, len(found)), 'fit',
                        node_evaluations=evaluated * len(program.nodes) * len(examples))


# ---------------------------------------------------------------------------
# mode: recurrence
# ---------------------------------------------------------------------------
#
# SEMANTICS, stated rather than guessed.
#
# A recurrent candidate is scored on a **declared tick budget with a declared
# trailing settle window**, not by running until its state stops changing.
#
# Both readings were available. "Run until stable" is the more flattering one and
# it was rejected for three reasons. It needs a stability detector, and that
# detector is a modelling choice that would sit inside the scorer where no
# experiment can see it -- a program that is wrong at every tick in the same way
# is perfectly stable. It makes the cost of one evaluation data-dependent and in
# principle unbounded, so a sweep can no longer report the work it will do before
# it does it. And it makes two candidates that settle at different ticks
# incomparable on one budget, which is exactly what a certificate needs them to
# be. A declared horizon is instead an ordinary configuration record of the kind
# ARCHITECTURE section 7 already requires for time alignment, it fixes the cost
# at `ticks * examples * programs`, and it is what the depth-generalization track
# measured against: horizon 12, trailing window 4.
#
# The trailing window subsumes the alternative the task offered rather than
# discarding it. `settle_window=1` is precisely "run to a fixed horizon and read
# the final value". `settle_window=k` additionally requires the scored value to
# be correct at each of the last k ticks, which is a settling criterion --
# "arrived and stayed" -- evaluated at a bounded cost. The depth track's program
# settles at exactly tick d-1, so at horizon 12 and window 4 it is scored well
# after settling for every depth up to 8, and a program that only transiently
# hits the answer fails.
#
# `tick_targets` covers the other supervision shape: an explicit per-tick target
# sequence, scored at each tick that declares one. The two compose; an example
# may carry both.
#
# NOT offered here: prefix reuse. Node values inside tick t>0 depend on state
# committed at tick t-1, which depends on every node, so an early node's value is
# not a function of the prefix of choices and the walk of `enumerate_prefix`
# would be unsound. That is a real limit, not an omission.

def tick_inputs(example, tick):
    """The input record for one tick: constant, or the tick-th of a sequence."""
    given = example['inputs']
    if isinstance(given, (list, tuple)):
        if tick >= len(given): raise ValueError('input sequence shorter than the tick budget')
        return given[tick]
    return given

def scored_ticks(example, ticks, settle_window):
    """Which ticks are scored against which targets, for one example.

    `targets` is the terminal reading and is required at every tick of the
    trailing settle window. `tick_targets` is the per-tick reading and is scored
    at each tick that declares one. An example may carry either or both.
    """
    if settle_window < 1: raise ValueError('settle window must cover at least one tick')
    if settle_window > ticks: raise ValueError('settle window longer than the tick budget')
    schedule = {}
    per = example.get('tick_targets')
    if per is not None:
        if len(per) < ticks: raise ValueError('per-tick targets shorter than the tick budget')
        for t in range(ticks):
            if per[t]: schedule[t] = dict(per[t])
    if example.get('targets'):
        for t in range(ticks - settle_window, ticks):
            schedule[t] = schedule.get(t, {}) | dict(example['targets'])
    if not schedule: raise ValueError('a recurrent example needs targets or tick_targets')
    return schedule

def evaluate_recurrent(program, selections, examples, signals, registry, ticks,
                       settle_window=1, tolerance=None):
    """Largest absolute error of one selection over the scored ticks, or None.

    Unusable is the same notion `evaluate` uses -- an illegal numeric domain or
    an address off the end of a tuple is a property of that candidate. A
    recurrence can also become unusable partway through an episode (an empty
    reduction once a fold has run past its data), which is why the whole tick
    loop rather than one execution sits inside the guard.
    """
    worst = 0.
    for example in examples:
        schedule = scored_ticks(example, ticks, settle_window)
        state = None
        for t in range(ticks):
            try:
                _, state, trace = program.execute(tick_inputs(example, t), state, registry, selections)
            except (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError, IndexError):
                return None
            targets = schedule.get(t)
            if not targets: continue
            for signal in signals:
                if signal.target not in targets: continue
                a = trace[signal.source].flat(); b = targets[signal.target].flat()
                worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
            if tolerance is not None and worst > tolerance:
                return worst
    return worst

def enumerate_recurrent(program, examples, signals, registry=None, ticks=1, settle_window=1,
                        tolerance=.001, max_programs=1 << 20, stop_at_first=False, rank='order'):
    """Search every discrete program in a recurrent scaffold, exactly.

    Identical contract to `enumerate_fit`: same `SearchResult`, same conforming
    count, same exhaustion test, same certificate. The only difference is that a
    candidate is driven for `ticks` ticks through `Program.state` and scored on
    the ticks `scored_ticks` selects, so a program whose answer exists only after
    a fold has run is reachable. `ticks=1` with a terminal target reduces exactly
    to `enumerate_fit` on a stateless program.
    """
    if rank not in {'order','description','cost'}: raise ValueError('unknown ranking')
    if rank != 'order' and stop_at_first: raise ValueError('ranking requires the full conforming set')
    if not examples: raise ValueError('training examples required')
    if ticks < 1: raise ValueError('a recurrent sweep needs at least one tick')
    r = registry or Registry()
    program.validate(r); program.validate_signals(signals)
    counts = candidate_counts(program); total = math.prod(counts)
    names = [n.name for n in program.nodes]
    started = time.perf_counter(); evaluated = 0; found = []; best = float('inf')
    for combination in itertools.product(*(range(c) for c in counts)):
        if evaluated >= max_programs: break
        selections = dict(zip(names, combination)); evaluated += 1
        error = evaluate_recurrent(program, selections, examples, signals, r, ticks,
                                   settle_window, tolerance)
        if error is None: continue
        best = min(best, error)
        if error <= tolerance:
            found.append(selections)
            if stop_at_first: break
    exhausted = evaluated >= total
    complete = not stop_at_first
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
                        (len(found) == 1) if exhausted and complete else None,
                        time.perf_counter() - started, tuple(program.trainable_constants),
                        len(found), rank, bits, cost,
                        certificate_of(exhausted, complete, len(found)), 'recurrent',
                        node_evaluations=evaluated * len(program.nodes) * len(examples) * ticks,
                        ticks=ticks, settle_window=settle_window)


# ---------------------------------------------------------------------------
# mode: live environment return
# ---------------------------------------------------------------------------
#
# SEMANTICS.
#
# A candidate is scored by the return it actually earns: harden the selection,
# drop the trainable-constant flag, hand the frozen program to the shipped
# `tcn.agent.Agent`, roll it in the generator the task names, and sum the
# `reward_components` of the step records. That is the same signal `Agent`
# already drives and the same quantity `JointTrainer` reinforces, so nothing new
# is being optimized -- only the search over it moves inside the backend.
#
# Conformance has two readings and both are supported, because they answer
# different questions. With a `threshold`, a program conforms if its mean return
# reaches it; exhausting the space then certifies how many programs in the whole
# space reach the objective, and `certificate='unique'` means exactly one does.
# Without a threshold, conformance means "attains the best return observed",
# which is only meaningful on an exhausted sweep and is what the depth track's
# bespoke loop computed. `exact_max_error` is the shortfall `max(0, threshold -
# best)` so that the field keeps its meaning: zero exactly when the sweep solved.
#
# `episodes` and `steps` are reported as first-class costs, because FINDINGS
# section 8 established that environment rollouts are the one axis on which the
# relaxed path genuinely wins, so a search that spends them silently is not
# comparable to one that does not.
#
# NOT offered here: prefix reuse or a beam. Return is only defined at a complete
# program -- there is nothing to roll out with half the nodes chosen -- so the
# choice tree has no partial score and the walk of `enumerate_prefix` has nothing
# to reuse. The way to spend fewer episodes is to shrink the candidate set with a
# supervised sweep first and enumerate by return over what survives; that staging
# is what `probe_examples` exists for.

class EpisodeLedger:
    """Environment episodes and steps consumed, counted where they are spent.

    A search that reports only "programs evaluated" is not comparable with one
    whose scarce resource is the environment. Every `Host` this module creates
    goes through a ledger, so an experiment that stages a supervised sweep and an
    environment sweep reports one honest total across both.
    """
    def __init__(self, limit=None):
        if limit is not None and limit < 0: raise ValueError('invalid episode budget')
        self.episodes = 0; self.steps = 0; self.limit = limit
    @property
    def spent(self): return self.limit is not None and self.episodes >= self.limit
    def create(self, generator, **kwargs):
        from .generation import Host
        self.episodes += 1
        return Host.create(generator, **kwargs)
    def charge_steps(self, host):
        self.steps += len(host.records) - 1
        return host
    def to_dict(self): return {'episodes': self.episodes, 'steps': self.steps, 'limit': self.limit}

@dataclass(frozen=True)
class AgentConfig:
    """Exactly the fields `tcn.agent.Agent` reads off a configuration.

    `TrainConfig` also satisfies this shape and can be passed instead; this
    exists so a pure search does not have to declare loss weights and a training
    budget it will never use.
    """
    observations: tuple
    action_templates: tuple
    action_bindings: tuple = ()
    dt: float = 1.
    horizon: int = 8

@dataclass(frozen=True)
class EnvironmentTask:
    """A live-return scoring problem: a generator, a config, and an action schema.

    `indices` are the episode addresses the sweep is scored on; `objectives`
    cycle over them positionally, as every other harness in this repository does.
    `scored_ticks` is a trailing reward window, the counterpart of the recurrent
    mode's settle window: a recurrence needs ticks to settle before its answer
    exists, and charging it for that settling measures the fold rather than the
    program. `None` scores the whole episode.
    """
    generator: str
    observations: tuple
    action_templates: tuple
    generator_config: dict = field(default_factory=dict)
    objectives: tuple = ({},)
    indices: tuple = (0,)
    horizon: int = 4
    dt: float = 1.
    seed: int = 0
    split: str = 'train'
    scored_ticks: int | None = None
    action_bindings: tuple = ()
    deterministic: bool = True
    agent_seed: int = 0
    actor: str = 'agent_0'
    def __post_init__(self):
        if not self.action_templates: raise ValueError('an action schema is required')
        if not self.objectives: raise ValueError('at least one objective is required')
        if not self.indices: raise ValueError('at least one episode address is required')
        if self.horizon < 1 or self.dt <= 0: raise ValueError('invalid rollout budget')
        if self.scored_ticks is not None and not 1 <= self.scored_ticks <= self.horizon:
            raise ValueError('scored window must lie inside the horizon')
    def agent_config(self):
        return AgentConfig(tuple(self.observations), tuple(self.action_templates),
                           tuple(self.action_bindings), self.dt, self.horizon)
    def objective(self, position): return dict(self.objectives[position % len(self.objectives)])
    def to_dict(self):
        return {'generator': self.generator, 'observations': list(self.observations),
                'actions': [a.to_dict() for a in self.action_templates],
                'generator_config': dict(self.generator_config),
                'objectives': [dict(o) for o in self.objectives], 'indices': list(self.indices),
                'horizon': self.horizon, 'dt': self.dt, 'seed': self.seed, 'split': self.split,
                'scored_ticks': self.scored_ticks, 'deterministic': self.deterministic}

def trailing_return(host, scored_ticks=None):
    """Summed reward components over the trailing window of an episode."""
    rewards = [sum(v.decoded for v in record.reward_components.values()) for record in host.records[1:]]
    return float(sum(rewards if scored_ticks is None else rewards[-scored_ticks:]))

def frozen_selection(program, selections, registry=None):
    """The exact callable one discrete selection names.

    Hardening fixes the choices; dropping `trainable_constants` states that the
    continuous part is being held at its declared values rather than searched.
    `SearchResult.continuous` reports which constants those were, so the
    restriction is never silent.
    """
    return replace(program.harden(selections), trainable_constants=()).validate(registry)

def episode_return(program, registry, task, index, position, ledger, split=None):
    """One rollout of a frozen program, charged to the ledger."""
    from .agent import Agent
    host = ledger.create(task.generator, seed=task.seed, index=index, split=split or task.split,
                         configuration=dict(task.generator_config) | {'horizon': task.horizon},
                         objective=task.objective(position))
    Agent(program, registry, task.agent_config(), seed=task.agent_seed).rollout(
        host, deterministic=task.deterministic, actor=task.actor)
    ledger.charge_steps(host)
    return trailing_return(host, task.scored_ticks), host

def program_return(program, registry, task, selections, ledger, split=None):
    """Mean return of one discrete selection over the task's episodes, or None.

    `None` is the same "unusable candidate" verdict `evaluate` returns: a program
    that cannot run in the environment is a property of that candidate. The
    episodes it consumed before failing are still charged, because they were.
    """
    try:
        exact = frozen_selection(program, selections, registry)
    except (ValueError, TypeError): return None
    total = 0.
    for position, index in enumerate(task.indices):
        try:
            score, _ = episode_return(exact, registry, task, index, position, ledger, split)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError, IndexError):
            return None
        total += score
    return total / len(task.indices)

def probe_examples(task, targets, ledger, indices=None, split=None, extra_inputs=None):
    """Supervised examples harvested from real episodes, charged to the ledger.

    One episode yields its tick-0 record, which `Host.create` already produces:
    the observations the actor is permitted and the probes the curriculum
    declares, kept on the separate channels ARCHITECTURE section 7 requires.
    `targets` is a sequence of `(name, channel, key)`. This exists so a staged
    search -- fit a world model by supervision, then settle the rest by reward --
    can report one environment-episode total across both stages.
    """
    out = []
    for position, index in enumerate(indices if indices is not None else task.indices):
        host = ledger.create(task.generator, seed=task.seed, index=index, split=split or task.split,
                             configuration=dict(task.generator_config) | {'horizon': task.horizon},
                             objective=task.objective(position))
        view = host.view(task.actor); record = host.records[0]
        inputs = {k: view.observations[k] for k in task.observations}
        inputs.update(extra_inputs or {})
        out.append({'inputs': inputs,
                    'targets': {name: getattr(record, channel)[key] for name, channel, key in targets}})
    return out

def enumerate_environment(program, task, registry=None, threshold=None, max_programs=1 << 20,
                          max_episodes=None, rank='order', stop_at_first=False, ledger=None):
    """Score every discrete program by the return it actually earns.

    Same `SearchResult` contract as `enumerate_fit`. `threshold` makes
    conformance an absolute test and so lets an exhausted sweep certify how many
    programs in the space reach the objective; without it conformance means
    "attains the best return in the enumerated set", which only means anything on
    an exhausted sweep and is reported as such.

    `max_episodes` bounds the environment resource rather than the program count.
    Hitting it stops the sweep and forfeits the certificate, exactly as
    `max_programs` does.
    """
    if rank not in {'order','description','cost'}: raise ValueError('unknown ranking')
    if rank != 'order' and stop_at_first: raise ValueError('ranking requires the full conforming set')
    if stop_at_first and threshold is None: raise ValueError('an early exit needs an acceptance threshold')
    r = registry or Registry()
    program.validate(r)
    counts = candidate_counts(program); total = math.prod(counts)
    names = [n.name for n in program.nodes]
    book = ledger if ledger is not None else EpisodeLedger(max_episodes)
    if max_episodes is not None and book.limit is None: book.limit = max_episodes
    started = time.perf_counter(); evaluated = 0; scores = []; best = None
    for combination in itertools.product(*(range(c) for c in counts)):
        if evaluated >= max_programs or book.spent: break
        selections = dict(zip(names, combination)); evaluated += 1
        score = program_return(program, r, task, selections, book)
        if score is None: continue
        best = score if best is None else max(best, score)
        scores.append((selections, score))
        if threshold is not None and stop_at_first and score >= threshold: break
    if threshold is not None:
        found = [s for s, v in scores if v >= threshold]
    else:
        found = [s for s, v in scores if best is not None and v >= best]
    exhausted = evaluated >= total and not book.spent
    complete = not stop_at_first
    chosen = found[0] if found else None
    bits = cost = None
    if found:
        if rank == 'order':
            bits, cost = program_cost(program, chosen, r)
        else:
            ranked = [(s,) + program_cost(program, s, r) for s in found]
            key = (lambda x: (x[1], x[2])) if rank == 'description' else (lambda x: (x[2], x[1]))
            chosen, bits, cost = min(ranked, key=key)
    shortfall = 0. if threshold is None or best is None else max(0., threshold - best)
    return SearchResult(bool(found), chosen, shortfall, evaluated, total, exhausted,
                        (len(found) == 1) if exhausted and complete else None,
                        time.perf_counter() - started, tuple(program.trainable_constants),
                        len(found), rank, bits, cost,
                        certificate_of(exhausted, complete, len(found)), 'environment',
                        episodes=book.episodes, steps=book.steps,
                        ticks=task.horizon, settle_window=task.scored_ticks or task.horizon,
                        best_return=best, threshold=threshold)


# ---------------------------------------------------------------------------
# mode: prefix reuse, and the bounded beam over it
# ---------------------------------------------------------------------------
#
# SEMANTICS.
#
# `enumerate_fit` re-executes the whole program once per discrete selection. The
# choice tree makes that redundant: a `Program` is validated to have no forward
# reference, so node i's value depends only on the choices at nodes before it.
# Walking the tree depth-first and computing each node once per *prefix* rather
# than once per *leaf* therefore returns the identical conforming set. The
# object-identity track measured 393,216 programs at 196 s against a projected
# 1,217 s for the flat sweep, and `research/discrete-perception/incremental.py`
# measured 17.8x at R=8 with the sets checked equal.
#
# Rejecting a subtree is sound for the same reason. Conformance requires every
# probed node to match within tolerance; a probed node whose value already misses
# cannot be repaired by a later choice, so the whole subtree is *decided*, not
# skipped. `evaluated` counts decided programs, so `evaluated == space_size`
# remains the exhaustion test and the certificate survives intact:
# `enumerate_prefix(beam=None)` is exhaustive and certifies uniqueness.
#
# `beam=k` is the part that gives something up, and it says so. After each node
# the surviving prefixes are ordered by the worst probe error decided so far and
# only the best k are kept; the rest are discarded **without being decided**.
# `discarded` counts them, `exhausted` is False, `unique` is None and
# `certificate` is `'none'`. Nothing about the unexamined part of the space is
# claimed. The uniqueness certificate is this backend's main advantage over a
# gradient run, and the beam is the mode that trades it away, so the trade is
# recorded in the result rather than left to a reader of the call site.
#
# One honest limit on how much the beam can buy: the ordering it prunes by is the
# probe error of nodes already decided. Where a scaffold has dense intermediate
# supervision -- the staging FINDINGS section 14 found is what makes rung 3.5
# reachable at all -- that ordering is informative and the beam keeps the right
# prefixes. Where the only probe is at the output, no prefix has a score until
# the last node, every prefix ties, and the beam degenerates to "keep the first k
# in enumeration order", which is a budget, not a search. The beam is a
# supervision-shaped tool.

def signal_index(signals):
    """Probe signals grouped by the port they read."""
    by = {}
    for s in signals: by.setdefault(s.source, []).append(s)
    return by

def enumerate_prefix(program, examples, signals, registry=None, tolerance=.001, beam=None,
                     max_programs=1 << 20, rank='order', stop_at_first=False):
    """The conforming set of `enumerate_fit`, walked as a tree over the prefix.

    With `beam=None` this is exhaustive and the certificate is preserved. With
    `beam=k` the walk is bounded, `discarded` counts the programs dropped without
    being decided, and the certificate is forfeited.
    """
    if rank not in {'order','description','cost'}: raise ValueError('unknown ranking')
    if rank != 'order' and stop_at_first: raise ValueError('ranking requires the full conforming set')
    if not examples: raise ValueError('training examples required')
    if beam is not None and beam < 1: raise ValueError('beam width must be positive')
    if beam is not None and stop_at_first: raise ValueError('a beam already forfeits the certificate; do not also exit early')
    if program.state:
        raise ValueError('prefix reuse is unsound across a recurrence: a node inside tick t>0 '
                         'depends on state committed by every node at tick t-1, so its value is '
                         'not a function of the choice prefix. Use enumerate_recurrent.')
    r = registry or Registry()
    program.validate(r); program.validate_signals(signals)
    nodes = list(program.nodes); counts = candidate_counts(program); total = math.prod(counts)
    below = [1] * (len(nodes) + 1)
    for i in range(len(nodes) - 1, -1, -1): below[i] = below[i + 1] * counts[i]
    below = below[1:] + [1]                      # below[i] = programs under one choice at node i
    by_source = signal_index(signals)
    started = time.perf_counter()
    bases = [dict(ex['inputs']) | dict(program.constants) for ex in examples]
    targets = [{s.target: ex['targets'][s.target].flat() for s in signals} for ex in examples]
    n = len(examples)

    def probe_error(name, e, value):
        worst = 0.; a = value.flat()
        for s in by_source[name]:
            b = targets[e][s.target]
            worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
        return worst

    # A probe may read an input, a constant or a state port rather than a node.
    # Such a value is the same for every discrete program, so it is checked once:
    # if it misses, the space is exhaustively empty and nothing needs walking.
    node_names = {node.name for node in nodes}
    for name in by_source:
        if name in node_names: continue
        for e in range(n):
            if probe_error(name, e, bases[e][name]) > tolerance:
                return SearchResult(False, None, 0., total, total, True, False,
                                    time.perf_counter() - started,
                                    tuple(program.trainable_constants), 0, rank, None, None,
                                    certificate_of(True, True, 0),
                                    'prefix' if beam is None else 'beam', beam=beam)

    unusable = (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError, IndexError)
    found = []; decided = 0; discarded = 0; node_evaluations = 0; best = float('inf')

    def extend(values, node, candidate, worst, probed):
        """Node values for one prefix extension, or None if it cannot conform."""
        nonlocal node_evaluations
        out = [None] * n
        try:
            for e in range(n):
                v = r.exact(candidate.operator, [values[e][s] for s in candidate.sources])
                node_evaluations += 1; out[e] = v
                if probed:
                    w = probe_error(node.name, e, v)
                    if w > worst: worst = w
                    if worst > tolerance: return None
        except unusable:
            return None
        return out, worst

    if beam is None:
        values = [dict(b) for b in bases]; selections = {}; stopped = False

        def walk(i, worst):
            nonlocal decided, best, stopped
            if stopped: return
            if i == len(nodes):
                decided += 1; found.append(dict(selections)); best = min(best, worst)
                if stop_at_first: stopped = True
                return
            node = nodes[i]; probed = node.name in by_source
            for k, candidate in enumerate(node.candidates):
                if stopped: return
                if decided >= max_programs: stopped = True; return
                step = extend(values, node, candidate, worst, probed)
                if step is None:
                    decided += below[i]; continue
                out, w = step
                for e in range(n): values[e][node.name] = out[e]
                selections[node.name] = k
                walk(i + 1, w)
            selections.pop(node.name, None)

        walk(0, 0.)
    else:
        # Level-synchronous, so "width" means what it says: at most `beam` live
        # prefixes after each node. Values are materialized only for the prefixes
        # that survive the cut.
        live = [({}, [dict(b) for b in bases], 0.)]
        for i, node in enumerate(nodes):
            probed = node.name in by_source; children = []
            for parent, (selections, values, worst) in enumerate(live):
                for k, candidate in enumerate(node.candidates):
                    if decided >= max_programs: break
                    step = extend(values, node, candidate, worst, probed)
                    if step is None:
                        decided += below[i]; continue
                    out, w = step
                    children.append((w, len(children), parent, k, out))
            children.sort(key=lambda x: (x[0], x[1]))     # error first, enumeration order on ties
            if len(children) > beam:
                for _ in children[beam:]: discarded += below[i]
                children = children[:beam]
            live = [(dict(live[p][0]) | {node.name: k},
                     [dict(live[p][1][e]) | {node.name: out[e]} for e in range(n)], w)
                    for w, _, p, k, out in children]
            if not live: break
        for selections, _, worst in live:
            decided += 1; found.append(dict(selections)); best = min(best, worst)

    exhausted = decided >= total and discarded == 0
    complete = not stop_at_first and discarded == 0
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
                        0. if not found else min(best, tolerance), decided, total, exhausted,
                        (len(found) == 1) if exhausted and complete else None,
                        time.perf_counter() - started, tuple(program.trainable_constants),
                        len(found), rank, bits, cost,
                        certificate_of(exhausted, complete, len(found)),
                        'prefix' if beam is None else 'beam',
                        node_evaluations=node_evaluations, beam=beam, discarded=discarded)


# ---------------------------------------------------------------------------
# routing: what a selector needs from this backend
# ---------------------------------------------------------------------------
#
# `tcn/select.py` is not on main at the time of writing, so the wiring is exposed
# here rather than performed. A selector needs two things from a discrete
# backend: whether it can score a given problem at all, and what it will cost.
# `DiscreteProblem` is the declarative shape, `route` answers the first question
# and `solve` performs it, so `mode="auto"` can send a recurrent or
# environment-coupled problem here instead of falling through to relaxation.

MODES = {
    'fit':         'feed-forward, probe targets, exact; certificate available',
    'recurrent':   'Program.state driven for a declared tick budget and settle window; certificate available',
    'environment': 'live rollouts scored by summed reward components; reports environment episodes',
    'prefix':      'the fit conforming set, walked with prefix reuse; certificate available',
    'beam':        'prefix walk bounded to a width; certificate forfeited, discarded reported',
}

@dataclass(frozen=True)
class DiscreteProblem:
    """One synthesis problem, in the shape the discrete backend can route on."""
    program: object
    examples: tuple = ()
    signals: tuple = ()
    task: EnvironmentTask | None = None
    ticks: int = 1
    settle_window: int = 1
    beam: int | None = None
    prefix: bool = False
    tolerance: float = .001
    registry: object = None

def route(problem):
    """Which mode scores this problem, or None if the backend cannot.

    Environment coupling wins over everything, because a live return is the only
    signal that problem declares. Recurrence is next: a program with state cannot
    be scored feed-forward at all, and prefix reuse is unsound across it. A beam
    is a budget decision and applies only to the feed-forward shape. The flat
    exhaustive sweep stays the default: `prefix=True` opts into the faster
    exhaustive walk, which certifies the same thing, and `beam` into the bounded
    one, which does not.
    """
    if problem.task is not None: return 'environment'
    if not problem.examples or not problem.signals: return None
    if getattr(problem.program, 'state', ()) or problem.ticks > 1: return 'recurrent'
    if problem.beam is not None: return 'beam'
    return 'prefix' if problem.prefix else 'fit'

def solve(problem, **kwargs):
    """Route and run. Every mode returns the same `SearchResult`."""
    mode = route(problem)
    if mode is None: raise ValueError('the discrete backend cannot score this problem')
    if mode == 'environment':
        return enumerate_environment(problem.program, problem.task, problem.registry, **kwargs)
    if mode == 'recurrent':
        return enumerate_recurrent(problem.program, problem.examples, problem.signals,
                                   problem.registry, problem.ticks, problem.settle_window,
                                   problem.tolerance, **kwargs)
    if mode in {'beam', 'prefix'}:
        return enumerate_prefix(problem.program, problem.examples, problem.signals,
                                problem.registry, problem.tolerance, problem.beam, **kwargs)
    return enumerate_fit(problem.program, problem.examples, problem.signals,
                         problem.registry, problem.tolerance, **kwargs)

def viability(program, examples=1, ticks=1, episodes=0, rate=2e4):
    """Where a mode stops being viable, in the only terms that decide it.

    `rate` is executions per second, measured rather than assumed by the caller.
    Environment episodes dominate whenever they are charged at all, so they are
    reported separately: a sweep that spends one episode per program is bounded
    by the environment, not by the CPU.
    """
    size = space_size(program)
    executions = size * examples * max(1, ticks)
    return {'space_size': size, 'executions': executions, 'projected_seconds': executions / rate,
            'environment_episodes': size * episodes,
            'exhaustible': executions / rate < 3600.}
