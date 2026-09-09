"""A depth-independent circuit interpreter built from existing operators only.

Evaluating a depth-d circuit is inherently sequential: gate i's inputs may be
gate i-1's output. `map` is parallel over a set and carries no fold, so a single
map cannot do it. `ARCHITECTURE.md` section 4 supplies the missing mechanism
directly -- "within a tick it is a DAG; recurrence reads old state and commits
new state through explicit delays" -- so the fold is one gate per tick over
`Program.state`.

The program text below is a fixed graph. It contains no per-gate node, no
per-depth constant and no unrolled block: depth appears only as the number of
ticks the recurrence runs and as the cardinality of the `gates` input set. That
is the property `program`'s `3 * depth` tuple could not have.

    state values  : set[(wire, bool)]   the wires computed so far
    state counter : int                 which gate this tick evaluates

    held     = insert({}, counter)                       set[int], capacity 1
    records  = pair(gates, held)                         set[(gate, counter)]
    current  = filter(records; index == counter)         the one gate of this tick
    wire_a   = sum(map(current; gate.a))                 0 when the circuit is finished
    va       = mux(wire_a < w, index(bits, wire_a % w), member(values, (wire_a, true)))
    ... likewise wire_b, vb, table
    relation = index(<truth_0..truth_15(va,vb)>, table)  <- the searched choice
    values'  = insert(values, (w + counter, relation))
    counter' = counter + 1
    answer   = member(values, (w + count(gates) - 1, true))

Three details are load-bearing and each was forced by a real constraint:

* `sum` extracts the singleton, not `reduce_max`: `Registry.exact` raises
  "empty reduction" for min/max/mean on an empty set but `sum` of the empty set
  is 0, and the filtered set *is* empty once the counter passes the last gate.
  That makes the recurrence total rather than a crash after settling.
* the wire lookup is a `mux`, because a wire below `w` is an input bit (read
  with `index` over `bits`) and a wire at or above `w` is a gate output (read
  with `member` over the state relation). Both branches are always evaluated, so
  the `index` address is taken modulo `w` to keep it in range unconditionally.
* the value relation stores `(wire, bool)` rather than a bare wire set, so
  "false" and "not yet computed" are distinguishable by construction; the
  membership test asks for `(wire, true)`.

Everything here resolves through `tcn.operators.Registry`. No operator, type,
relaxation or loss term is added.
"""
from __future__ import annotations
import math
import random

import torch

from tcn.types import BOOL, Value, product, integer, floating, setof
from tcn.generation import Host, Action
from tcn.graph import Program, Node, Candidate
from tcn.operators import Registry
from tcn.learning import SoftProgram
from tcn.training import TrainConfig, Target, JointTrainer

F = floating()
WIRE = integer(8, signed=False)         # the gate-relation field encoding
COUNT = integer(32, signed=False)       # what `count` over a set returns
OBJECTIVES = ({'invert': False}, {'invert': True})

CONSTANTS = (('w0', Value.of(F, -2.)), ('w1', Value.of(F, 2.)),
             ('bias0', Value.of(F, 1.)), ('bias1', Value.of(F, -1.)),
             ('baseline', Value.of(F, 1.)))


def _policy_tail(r, nodes, base_depth, relation_source, goal_source='goal_relation'):
    """The z/world/prediction/policy/value tail of examples/joint.py, verbatim."""
    d = base_depth
    for name, source in [('z', goal_source), ('world', relation_source)]:
        nodes.append(Node(name, F, (Candidate(r.resolve('encode', (BOOL,), F), (source,)),), 'encoding', d))
    for i in range(2):
        nodes.append(Node(f'mul{i}', F, (Candidate(r.resolve('mul', (F, F)), ('z', f'w{i}')),), 'policy', d + 1))
        nodes.append(Node(f'logit{i}', F, (Candidate(r.resolve('add', (F, F)), (f'mul{i}', f'bias{i}')),), 'policy', d + 2))
    nodes.append(Node('policy', product(F, F), (Candidate(r.resolve('tuple', (F, F)), ('logit0', 'logit1')),), 'policy', d + 3))
    nodes.append(Node('prediction', product(F, F), (Candidate(r.resolve('tuple', (F, F)), ('z', 'world')),), 'prediction', d + 1))
    nodes.append(Node('value', product(F), (Candidate(r.resolve('tuple', (F,)), ('baseline',)),), 'value', 1))


# ---------------------------------------------------------------------------
# the four tiny modules the set operators need
# ---------------------------------------------------------------------------
def _module(r, name, element, body):
    """Freeze a one-node-per-step sub-program and register it as an operator.

    `map` and `filter` take a module, and a module cannot close over an outer
    port, so whatever the body needs must arrive inside the set element. That is
    why `pair` attaches the counter to every gate record.
    """
    nodes = []
    types = {'x': element}
    for step, (label, op, sources) in enumerate(body):
        operator = r.resolve(op[0], tuple(types[s] for s in sources), None, op[1] if len(op) > 1 else None)
        nodes.append(Node(label, operator.output, (Candidate(operator, tuple(sources)),), 'core', step + 1, 0))
        types[label] = operator.output
    p = Program((('x', element),), tuple(nodes), ((name, nodes[-1].name),)).validate(r)
    return r.register_module(p)


def gate_modules(registry, element):
    """`is_current`, and one field reader for each of wire_a, wire_b, table."""
    names = {}
    names['match'] = _module(registry, 'flag', element, [
        ('gate', ('project', {'index': 0}), ('x',)),
        ('idx', ('project', {'index': 0}), ('gate',)),
        ('counter', ('project', {'index': 1}), ('x',)),
        ('flag', ('eq',), ('idx', 'counter')),
    ])
    for label, field in (('a', 1), ('b', 2), ('table', 3)):
        names[label] = _module(registry, 'field', element, [
            ('gate', ('project', {'index': 0}), ('x',)),
            ('field', ('project', {'index': field}), ('gate',)),
        ])
    return names


# ---------------------------------------------------------------------------
# the scaffold
# ---------------------------------------------------------------------------
def interpreter_scaffold(host, width, capacity, horizon, wire_choice=False,
                         interpreter_only=False, settle_mux=True, registry=None):
    """One fixed graph that evaluates a circuit of any depth up to `capacity`.

    `wire_choice=True` also puts the wire lookup up for search: each of the two
    gate operands gets a second candidate that reads a fixed bit of `bits`, the
    shortcut that suffices at depth 1 with fixed wiring and fails beyond it.
    """
    r = registry or Registry()
    names = ('bits', 'goal', 'gates')
    view = host.view().observations
    gates_t = view['gates'].type
    bits_t = view['bits'].type
    if gates_t.capacity != capacity or len(bits_t.items) != width:
        raise ValueError('scaffold and episode disagree on width or capacity')
    element = product(gates_t.items[0], WIRE)                 # (gate record, counter)
    modules = gate_modules(r, element)

    # The wire relation holds one entry per tick, and the recurrence keeps
    # ticking after the circuit is finished, so capacity is charged for the whole
    # horizon rather than for the circuit.
    entry = product(WIRE, BOOL)
    values_t = setof(entry, width + capacity + horizon)
    holder = setof(WIRE, 1)

    inputs = tuple((k, view[k].type) for k in names) + (('action', product(F, F)), ('dt', F))
    constants = CONSTANTS + (
        ('empty', Value.of(holder, ())),
        ('one', Value.of(WIRE, 1)),
        ('width', Value.of(WIRE, width)),
        ('width_minus_one', Value.of(WIRE, width - 1)),
        ('true', Value.of(BOOL, True)),
    )
    state = (('values', Value.of(values_t, ()), 'values_next'),
             ('counter', Value.of(WIRE, 0), 'counter_next'))

    nodes = []
    port = dict(inputs) | {k: v.type for k, v in constants} | {k: v.type for k, v, _ in state}

    def node(name, op, sources, region, depth, parameters=None, output=None, alternatives=()):
        """One node; `alternatives` are (operator, sources, parameters) triples."""
        specs = [(op, tuple(sources), parameters)] + [(a, tuple(s), p) for a, s, p in alternatives]
        candidates = []
        for a, s, p in specs:
            candidates.append(Candidate(r.resolve(a, tuple(port[k] for k in s), output, p), s))
        # `selected` stays None even for a one-candidate node. `SoftProgram`
        # treats every node with a selection as frozen and routes it through
        # `exact_tensor(...).detach()`, which severs the graph: pinning the
        # single-candidate plumbing would leave the `relation` choice with no
        # gradient path at all. `examples/joint.py` leaves them None for the same
        # reason. Exact execution supplies 0 for them explicitly.
        n = Node(name, candidates[0].operator.output, tuple(candidates), region, depth, None)
        nodes.append(n)
        port[name] = n.output
        return name

    # -- select this tick's gate ------------------------------------------------
    node('held', 'insert', ('empty', 'counter'), 'dispatch', 1)
    node('records', 'pair', ('gates', 'held'), 'dispatch', 2)
    node('current', 'filter', ('records',), 'dispatch', 3, {'module': modules['match']})
    for label in ('a', 'b', 'table'):
        node(f'{label}_set', 'map', ('current',), 'dispatch', 4, {'module': modules[label]})
        node(f'{label}_value', 'sum', (f'{label}_set',), 'dispatch', 5)

    # -- read the two operand wires --------------------------------------------
    for i, label in enumerate(('a', 'b')):
        node(f'{label}_is_input', 'lt', (f'{label}_value', 'width'), 'read', 6)
        node(f'{label}_address', 'mod', (f'{label}_value', 'width'), 'read', 6)
        node(f'{label}_from_bits', 'index', ('bits', f'{label}_address'), 'read', 7)
        node(f'{label}_key', 'tuple', (f'{label}_value', 'true'), 'read', 6)
        node(f'{label}_from_gates', 'member', ('values', f'{label}_key'), 'read', 7)
        shortcut = (('project', ('bits',), {'index': i}),) if wire_choice else ()
        node(f'v{label}', 'mux', (f'{label}_is_input', f'{label}_from_bits', f'{label}_from_gates'),
             'read', 8, alternatives=shortcut)

    # -- evaluate the gate ------------------------------------------------------
    for j in range(16):
        node(f'gate_{j}', f'truth_{j}', ('va', 'vb'), 'evaluate', 9)
    node('gate_table', 'tuple', tuple(f'gate_{j}' for j in range(16)), 'evaluate', 10)
    lookup = ('index', ('gate_table', 'table_value'), None)
    fixed = [(f'truth_{j}', ('va', 'vb'), None) for j in range(1, 16)]
    if interpreter_only:
        node('relation', 'index', ('gate_table', 'table_value'), 'evaluate', 11)
    else:
        node('relation', 'truth_0', ('va', 'vb'), 'evaluate', 11,
             alternatives=tuple(fixed) + (lookup,))

    # -- commit the new wire ----------------------------------------------------
    node('new_wire', 'add', ('counter', 'width'), 'commit', 12)
    node('new_entry', 'tuple', ('new_wire', 'relation'), 'commit', 13)
    node('values_next', 'insert', ('values', 'new_entry'), 'commit', 14)
    node('counter_next', 'add', ('counter', 'one'), 'commit', 12)

    # -- read the answer wire out of the relation -------------------------------
    node('gate_count', 'count', ('gates',), 'answer', 1)
    node('depth_wire', 'encode', ('gate_count',), 'answer', 2, output=WIRE)
    node('answer_wire', 'add', ('depth_wire', 'width_minus_one'), 'answer', 3)
    node('answer_key', 'tuple', ('answer_wire', 'true'), 'answer', 4)
    node('answer_bit', 'member', ('values', 'answer_key'), 'answer', 5)

    # `insert`, `member`, `pair`, `map`, `filter` and set `sum` all declare
    # gradient="none", so every value carried through the recurrence reaches the
    # next tick detached: nothing downstream of `values_next` can deliver a
    # gradient to the `relation` choice logits. That is declared behaviour, not a
    # bug, and it is the same wall `research/positional-reuse` hit.
    #
    # The composition that reopens the path costs two nodes. On the tick that
    # computes the last gate -- counter + w == w + d - 1 -- `relation` *is* the
    # answer, so mux it in directly instead of waiting to read it back out of the
    # state relation next tick. `mux` relaxes to a*x + (1-a)*y, so the choice
    # logits at `relation` are differentiable through it. It also makes the
    # answer available one tick earlier.
    if settle_mux:
        node('is_final', 'eq', ('new_wire', 'answer_wire'), 'answer', 13)
        node('answer_now', 'mux', ('is_final', 'relation', 'answer_bit'), 'answer', 14)
        read = 'answer_now'
    else:
        # The ablation: read the answer only out of the state relation, which is
        # the construction the recurrence suggests on its own. It is equally
        # exact -- one tick slower to settle -- and gradient-dead.
        read = 'answer_bit'
    node('goal_relation', 'truth_0', (read, 'goal'), 'answer', 15,
         alternatives=tuple((f'truth_{j}', (read, 'goal'), None) for j in range(1, 16)))

    _policy_tail(r, nodes, 16, read)
    program = Program(inputs, tuple(nodes),
                      (('policy', 'policy'), ('prediction', 'prediction'),
                       ('probe', 'prediction'), ('value', 'value')),
                      constants, state,
                      trainable_constants=tuple(k for k, _ in CONSTANTS)).validate(r)
    return names, program, r


def record_scaffold(host):
    """The recorded joint scaffold: `bits` and `goal` only, one global gate choice."""
    r = Registry()
    names = ('bits', 'goal')
    view = host.view().observations
    inputs = tuple((k, view[k].type) for k in names) + (('action', product(F, F)), ('dt', F))
    nodes = []
    for i in range(2):
        op = r.resolve('project', (dict(inputs)['bits'],), parameters={'index': i})
        nodes.append(Node(f'bit_{i}', BOOL, (Candidate(op, ('bits',)),), 'observation', 1))
    for name, sources, depth in [('relation', ('bit_0', 'bit_1'), 2), ('goal_relation', ('relation', 'goal'), 3)]:
        nodes.append(Node(name, BOOL, tuple(Candidate(r.resolve(f'truth_{i}', (BOOL, BOOL)), sources)
                                            for i in range(16)), 'latent', depth))
    _policy_tail(r, nodes, 4, 'relation')
    program = Program(inputs, tuple(nodes),
                      (('policy', 'policy'), ('prediction', 'prediction'),
                       ('probe', 'prediction'), ('value', 'value')),
                      CONSTANTS, trainable_constants=tuple(k for k, _ in CONSTANTS)).validate(r)
    return names, program, r


# ---------------------------------------------------------------------------
# training harness
# ---------------------------------------------------------------------------
ACTIONS = (Action('answer', arguments=(('value', Value.of(BOOL, False)),)),
           Action('answer', arguments=(('value', Value.of(BOOL, True)),)))


def make_config(names, settings, episodes, seed, horizon, lr=.04):
    return TrainConfig('logic', names,
                       (Target('probes', 'target', 1), Target('probes', 'gate', 1)),
                       ACTIONS, dict(settings), objectives=OBJECTIVES,
                       episodes=episodes, horizon=horizon, lr=lr, probe_weight=1., seed=seed)


class ScheduledTrainer(JointTrainer):
    """`TrainConfig.generator_config` is one dict per run; depth must vary per episode."""

    def __init__(self, model, config, schedule, scored_ticks=None):
        super().__init__(model, config)
        self.schedule = schedule
        self.scored_ticks = scored_ticks

    def episode(self, index, train=True, split='train', loss_only=False, regularized=True):
        self.config.generator_config = dict(self.schedule(index, split))
        result = super().episode(index, train, split, loss_only, regularized)
        if loss_only:
            return result
        metrics, host = result
        if self.scored_ticks is not None:
            metrics['settled_return'] = settled_return(host, self.scored_ticks)
        return metrics, host


def settled_return(host, scored_ticks):
    """Reward over the trailing `scored_ticks` ticks.

    The recurrence needs one tick per gate before the answer wire exists, so a
    return summed from tick 0 charges the interpreter for the settling it is
    doing. Scoring the trailing window measures the answer once the fold has
    run, on the same 0..4 scale every other logic result in this repository
    uses. Every baseline below is measured on exactly the same window.
    """
    rewards = [sum(v.decoded for v in rec.reward_components.values()) for rec in host.records[1:]]
    return float(sum(rewards[-scored_ticks:]))


def depth_schedule(depths, base):
    def f(index, split):
        return dict(base) | {'depth': depths[(index // 2) % len(depths)]}
    return f


def fixed_depth(depth, base):
    def f(index, split):
        return dict(base) | {'depth': depth}
    return f


# ---------------------------------------------------------------------------
# baselines, on the identical episodes and the identical scored window
# ---------------------------------------------------------------------------
def _rollout(index, split, schedule, answer, seed, horizon, scored_ticks):
    goal = OBJECTIVES[index % len(OBJECTIVES)]
    host = Host.create('logic', seed=seed, index=index, split=split,
                       configuration=dict(schedule(index, split)) | {'horizon': horizon}, objective=goal)
    for _ in range(horizon):
        if host.records[-1].done:
            break
        host.step((Action('answer', arguments=(('value', Value.of(BOOL, answer()
                                                                  if callable(answer) else answer)),)),), 1.)
    return settled_return(host, scored_ticks), bool(host.records[0].probes['target'].decoded)


def baselines(indices, split, schedule, seed, horizon, scored_ticks, rng_seed=0):
    rng = random.Random(rng_seed)
    truth = []
    out = {}
    for label, answer in (('always_true', True), ('always_false', False),
                          ('uniform_random', lambda: rng.random() < .5)):
        returns = []
        for i in indices:
            value, target = _rollout(i, split, schedule, answer, seed, horizon, scored_ticks)
            returns.append(value)
            if label == 'always_true':
                truth.append(target)
        out[label] = summarize(returns)
    out['fraction_target_true'] = sum(truth) / len(truth)
    out['best_constant'] = max(out['always_true']['mean'], out['always_false']['mean'])
    return out


def summarize(xs):
    n = len(xs)
    m = sum(xs) / n if n else 0.
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, n - 1)) if n > 1 else 0.
    return {'mean': m, 'sd': sd, 'n': n, 'min': min(xs) if xs else None, 'max': max(xs) if xs else None}


def perturb(model, scale, seed):
    """`SoftProgram` zero-initializes every choice logit, so `torch.manual_seed`
    does not vary synthesis at all (FINDINGS section 11, fault P2). Explicit
    initialization noise is the only thing that makes per-seed statistics mean
    anything, and every number reported from this directory that carries a
    standard deviation was produced with it."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.choices:
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * scale)


def deterministic_settled(trainer, indices, split, schedule, scored_ticks):
    """Deterministic (argmax) settled return of the soft model on given episodes."""
    saved_schedule, saved_config = trainer.schedule, trainer.config.generator_config
    trainer.schedule = schedule
    try:
        return [trainer.episode(i, train=False, split=split)[0]['settled_return'] for i in indices]
    finally:
        trainer.schedule, trainer.config.generator_config = saved_schedule, saved_config
