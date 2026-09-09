"""Track 4: structural generalization of the joint logic agent.

Nothing here modifies `tcn/` or `generators/`. Scaffold variants are built from
the public graph/operator API exactly the way `examples/joint.py` builds the one
scaffold in the validation record.
"""
from __future__ import annotations
import copy
import math
import random

import torch

from tcn.types import BOOL, Value, product, integer, floating
from tcn.generation import Host, Action
from tcn.graph import Program, Node, Candidate
from tcn.operators import Registry
from tcn.learning import SoftProgram
from tcn.training import TrainConfig, Target, JointTrainer

F = floating()
TABLE_INT = integer(8, signed=False)

# f(a,b) = c0 xor c1*a xor c2*b, with index = 2*a+b -> these eight tables.
AFFINE = (0, 3, 5, 6, 9, 10, 12, 15)
NONAFFINE = (1, 2, 4, 7, 8, 11, 13, 14)
OBJECTIVES = ({'invert': False}, {'invert': True})


# --------------------------------------------------------------------------
# scaffolds
# --------------------------------------------------------------------------
def _policy_tail(r, nodes, base_depth, relation_source='relation', goal_relation_source='goal_relation'):
    """z/world/prediction/policy/value tail, identical to examples/joint.py."""
    d = base_depth
    for name, source in [('z', goal_relation_source), ('world', relation_source)]:
        nodes.append(Node(name, F, (Candidate(r.resolve('encode', (BOOL,), F), (source,)),), 'encoding', d))
    for i in range(2):
        nodes.append(Node(f'mul{i}', F, (Candidate(r.resolve('mul', (F, F)), ('z', f'w{i}')),), 'policy', d + 1))
        nodes.append(Node(f'logit{i}', F, (Candidate(r.resolve('add', (F, F)), (f'mul{i}', f'bias{i}')),), 'policy', d + 2))
    nodes.append(Node('policy', product(F, F), (Candidate(r.resolve('tuple', (F, F)), ('logit0', 'logit1')),), 'policy', d + 3))
    nodes.append(Node('prediction', product(F, F), (Candidate(r.resolve('tuple', (F, F)), ('z', 'world')),), 'prediction', d + 1))
    nodes.append(Node('value', product(F), (Candidate(r.resolve('tuple', (F,)), ('baseline',)),), 'value', 1))


CONSTANTS = (('w0', Value.of(F, -2.)), ('w1', Value.of(F, 2.)),
             ('bias0', Value.of(F, 1.)), ('bias1', Value.of(F, -1.)),
             ('baseline', Value.of(F, 1.)))


def record_scaffold(host):
    """Byte-for-byte the graph of examples/joint.py: observes bits + goal only."""
    r = Registry()
    names = ('bits', 'goal')
    inputs = tuple((k, host.view().observations[k].type) for k in names) + (('action', product(F, F)), ('dt', F))
    nodes = []
    for i in range(2):
        op = r.resolve('project', (dict(inputs)['bits'],), parameters={'index': i})
        nodes.append(Node(f'bit_{i}', BOOL, (Candidate(op, ('bits',)),), 'observation', 1))
    for name, sources, depth in [('relation', ('bit_0', 'bit_1'), 2), ('goal_relation', ('relation', 'goal'), 3)]:
        nodes.append(Node(name, BOOL, tuple(Candidate(r.resolve(f'truth_{i}', (BOOL, BOOL)), sources) for i in range(16)), 'latent', depth))
    _policy_tail(r, nodes, 4)
    program = Program(inputs, tuple(nodes),
                      (('policy', 'policy'), ('prediction', 'prediction'), ('probe', 'prediction'), ('value', 'value')),
                      CONSTANTS, trainable_constants=tuple(k for k, _ in CONSTANTS))
    return names, SoftProgram(program)


def program_scaffold(host, interpreter_only=False, wired=False):
    """Record scaffold + the `program` observation + a table-conditioned lookup.

    `relation` gains a 17th candidate: index(<truth_0..truth_15 of the two gate
    inputs>, round(program[2])).  That candidate computes whichever gate the
    episode's truth-table field names, so it is the minimum structural change
    that could let one frozen program answer for gate families it never saw.
    With `wired=True` the two gate inputs are themselves selected out of `bits`
    by the wiring fields program[0]/program[1] instead of being hard-wired to
    bits 0 and 1, so the program also generalizes over gate wiring.
    """
    r = Registry()
    names = ('bits', 'goal', 'program')
    inputs = tuple((k, host.view().observations[k].type) for k in names) + (('action', product(F, F)), ('dt', F))
    ptype = dict(inputs)['program']
    btype = dict(inputs)['bits']
    if len(ptype.items) != 3:
        raise ValueError('table-conditioned scaffold is typed for a single-gate program observation')
    nodes = []
    for j, field in enumerate(('wire_a', 'wire_b', 'table_f')):
        nodes.append(Node(field, F, (Candidate(r.resolve('project', (ptype,), parameters={'index': j}), ('program',)),), 'observation', 1))
    for field, src in (('idx_a', 'wire_a'), ('idx_b', 'wire_b'), ('table_i', 'table_f')):
        nodes.append(Node(field, TABLE_INT, (Candidate(r.resolve('encode', (F,), TABLE_INT), (src,)),), 'observation', 2))
    for i, idx in enumerate(('idx_a', 'idx_b')):
        if wired:
            cand = Candidate(r.resolve('index', (btype, TABLE_INT)), ('bits', idx))
        else:
            cand = Candidate(r.resolve('project', (btype,), parameters={'index': i}), ('bits',))
        nodes.append(Node(f'bit_{i}', BOOL, (cand,), 'observation', 3))
    for j in range(16):
        nodes.append(Node(f'gate_{j}', BOOL, (Candidate(r.resolve(f'truth_{j}', (BOOL, BOOL)), ('bit_0', 'bit_1')),), 'latent', 4))
    gates = tuple(f'gate_{j}' for j in range(16))
    nodes.append(Node('gate_table', product(*(BOOL for _ in gates)),
                      (Candidate(r.resolve('tuple', tuple(BOOL for _ in gates)), gates),), 'latent', 5))
    lookup = Candidate(r.resolve('index', (product(*(BOOL for _ in gates)), TABLE_INT)), ('gate_table', 'table_i'))
    fixed = tuple(Candidate(r.resolve(f'truth_{i}', (BOOL, BOOL)), ('bit_0', 'bit_1')) for i in range(16))
    nodes.append(Node('relation', BOOL, (lookup,) if interpreter_only else fixed + (lookup,), 'latent', 6))
    nodes.append(Node('goal_relation', BOOL,
                      tuple(Candidate(r.resolve(f'truth_{i}', (BOOL, BOOL)), ('relation', 'goal')) for i in range(16)), 'latent', 7))
    _policy_tail(r, nodes, 8)
    program = Program(inputs, tuple(nodes),
                      (('policy', 'policy'), ('prediction', 'prediction'), ('probe', 'prediction'), ('value', 'value')),
                      CONSTANTS, trainable_constants=tuple(k for k, _ in CONSTANTS))
    return names, SoftProgram(program)


def residual_init(model):
    """Same symmetry break as examples/joint.py: prefer 'copy input one'."""
    with torch.no_grad():
        for n, p in zip(model.program.nodes, model.choices):
            if len(n.candidates) >= 16:
                p[12] = 2.


# --------------------------------------------------------------------------
# trainer with a per-episode generator configuration
# --------------------------------------------------------------------------
class ScheduledTrainer(JointTrainer):
    """JointTrainer whose generator configuration is a function of the episode.

    `TrainConfig.generator_config` is a single dict for a whole run, which is
    exactly why the record could only ever instantiate one Boolean function.
    """

    def __init__(self, model, config, schedule):
        super().__init__(model, config)
        self.schedule = schedule

    def episode(self, index, train=True, split='train', loss_only=False):
        self.config.generator_config = dict(self.schedule(index, split))
        return super().episode(index, train, split, loss_only)


ACTIONS = (Action('answer', arguments=(('value', Value.of(BOOL, False)),)),
           Action('answer', arguments=(('value', Value.of(BOOL, True)),)))


def make_config(names, settings, episodes, seed, horizon=4, lr=.04):
    return TrainConfig('logic', names,
                       (Target('probes', 'target', 1), Target('probes', 'gate', 1)),
                       ACTIONS, dict(settings), objectives=OBJECTIVES,
                       episodes=episodes, horizon=horizon, lr=lr, probe_weight=1., seed=seed)


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------
def fixed_schedule(settings):
    return lambda index, split: dict(settings)


def table_schedule(tables, base=None):
    """Table cycles every two episodes so each table meets both objectives."""
    base = dict(base or {'depth': 1, 'fixed_inputs': True})
    def f(index, split):
        return base | {'table': tables[(index // 2) % len(tables)]}
    return f


def depth_schedule(depths, base=None):
    base = dict(base or {'fixed_inputs': False})
    def f(index, split):
        return base | {'depth': depths[(index // 2) % len(depths)]}
    return f


# --------------------------------------------------------------------------
# evaluation and reference baselines
# --------------------------------------------------------------------------
def deterministic_returns(trainer, indices, split, schedule):
    saved = trainer.config.generator_config
    saved_schedule = trainer.schedule
    trainer.schedule = schedule
    try:
        out = [trainer.episode(i, train=False, split=split)[0]['return'] for i in indices]
    finally:
        trainer.schedule = saved_schedule
        trainer.config.generator_config = saved
    return out


def episode_targets(indices, split, schedule, seed=0, horizon=4):
    """The privileged `target` probe for each evaluation episode."""
    out = []
    for i in indices:
        goal = OBJECTIVES[i % len(OBJECTIVES)]
        host = Host.create('logic', seed=seed, index=i, split=split,
                           configuration=dict(schedule(i, split)) | {'horizon': horizon}, objective=goal)
        out.append(bool(host.records[0].probes['target'].decoded))
    return out


def constant_policy_returns(indices, split, schedule, answer, seed=0, horizon=4):
    returns = []
    for i in indices:
        goal = OBJECTIVES[i % len(OBJECTIVES)]
        host = Host.create('logic', seed=seed, index=i, split=split,
                           configuration=dict(schedule(i, split)) | {'horizon': horizon}, objective=goal)
        total = 0.
        for _ in range(horizon):
            if host.records[-1].done:
                break
            rec = host.step((Action('answer', arguments=(('value', Value.of(BOOL, answer)),)),), 1.)
            total += sum(v.decoded for v in rec.reward_components.values())
        returns.append(total)
    return returns


def random_policy_returns(indices, split, schedule, seed=0, horizon=4, rng_seed=0):
    rng = random.Random(rng_seed)
    returns = []
    for i in indices:
        goal = OBJECTIVES[i % len(OBJECTIVES)]
        host = Host.create('logic', seed=seed, index=i, split=split,
                           configuration=dict(schedule(i, split)) | {'horizon': horizon}, objective=goal)
        total = 0.
        for _ in range(horizon):
            if host.records[-1].done:
                break
            rec = host.step((Action('answer', arguments=(('value', Value.of(BOOL, rng.random() < .5)),)),), 1.)
            total += sum(v.decoded for v in rec.reward_components.values())
        returns.append(total)
    return returns


def summarize(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, n - 1)) if n > 1 else 0.
    return {'mean': m, 'sd': sd, 'sem': sd / math.sqrt(n) if n else 0., 'n': n,
            'min': min(xs) if xs else None, 'max': max(xs) if xs else None}


def baselines(indices, split, schedule, seed=0, horizon=4):
    t = episode_targets(indices, split, schedule, seed, horizon)
    return {
        'fraction_target_true': sum(t) / len(t),
        'always_true': summarize(constant_policy_returns(indices, split, schedule, True, seed, horizon)),
        'always_false': summarize(constant_policy_returns(indices, split, schedule, False, seed, horizon)),
        'uniform_random': summarize(random_policy_returns(indices, split, schedule, seed, horizon)),
        'majority_constant': max(sum(t) / len(t), 1 - sum(t) / len(t)) * horizon,
    }
