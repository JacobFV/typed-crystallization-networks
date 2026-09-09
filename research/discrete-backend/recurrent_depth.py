"""Acceptance test for the recurrent mode: the depth-generalization scaffold.

`tcn.search.enumerate_recurrent` is measured against the one program in this
repository that demonstrably needs a recurrence -- the circuit interpreter of
`research/depth-generalization/interpreter.py`, which evaluates one gate per tick
through `Program.state` and settles at exactly tick d-1.

Three things are measured on exactly the same scaffold, the same episodes and the
same candidate space:

* `enumerate_fit`, the feed-forward mode, which reads the program at tick 0 --
  before the fold has run -- and is the baseline the track's claim rests on.
* `enumerate_recurrent`, driven for HORIZON ticks and scored on the trailing
  SETTLE window, which is the new capability.
* the returned program applied at held-out depths 3, 4, 6 and 8, both as probe
  exactness and as actual environment return through `tcn.search.episode_return`.

Hand-initializations, declared. The scaffold is the depth track's, so its
declared priors carry over unchanged: the coarse graph (dispatch/read/evaluate/
commit/answer regions and their wiring), the gate-field reader modules, and the
five policy-tail constants w0=-2, w1=2, bias0=1, bias1=-1, baseline=1, which
already implement the correct decision rule before any search. What is searched
is the discrete program alone: the gate `relation` (17 candidates) and the
`goal_relation` (16), plus in the `free` arm the two wire lookups (2 each). No
choice is initialized; enumeration visits every one of them.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research' / 'depth-generalization'))

from tcn.generation import Host, Action
from tcn.graph import Signal
from tcn.operators import Registry
from tcn.search import (EnvironmentTask, EpisodeLedger, enumerate_fit, enumerate_recurrent,
                        evaluate_recurrent, episode_return, frozen_selection, space_size,
                        candidate_counts, viability)
from tcn.types import BOOL, Value, floating, product

import interpreter as I

F = floating()
WIDTH, CAPACITY, HORIZON, SETTLE = 4, 8, 12, 4
BASE = {'inputs': WIDTH, 'gate_capacity': CAPACITY, 'nondegenerate': True, 'min_relevant_inputs': 2}
TRAIN_DEPTHS = (1, 2)
EVAL_DEPTHS = (1, 2, 3, 4, 6, 8)
TRAIN_INDICES = tuple(range(16))
EVAL_INDICES = tuple(range(10000, 10040))
OUT = Path(__file__).parent / 'out'

# `answer_now` is the circuit's own output, `goal_relation` the objective-adjusted
# decision. The generator publishes both on the probe channel as `gate` and
# `target`; nothing is read off the observation stream that the actor may not see.
SIGNALS = (Signal('answer_now', 'gate', ('answer',), BOOL, 'bce'),
           Signal('goal_relation', 'target', ('answer',), BOOL, 'bce'))


def scaffold(wire_choice):
    host = Host.create('logic', seed=0, index=0, split='train',
                       configuration=BASE | {'depth': 1, 'horizon': HORIZON})
    return I.interpreter_scaffold(host, WIDTH, CAPACITY, HORIZON, wire_choice=wire_choice)


def examples(names, indices, depths, split, ledger):
    """One recurrent example per episode: constant inputs, terminal targets.

    Inputs are constant across ticks because the `logic` state is constant: no
    action changes it, so an episode's observation is the same record at every
    tick. The recurrence is the program's, not the environment's.
    """
    rows = []
    for position, index in enumerate(indices):
        depth = depths[(index // 2) % len(depths)] if len(depths) > 1 else depths[0]
        host = ledger.create('logic', seed=0, index=index, split=split,
                             configuration=BASE | {'depth': depth, 'horizon': HORIZON},
                             objective=I.OBJECTIVES[position % len(I.OBJECTIVES)])
        view = host.view(); record = host.records[0]
        inputs = {k: view.observations[k] for k in names}
        inputs['action'] = Value.of(product(F, F), (1., 0.))
        inputs['dt'] = Value.of(F, 1.)
        rows.append({'inputs': inputs, 'depth': depth,
                     'targets': {'gate': record.probes['gate'], 'target': record.probes['target']}})
    return rows


def environment_task(names, indices, depth):
    return EnvironmentTask(generator='logic', observations=tuple(names),
                           action_templates=I.ACTIONS,
                           generator_config=BASE | {'depth': depth},
                           objectives=I.OBJECTIVES, indices=tuple(indices),
                           horizon=HORIZON, seed=0, split='test', scored_ticks=SETTLE)


def arm(kind):
    wire_choice = kind == 'free'
    names, program, registry = scaffold(wire_choice)
    ledger = EpisodeLedger()
    train = examples(names, TRAIN_INDICES, TRAIN_DEPTHS, 'train', ledger)
    row = {'arm': kind, 'space_size': space_size(program),
           'candidate_counts': list(candidate_counts(program)),
           'free_nodes': {n.name: len(n.candidates) for n in program.nodes if len(n.candidates) > 1},
           'nodes': len(program.nodes), 'program_digest': program.digest,
           'train_episodes': len(train), 'ticks': HORIZON, 'settle_window': SETTLE,
           'viability': viability(program, examples=len(train), ticks=HORIZON)}

    # Baseline: the feed-forward mode on the identical program and data. It runs
    # tick 0 only, so the fold has not happened and the answer wire does not exist.
    row['enumerate_fit'] = enumerate_fit(program, train, SIGNALS, registry, tolerance=1e-6).to_dict()

    found = enumerate_recurrent(program, train, SIGNALS, registry, ticks=HORIZON,
                                settle_window=SETTLE, tolerance=1e-6)
    row['enumerate_recurrent'] = found.to_dict()
    if not found.solved:
        row['holdout'] = None
        return row

    selections = found.selections
    row['selection_free'] = {k: v for k, v in selections.items() if k in row['free_nodes']}
    exact = frozen_selection(program, selections, registry)
    row['holdout'] = {}
    for depth in EVAL_DEPTHS:
        held = examples(names, EVAL_INDICES, (depth,), 'test', ledger)
        error = evaluate_recurrent(program, selections, held, SIGNALS, registry,
                                   ticks=HORIZON, settle_window=SETTLE)
        task = environment_task(names, EVAL_INDICES, depth)
        returns = [episode_return(exact, registry, task, index, position, ledger)[0]
                   for position, index in enumerate(EVAL_INDICES)]
        row['holdout'][str(depth)] = {
            'episodes': len(EVAL_INDICES), 'probe_max_error': error,
            'mean_settled_return': sum(returns) / len(returns),
            'exact_episodes': sum(1 for x in returns if x == SETTLE)}
    row['environment_episodes_total'] = ledger.episodes
    row['environment_steps_total'] = ledger.steps
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {'width': WIDTH, 'capacity': CAPACITY, 'horizon': HORIZON, 'settle_window': SETTLE,
              'train_depths': list(TRAIN_DEPTHS), 'eval_depths': list(EVAL_DEPTHS), 'arms': []}
    for kind in ('pinned', 'free'):
        row = arm(kind)
        report['arms'].append(row)
        f, r = row['enumerate_fit'], row['enumerate_recurrent']
        print(f"{kind:7s} space {row['space_size']:5d}  fit: conforming {f['conforming']} "
              f"({f['seconds']:.1f}s)  recurrent: conforming {r['conforming']} "
              f"certificate {r['certificate']} ({r['seconds']:.1f}s)", flush=True)
        if row.get('holdout'):
            for depth, h in row['holdout'].items():
                print(f"    d{depth}: probe max error {h['probe_max_error']} "
                      f"return {h['mean_settled_return']:.2f}/{SETTLE} "
                      f"({h['exact_episodes']}/{h['episodes']} exact)", flush=True)
    report['seconds'] = time.time() - started
    (OUT / 'recurrent_depth.json').write_text(json.dumps(report, indent=2))
    print(f"total {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
