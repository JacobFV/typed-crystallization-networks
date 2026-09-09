"""Depth generalization: train at depths 1-2, evaluate at depths 3, 4, 6 and 8.

Every condition is evaluated on the same held-out episode addresses, against the
two constant policies and a uniform-random policy rolled on those same episodes,
over the same trailing scored window.

    python run.py representation
    python run.py baselines
    python run.py interpreter
    python run.py interpreter_fixed
    python run.py record
    python run.py enumerate
"""
from __future__ import annotations
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from dataclasses import replace

from tcn.generation import Host
from tcn.learning import SoftProgram
from tcn.agent import Agent
from tcn.search import space_size, candidate_counts
from tcn.types import Value, floating, BOOL, product

import interpreter as I

WIDTH, CAPACITY, HORIZON, SCORED = 4, 8, 12, 4
BASE = {'inputs': WIDTH, 'gate_capacity': CAPACITY, 'nondegenerate': True, 'min_relevant_inputs': 2}
TRAIN_DEPTHS = (1, 2)
EVAL_DEPTHS = (1, 2, 3, 4, 6, 8)
EVAL_INDICES = tuple(range(10000, 10064))
SEEDS = tuple(range(8))
EPISODES = 320
NOISE = .05
OUT = Path(__file__).parent / 'out'


def frozen(program, selections, registry=None):
    """Harden the discrete choices and drop the trainable-constant flag.

    The five policy-tail constants keep their declared values. As FINDINGS
    section 7 records for `examples/joint.py`, that readout already implements
    the correct decision rule before any training, so what enumeration searches
    is the discrete program alone -- exactly what `tcn.search` addresses and
    exactly what step 4's own enumeration searched.
    """
    return replace(program.harden(selections), trainable_constants=()).validate(registry)


def host_for_scaffold(seed=0):
    return Host.create('logic', seed=seed, index=0, split='train',
                       configuration=BASE | {'depth': 1, 'horizon': HORIZON})


def build(kind, seed):
    host = host_for_scaffold()
    if kind == 'record':
        names, program, registry = I.record_scaffold(host)
    else:
        names, program, registry = I.interpreter_scaffold(
            host, WIDTH, CAPACITY, HORIZON,
            wire_choice=kind.startswith('interpreter') and 'fixed' not in kind,
            settle_mux='nomux' not in kind)
    model = SoftProgram(program, registry)
    # SoftProgram zero-initializes every choice logit, so torch.manual_seed does
    # not vary synthesis (FINDINGS section 11, fault P2). Without this the eight
    # "seeds" below would be one outcome repeated eight times.
    I.perturb(model, NOISE, seed)
    return names, program, registry, model


def exact_returns(program, registry, config, indices, depth, seed):
    agent = Agent(program, registry, config, seed=seed)
    schedule = I.fixed_depth(depth, BASE)
    out = []
    for i in indices:
        host = Host.create('logic', seed=config.seed, index=i, split='test',
                           configuration=dict(schedule(i, 'test')) | {'horizon': HORIZON},
                           objective=I.OBJECTIVES[i % len(I.OBJECTIVES)])
        agent.rollout(host, deterministic=True)
        out.append(I.settled_return(host, SCORED))
    return out


def soft_returns(trainer, indices, depth):
    return I.deterministic_settled(trainer, indices, 'test', I.fixed_depth(depth, BASE), SCORED)


def selections_report(program, model):
    picked = model.selections()
    return {n.name: picked[n.name] for n in program.nodes if len(n.candidates) > 1}


# ---------------------------------------------------------------------------
def condition_baselines():
    out = {}
    for depth in EVAL_DEPTHS:
        out[str(depth)] = I.baselines(EVAL_INDICES, 'test', I.fixed_depth(depth, BASE),
                                      seed=0, horizon=HORIZON, scored_ticks=SCORED)
        print(depth, json.dumps(out[str(depth)]))
    (OUT / 'baselines.json').write_text(json.dumps(out, indent=2))


def condition_representation():
    """The representational fact, measured rather than asserted."""
    report = {'observation_widths': {}, 'gate_set': {}}
    for depth in EVAL_DEPTHS:
        host = Host.create('logic', seed=0, index=0, split='test',
                           configuration=BASE | {'depth': depth, 'horizon': HORIZON})
        view = host.view().observations
        report['observation_widths'][str(depth)] = {
            'program_tuple_fields': len(view['program'].type.items),
            'program_flat_width': view['program'].type.width,
            'gates_capacity': view['gates'].type.capacity,
            'gates_flat_width': view['gates'].type.width,
            'gates_cardinality': len(view['gates'].raw),
            'gates_type_digest': json.dumps(view['gates'].type.to_dict(), sort_keys=True),
        }
    types = {v['gates_type_digest'] for v in report['observation_widths'].values()}
    report['gate_set']['one_type_at_every_depth'] = len(types) == 1
    report['gate_set']['program_types_distinct'] = len(
        {v['program_tuple_fields'] for v in report['observation_widths'].values()})

    # The step-4 scaffold, built for a depth-1 `program` observation, refused at
    # any other depth: this is the blocker restated as an executable check.
    failures = {}
    from research_step4_probe import step4_lookup_accepts
    for depth in EVAL_DEPTHS:
        failures[str(depth)] = step4_lookup_accepts(depth, BASE, HORIZON)
    report['step4_lookup_scaffold'] = failures
    (OUT / 'representation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def condition_gradients():
    """Where a gradient reaches, at initialization, node by node.

    `insert`, `member`, `pair`, `map`, `filter` and set `sum` declare
    gradient="none", and `exact_tensor` detaches its inputs, so the whole
    recurrence is a gradient dead end. This measures which choice logits still
    receive a gradient from one training episode, with and without the two-node
    `is_final` mux that reads the last gate directly off `relation`.
    """
    report = {}
    for kind in ('interpreter', 'interpreter_nomux'):
        names, program, registry, model = build(kind, 0)
        config = I.make_config(names, BASE | {'depth': 1}, 8, 0, HORIZON)
        trainer = I.ScheduledTrainer(model, config, I.depth_schedule(TRAIN_DEPTHS, BASE), SCORED)
        for regularized in (True, False):
            # `regularized=False` drops the discreteness term over the choice
            # distributions. FINDINGS section 3 records that this term keeps a
            # severed logit reachable in the autograd graph, so it is the
            # difference between "a gradient exists" and "a task gradient
            # exists"; both are reported.
            trainer.episode(0, regularized=regularized)
            label = kind + ('' if regularized else '_task_only')
            report[label] = {n.name: (float(p.grad.norm()) if p.grad is not None else None)
                             for n, p in zip(program.nodes, model.choices) if len(n.candidates) > 1}
    (OUT / 'gradients.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def condition_train(kind):
    started = time.time()
    results = {'kind': kind, 'train_depths': TRAIN_DEPTHS, 'episodes': EPISODES,
               'noise': NOISE, 'seeds': list(SEEDS), 'eval_indices': len(EVAL_INDICES),
               'horizon': HORIZON, 'scored_ticks': SCORED, 'per_seed': []}
    for seed in SEEDS:
        names, program, registry, model = build(kind, seed)
        results['space_size'] = space_size(program)
        results['candidate_counts'] = list(candidate_counts(program))
        results['nodes'] = len(program.nodes)
        config = I.make_config(names, BASE | {'depth': 1}, EPISODES, 0, HORIZON)
        trainer = I.ScheduledTrainer(model, config, I.depth_schedule(TRAIN_DEPTHS, BASE), SCORED)
        history = trainer.run()
        row = {'seed': seed, 'selections': selections_report(program, model),
               'train_return_last32': sum(h['settled_return'] for h in history[-32:]) / 32,
               'soft': {}, 'exact': {}}
        exported = model.export()
        for depth in EVAL_DEPTHS:
            row['soft'][str(depth)] = I.summarize(soft_returns(trainer, EVAL_INDICES, depth))
            row['exact'][str(depth)] = I.summarize(
                exact_returns(exported, registry, config, EVAL_INDICES, depth, seed))
        results['per_seed'].append(row)
        print(f'seed {seed} {round(time.time()-started)}s '
              f'sel={row["selections"]} '
              + ' '.join(f'd{d}:{row["exact"][str(d)]["mean"]:.2f}' for d in EVAL_DEPTHS), flush=True)
    results['seconds'] = time.time() - started
    results['aggregate'] = {
        which: {str(d): I.summarize([r[which][str(d)]['mean'] for r in results['per_seed']])
                for d in EVAL_DEPTHS}
        for which in ('soft', 'exact')}
    (OUT / f'{kind}.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results['aggregate'], indent=2))


def condition_enumerate(kind='interpreter'):
    """The discrete reference over exactly the space `SoftProgram` relaxes.

    `tcn.search.enumerate_fit` fits a feed-forward program against supervised
    signals and cannot drive a recurrence, so this walks the same product of node
    candidates -- `tcn.search.candidate_counts` -- and scores each discrete
    program by actual settled return on the training episodes, the objective the
    gradient run optimizes. Exhausting the space certifies uniqueness.
    """
    started = time.time()
    names, program, registry, model = build(kind, 0)
    counts = candidate_counts(program)
    free = [(i, n.name, len(n.candidates)) for i, n in enumerate(program.nodes) if len(n.candidates) > 1]
    config = I.make_config(names, BASE | {'depth': 1}, EPISODES, 0, HORIZON)
    train_indices = tuple(range(16))
    schedule = I.depth_schedule(TRAIN_DEPTHS, BASE)
    node_names = [n.name for n in program.nodes]
    best, found = -1., []
    evaluated = 0
    for combination in itertools.product(*(range(c) for _, _, c in free)):
        selections = {name: 0 for name in node_names}
        selections.update({name: v for (_, name, _), v in zip(free, combination)})
        exact = frozen(program, selections, registry)
        agent = Agent(exact, registry, config, seed=0)
        total = 0.
        for i in train_indices:
            host = Host.create('logic', seed=0, index=i, split='train',
                               configuration=dict(schedule(i, 'train')) | {'horizon': HORIZON},
                               objective=I.OBJECTIVES[i % len(I.OBJECTIVES)])
            agent.rollout(host, deterministic=True)
            total += I.settled_return(host, SCORED)
        score = total / len(train_indices)
        evaluated += 1
        if score > best:
            best, found = score, [dict(zip([n for _, n, _ in free], combination))]
        elif score == best:
            found.append(dict(zip([n for _, n, _ in free], combination)))
    report = {'kind': kind, 'space_size': space_size(program), 'candidate_counts': list(counts),
              'free_nodes': [(n, c) for _, n, c in free], 'evaluated': evaluated,
              'exhausted': evaluated == space_size(program), 'best_train_return': best,
              'optimal_count': len(found), 'unique': len(found) == 1,
              'selection': found[0], 'seconds': time.time() - started,
              'train_episodes': len(train_indices), 'holdout': {}}
    selections = {name: 0 for name in node_names} | found[0]
    exact = frozen(program, selections, registry)
    for depth in EVAL_DEPTHS:
        report['holdout'][str(depth)] = I.summarize(
            exact_returns(exact, registry, config, EVAL_INDICES, depth, 0))
    (OUT / f'enumerate_{kind}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    what = sys.argv[1]
    if what == 'gradients':
        condition_gradients()
    elif what == 'baselines':
        condition_baselines()
    elif what == 'representation':
        condition_representation()
    elif what.startswith('enumerate'):
        condition_enumerate(sys.argv[2] if len(sys.argv) > 2 else 'interpreter')
    else:
        condition_train(what)
