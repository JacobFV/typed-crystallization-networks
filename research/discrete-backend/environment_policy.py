"""Acceptance test for the environment mode: the policy track's 27 episodes.

FINDINGS section 22 ranks the approaches to `logic` by the resource that actually
matters -- environment episodes -- and the cheapest by an order of magnitude is
"frozen exact model + enumeration, no policy at all" at 27 episodes, 25 of them
supervision and 2 of them reward. It was measured with a hand-rolled loop
(`research/policy-learning/e6_mpc.py`). This reproduces the number through
`tcn.search`, with every episode charged to one `EpisodeLedger` across both
stages, and adds the two comparisons the bespoke script could not make:

* what the same answer costs with no staging at all -- enumerate the whole space
  by live return -- which is where the mode stops being viable;
* whether the staged sweep produces a certificate, which the gradient arm of the
  original could not.

Hand-initializations, declared. The scaffold is `examples/joint.py`'s with the
hand-supplied policy decoder removed, as `research/policy-learning/pl.py`
requires: the two observation projections and the value baseline are declared
plumbing, and the readout is a two-candidate node rather than four trainable
constants, so the one bit reward has to supply is a discrete choice the backend
can enumerate instead of a continuous parameter it cannot. Nothing is initialized
toward an answer; all 512 programs are visited in the unstaged arm and all 2
survivors in the staged one.
"""
from __future__ import annotations
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcn.generation import Host, Action
from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.search import (EnvironmentTask, EpisodeLedger, enumerate_environment, enumerate_fit,
                        episode_return, frozen_selection, probe_examples, space_size,
                        trailing_return, viability)
from tcn.types import BOOL, Value, floating, product

F = floating()
SETTINGS = {'depth': 1, 'table': 6, 'fixed_inputs': True}
ACTIONS = (Action('answer', arguments=(('value', Value.of(BOOL, False)),)),
           Action('answer', arguments=(('value', Value.of(BOOL, True)),)))
OBJECTIVES = ({'invert': False}, {'invert': True})
HORIZON = 4
PROBE_EPISODES = 25
SIGN_EPISODES = 1                       # per candidate sign; two candidates -> two episodes
EVAL = 64
OUT = Path(__file__).parent / 'out'

OBSERVATIONS = ('bits', 'goal')
SIGNALS = (Signal('relation', 'gate', ('latent',), BOOL, 'bce'),
           Signal('goal_relation', 'target', ('latent',), BOOL, 'bce'))
PROBE_TARGETS = (('gate', 'probes', 'gate'), ('target', 'probes', 'target'))


def scaffold():
    """The joint scaffold with the readout sign exposed as a discrete choice.

    `examples/joint.py` puts the readout in four trainable constants, which no
    discrete backend can search. The content is one bit -- whether the policy
    prefers the action equal to the decision or its complement -- so it is a
    two-candidate node here. That is a representation change, not an easier
    problem: the same one bit still has to come from reward.
    """
    r = Registry()
    host = Host.create('logic', configuration=SETTINGS)
    inputs = tuple((k, host.view().observations[k].type) for k in OBSERVATIONS) + \
             (('action', product(F, F)), ('dt', F))
    bits = dict(inputs)['bits']
    constants = (('half', Value.of(F, .5)), ('baseline', Value.of(F, 1.)))
    nodes = []
    for i in range(2):
        nodes.append(Node(f'bit_{i}', BOOL,
                          (Candidate(r.resolve('project', (bits,), parameters={'index': i}), ('bits',)),),
                          'observation', 1))
    for name, sources, depth in (('relation', ('bit_0', 'bit_1'), 2),
                                 ('goal_relation', ('relation', 'goal'), 3)):
        nodes.append(Node(name, BOOL,
                          tuple(Candidate(r.resolve(f'truth_{i}', (BOOL, BOOL)), sources) for i in range(16)),
                          'latent', depth))
    for name, source in (('z', 'goal_relation'), ('world', 'relation')):
        nodes.append(Node(name, F, (Candidate(r.resolve('encode', (BOOL,), F), (source,)),), 'encoding', 4))
    nodes.append(Node('zpos', F, (Candidate(r.resolve('sub', (F, F)), ('z', 'half')),), 'policy', 5))
    nodes.append(Node('zneg', F, (Candidate(r.resolve('sub', (F, F)), ('half', 'z')),), 'policy', 5))
    nodes.append(Node('policy', product(F, F),
                      (Candidate(r.resolve('tuple', (F, F)), ('zneg', 'zpos')),
                       Candidate(r.resolve('tuple', (F, F)), ('zpos', 'zneg'))), 'policy', 6))
    nodes.append(Node('prediction', product(F, F), (Candidate(r.resolve('tuple', (F, F)), ('z', 'world')),), 'prediction', 5))
    nodes.append(Node('value', product(F), (Candidate(r.resolve('tuple', (F,)), ('baseline',)),), 'value', 1))
    program = Program(inputs, tuple(nodes),
                      (('policy', 'policy'), ('prediction', 'prediction'),
                       ('probe', 'prediction'), ('value', 'value')), constants).validate(r)
    return program, r


def task(indices, split, seed=0):
    return EnvironmentTask(generator='logic', observations=OBSERVATIONS, action_templates=ACTIONS,
                           generator_config=SETTINGS, objectives=OBJECTIVES,
                           indices=tuple(indices), horizon=HORIZON, seed=seed, split=split)


CONSTANT_INPUTS = {'action': Value.of(product(F, F), (1., 0.)), 'dt': Value.of(F, 1.)}


def reference_returns(indices, ledger):
    """The same four references section 22 reports, on the same episodes."""
    names = ('always_false', 'always_true', 'uniform', 'oracle')
    totals = {k: 0. for k in names}
    rng = random.Random(12345)
    t = task(indices, 'test')
    for position, index in enumerate(indices):
        for name in names:
            host = ledger.create('logic', seed=0, index=index, split='test',
                                 configuration=SETTINGS | {'horizon': HORIZON},
                                 objective=t.objective(position))
            for _ in range(HORIZON):
                view = host.view()
                b = view.observations['bits'].decoded; g = view.observations['goal'].decoded
                a = {'always_false': 0, 'always_true': 1, 'uniform': rng.randrange(2),
                     'oracle': int((b[0] ^ b[1]) ^ g)}[name]
                if host.step((ACTIONS[a],), 1.).done: break
            ledger.charge_steps(host)
            totals[name] += trailing_return(host)
    return {k: v / len(indices) for k, v in totals.items()}


def staged():
    """25 supervised episodes, then 2 reward episodes, on one ledger."""
    program, registry = scaffold()
    ledger = EpisodeLedger()
    started = time.time()
    train = probe_examples(task(range(PROBE_EPISODES), 'train'), PROBE_TARGETS, ledger,
                           extra_inputs=CONSTANT_INPUTS)
    supervised = enumerate_fit(program, train, SIGNALS, registry, tolerance=1e-6)
    stage1_episodes = ledger.episodes

    # The supervised stage fixes everything the probes can see and leaves exactly
    # the readout bit free. Hardening what it decided leaves a two-program space.
    decided = {k: v for k, v in supervised.selections.items() if k in {'relation', 'goal_relation'}}
    residual = program.harden(decided)
    sign_task = task((700000,), 'train')
    reward = enumerate_environment(residual, sign_task, registry, threshold=float(HORIZON),
                                   ledger=ledger)
    search_episodes = ledger.episodes

    # `reward.selections` is indexed against `residual`, whose supervised nodes now
    # hold one candidate each, so it must be applied to `residual` rather than
    # merged into the original scaffold's indices.
    chosen = decided | {'policy': reward.selections['policy']}
    exact = frozen_selection(residual, reward.selections, registry)
    evaluation = EpisodeLedger()
    indices = tuple(range(10000, 10000 + EVAL))
    t = task(indices, 'test')
    returns = [episode_return(exact, registry, t, index, position, evaluation)[0]
               for position, index in enumerate(indices)]
    references = reference_returns(indices, evaluation)
    return {'space_size': space_size(program),
            'supervised': supervised.to_dict(), 'reward': reward.to_dict(),
            'stage1_episodes': stage1_episodes,
            'stage2_episodes': search_episodes - stage1_episodes,
            'search_episodes': search_episodes, 'search_steps': ledger.steps,
            'residual_space': space_size(residual),
            'selection': chosen,
            'evaluation_episodes': evaluation.episodes,
            'held_out_mean_return': sum(returns) / len(returns),
            'held_out_perfect': sum(1 for x in returns if x == HORIZON),
            'references': references, 'seconds': time.time() - started}


def unstaged(episodes_per_program):
    """No staging: enumerate the whole space by live return. The honest ceiling."""
    program, registry = scaffold()
    ledger = EpisodeLedger()
    started = time.time()
    indices = tuple(700000 + i for i in range(episodes_per_program))
    found = enumerate_environment(program, task(indices, 'train'), registry,
                                  threshold=float(HORIZON), ledger=ledger)
    row = found.to_dict()
    row['episodes_per_program'] = episodes_per_program
    row['seconds'] = time.time() - started
    row['viability'] = viability(program, episodes=episodes_per_program)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'settings': SETTINGS, 'horizon': HORIZON, 'probe_episodes': PROBE_EPISODES,
              'evaluation_episodes': EVAL}
    report['staged'] = staged()
    s = report['staged']
    print(f"staged: supervision {s['stage1_episodes']} episodes -> conforming "
          f"{s['supervised']['conforming']}/{s['space_size']} certificate "
          f"{s['supervised']['certificate']}; reward {s['stage2_episodes']} episodes over a "
          f"{s['residual_space']}-program residual, certificate {s['reward']['certificate']}")
    print(f"        SEARCH TOTAL {s['search_episodes']} environment episodes "
          f"({s['search_steps']} steps), held-out return "
          f"{s['held_out_mean_return']:.2f}/{HORIZON} "
          f"({s['held_out_perfect']}/{EVAL} perfect) against " +
          ' '.join(f"{k} {v:.2f}" for k, v in s['references'].items()))
    report['unstaged'] = [unstaged(n) for n in (1, 4, 16)]
    for row in report['unstaged']:
        print(f"unstaged n={row['episodes_per_program']}: {row['episodes']} episodes, "
              f"{row['evaluated']}/{row['space_size']} programs, conforming {row['conforming']}, "
              f"certificate {row['certificate']}, best return {row['best_return']}, "
              f"{row['seconds']:.1f}s")
    (OUT / 'environment_policy.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
