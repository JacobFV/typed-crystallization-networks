"""Arithmetic/sine scaffold on the same logic task examples/joint.py solves.

Reconstruction of the `artifacts/joint-long` experiment recorded as unsuccessful
in docs/VALIDATION.md, plus knobs for ablating training-dynamics hypotheses.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tcn.types import BOOL, Value, product
from tcn.generation import Host, Action
from tcn.scaffold import F, arithmetic_scaffold
from tcn.learning import SoftProgram
from tcn.training import TrainConfig, Target, JointTrainer

SETTINGS = {'depth': 1, 'table': 6, 'fixed_inputs': True}
NAMES = ('bits', 'goal')


def program(hidden=8, seed=0, probe=False):
    host = Host.create('logic', configuration=SETTINGS)
    inputs = tuple((k, host.view().observations[k].type) for k in NAMES) + (('action', product(F, F)), ('dt', F))
    outputs = {'policy': 2, 'value': 1, 'prediction': 2}
    if probe:
        outputs['probe'] = 2
    return arithmetic_scaffold(inputs, outputs, hidden=hidden, seed=seed)


def trainer(episodes=640, seed=0, hidden=8, lr=.04, probe=False, **overrides):
    model = SoftProgram(program(hidden=hidden, seed=seed, probe=probe))
    actions = (Action('answer', arguments=(('value', Value.of(BOOL, False)),)),
               Action('answer', arguments=(('value', Value.of(BOOL, True)),)))
    kw = dict(objectives=({'invert': False}, {'invert': True}), episodes=episodes, horizon=4,
              lr=lr, probe_weight=1. if probe else 0., seed=seed)
    kw.update(overrides)
    config = TrainConfig('logic', NAMES,
                         (Target('probes', 'target', 1), Target('probes', 'gate', 1)),
                         actions, SETTINGS, **kw)
    return JointTrainer(model, config)


def evaluate(trainer_obj, episodes=16, split='test'):
    total = 0.
    for i in range(episodes):
        metrics, _ = trainer_obj.episode(i, train=False, split=split)
        total += metrics['return']
    return total / episodes


def summarize(history, window=8):
    first = history[:window]
    last = history[-window:]
    return {
        'episodes': len(history),
        'first_prediction_loss': sum(h['prediction_loss'] for h in first) / len(first),
        'final_prediction_loss': sum(h['prediction_loss'] for h in last) / len(last),
        'final_mean_return': sum(h['return'] for h in last) / len(last),
        'mean_return_all': sum(h['return'] for h in history) / len(history),
    }
