"""Does the recurrent interpreter actually compute the circuit, exactly?

Hardens the two searched nodes to the reference program -- `relation` = the
table-conditioned `index` lookup, `goal_relation` = xor with the objective bit --
and steps the graph tick by tick against the generator's own `values` latent.
No training, no relaxation: this is `Program.execute` only, so it answers the
expressibility question on its own.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcn.generation import Host
from tcn.types import Value, product, floating, BOOL
from interpreter import interpreter_scaffold, WIRE

F = floating()


def reference_selections(program, wire_choice):
    """The correct discrete program: the lookup candidate, and xor at the goal node."""
    out = {}
    for n in program.nodes:
        if n.selected is not None:
            continue
        if n.name == 'relation':
            out[n.name] = len(n.candidates) - 1              # the `index` lookup
        elif n.name == 'goal_relation':
            out[n.name] = 6                                  # truth_6 = xor
        elif n.name in {'va', 'vb'}:
            out[n.name] = 0                                  # the general mux lookup
        elif len(n.candidates) == 1:
            out[n.name] = 0
        else:
            raise AssertionError('unexpected free node ' + n.name)
    return out


def trace_episode(program, registry, selections, host, width, horizon, names):
    """Run the recurrence for `horizon` ticks and report the answer wire each tick."""
    types = dict(program.inputs)
    state = None
    answers = []
    for t in range(horizon):
        view = host.view()
        inputs = {k: view.observations[k] for k in names}
        inputs['action'] = Value.of(types['action'], (1., 0.))
        inputs['dt'] = Value.of(F, 1.)
        out, state, trace = program.execute(inputs, state, registry, selections)
        answers.append(bool(trace['answer_now'].decoded))
    return answers


def main():
    width, capacity, horizon = 4, 8, 12
    base = {'inputs': width, 'gate_capacity': capacity, 'nondegenerate': True,
            'min_relevant_inputs': 2, 'horizon': horizon}
    report = {'width': width, 'capacity': capacity, 'horizon': horizon, 'depths': {}}
    seed_host = Host.create('logic', seed=0, index=0, split='test', configuration=base | {'depth': 1})
    names, program, registry = interpreter_scaffold(seed_host, width, capacity, horizon, wire_choice=True)
    selections = reference_selections(program, True)
    report['nodes'] = len(program.nodes)
    report['free_nodes'] = {n.name: len(n.candidates) for n in program.nodes if n.selected is None}
    report['program_digest'] = program.digest

    for depth in (1, 2, 3, 4, 6, 8):
        agree = 0
        total = 0
        first_correct = []
        for index in range(40):
            host = Host.create('logic', seed=0, index=index, split='test',
                               configuration=base | {'depth': depth})
            truth = bool(host.state['values'][-1]) != bool(host.state['invert'])
            answers = trace_episode(program, registry, selections, host, width, horizon, names)
            settled = answers[horizon - 4:]
            total += 1
            agree += all(a == truth for a in settled)
            correct_from = next((i for i in range(horizon) if all(a == truth for a in answers[i:])), None)
            first_correct.append(correct_from)
        report['depths'][depth] = {
            'episodes': total,
            'settled_exact': agree,
            'first_settled_tick_max': max(first_correct),
            'first_settled_tick_mean': sum(first_correct) / len(first_correct),
        }
        print(depth, report['depths'][depth])

    Path(__file__).parent.joinpath('out/exactness.json').write_text(json.dumps(report, indent=2))
    assert all(v['settled_exact'] == v['episodes'] for v in report['depths'].values())
    print('exact at every depth')


if __name__ == '__main__':
    main()
