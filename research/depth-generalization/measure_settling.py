"""What the trailing-window convention costs: the same program scored three ways.

The recurrence needs d-1 ticks before the answer exists, so a return summed from
tick 0 charges the interpreter for its own settling. This measures the reference
program under three scoring conventions on the same 64 held-out episodes, next
to the constant and random baselines under each.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tcn.generation import Host
from tcn.agent import Agent
import interpreter as I
from run import BASE, WIDTH, CAPACITY, HORIZON, SCORED, EVAL_INDICES, frozen, build
from check_exact import reference_selections


def window(host, first, count):
    rewards = [sum(v.decoded for v in r.reward_components.values()) for r in host.records[1:]]
    return float(sum(rewards[first:first + count]))


def main():
    names, program, registry, model = build('interpreter', 0)
    exact = frozen(program, reference_selections(program, True), registry)
    config = I.make_config(names, BASE | {'depth': 1}, 8, 0, HORIZON)
    rows = {}
    for depth in (1, 2, 3, 4, 6, 8):
        agent = Agent(exact, registry, config, seed=0)
        trailing, first4, whole = [], [], []
        for i in EVAL_INDICES:
            host = Host.create('logic', seed=0, index=i, split='test',
                               configuration=BASE | {'depth': depth, 'horizon': HORIZON},
                               objective=I.OBJECTIVES[i % 2])
            agent.rollout(host, deterministic=True)
            trailing.append(window(host, HORIZON - SCORED, SCORED))
            first4.append(window(host, 0, SCORED))
            whole.append(window(host, 0, HORIZON) / HORIZON * SCORED)
        rows[str(depth)] = {'trailing_4': sum(trailing) / len(trailing),
                            'first_4': sum(first4) / len(first4),
                            'whole_horizon_rescaled': sum(whole) / len(whole)}
        print(depth, rows[str(depth)], flush=True)
    Path(__file__).parent.joinpath('out/settling.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
