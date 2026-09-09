"""Complete inference path for the computer-use agent (research/computer-capability).

The artifact is the frozen 23-node `agent_program` rebuilt from the enumerated
selections in `out/search.json`, exactly as `closed_loop.py` builds it. It reaches
10/10 on held-out documents in that track; nothing is re-searched here.

The path here has a stage the other artifacts do not: the observation is produced
by a live OS bridge (`generators/computer/engine`), so `acquire` is a real process
round-trip and not a synthetic render. That is the point of measuring it.

  acquire  `Host.step` -> the kernel executes the action and returns a terminal
  encode   already typed by the generator; `Agent.act` additionally builds the
           `action` / `action.2.text` port Values every step
  execute  `program.run` -- run TWICE per step by `tcn/agent.py:Agent.act`, once
           to choose the action and once to commit the executed action to state
  decode   `outputs['policy'].flat()` + `bind_action` -> a typed Action
"""
import sys, json, time
ROOT = '/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + '/research/computer-capability')
sys.path.insert(0, ROOT + '/research/inference-cost')

from harness import Counter, timed, dump, rss_mb, size_report
from tcn.agent import Agent
from tcn.operators import Registry
from tcn.runtime import save_program
import program as P
import task as T
import closed_loop as CL

BASE = ROOT + '/research/computer-capability/out/'


def main():
    found = json.loads(open(BASE + 'search.json').read())
    registry = Registry()
    transform = P.transform_program(registry)
    policy = P.policy_program(registry)
    frozen = P.agent_program(registry, transform, policy,
                             found['transform']['enumeration']['selections'],
                             found['policy']['enumeration']['selections'])
    config = CL.configuration()
    report = {'artifact': 'computer-capability frozen agent (successor task)',
              'size': size_report(frozen, registry),
              'search_space': {'transform': found['transform']['space_size'],
                               'policy': found['policy']['space_size'],
                               'transform_conforming': found['transform']['enumeration']['conforming'],
                               'policy_conforming': found['policy']['enumeration']['conforming']}}

    rows = []
    for i, (name, digit) in enumerate(T.TEST_DOCUMENTS[:4]):
        host = T.host(name, digit, 3, objective_path=T.TASK_PATH, seed=0, index=3000 + i, split='test')
        agent = Agent(frozen, registry, config, 0)
        agent.reset()
        step_rows = []
        for tick in range(3):
            if host.records[-1].done:
                break
            view = host.view('agent_0')
            with Counter(registry) as counter:
                act_ms, _, action = timed(lambda: agent.act(view, deterministic=True), repeats=3)
            ops = counter.n // 3
            from dataclasses import replace
            t = time.perf_counter_ns()
            host.step((replace(action, actor='agent_0'),), config.dt)
            acquire_ms = (time.perf_counter_ns() - t) / 1e6
            step_rows.append({'tick': tick, 'verb': action.verb,
                              'agent_act_ms': act_ms, 'operator_applications': ops,
                              'us_per_operator': act_ms * 1000 / max(1, ops),
                              'kernel_step_ms': acquire_ms})
            print(json.dumps(step_rows[-1]))
        rows.append({'document': T.document(name, digit), 'steps': step_rows,
                     'return': CL.episode_return(host)})
    report['episodes'] = rows
    agent_ms = [s['agent_act_ms'] for r in rows for s in r['steps']]
    kernel_ms = [s['kernel_step_ms'] for r in rows for s in r['steps']]
    ops = [s['operator_applications'] for r in rows for s in r['steps']]
    import statistics
    report['per_step'] = {'agent_act_ms_median': statistics.median(agent_ms),
                          'kernel_step_ms_median': statistics.median(kernel_ms),
                          'operator_applications_median': statistics.median(ops),
                          'program_share_of_step': statistics.median(agent_ms) /
                                                   (statistics.median(agent_ms) + statistics.median(kernel_ms))}
    report['peak_rss_mb'] = rss_mb()
    save_program(frozen, BASE + 'agent_program.json', registry)
    print(json.dumps(report['per_step'], indent=1))
    dump('computer', report)


if __name__ == '__main__':
    main()
