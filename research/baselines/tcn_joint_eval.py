"""Evaluate the frozen TCN joint agent on the SAME held-out episode sets the baselines use.

tcn/cli.py reports two numbers: the soft model on episodes 10000..10015 and the
frozen program on 30000..30015 -- both n=16.  Sixteen episodes of horizon four
are only sixteen independent decisions (the environment state is constant within
an episode), so this script also reports n=256 on the same index family.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcn.runtime import load_program  # noqa: E402
from tcn.training import TrainConfig  # noqa: E402
from tcn.agent import Agent  # noqa: E402
import joint_baseline as JB  # noqa: E402


def run(program, registry, config, start, n):
    returns = []
    for i in range(n):
        host = JB.make_host(start + i, "test", JB.OBJECTIVES[i % len(JB.OBJECTIVES)])
        Agent(program, registry, config).rollout(host, deterministic=True)
        returns.append(sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records))
    return sum(returns) / len(returns), returns


def main():
    out = Path("research/baselines/out/tcn_joint")
    p, r = load_program(out / "program.json")
    cfg = TrainConfig.from_dict(json.loads((out / "agent.json").read_text()))
    res = {"description_bits": p.description_bits(r),
           "execution_cost": p.execution_cost(r),
           "trainable_parameters_after_freeze": 0}
    for start in (10000, 30000):
        for n in (16, 256):
            mean, _ = run(p, r, cfg, start, n)
            res[f"frozen_mean_return_start{start}_n{n}"] = mean
    print(json.dumps(res, indent=2))
    Path("research/baselines/out/tcn_joint_eval.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
