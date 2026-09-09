"""Strict budget parity runs.

Instrumenting the repo's own entry points gives the TCN side's real budget:
  mixed  (tcn.cli:mixed, 300 fit steps)  -> 340 full-batch Adam steps, 16 examples
  joint  (tcn.cli:joint, 160 episodes)   -> 161 train + 170 validation episodes
                                            = 331 environment episodes, 194 Adam steps
(the 170 validation episodes are the crystallizer's loss closure, which rolls two
`split='validation'` episodes on every call; tcn/cli.py does not report them).

This script re-runs the MLP baselines under exactly those two budgets.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import description_bits, set_threads  # noqa: E402
import mixed_baseline as MB  # noqa: E402
import joint_baseline as JB  # noqa: E402
import joint_supervised as JS  # noqa: E402

MIXED_STEPS = 340
JOINT_EPISODES = 331
JOINT_STEPS = 194


def joint_matched(width, depth, lr, seed, episodes=JOINT_EPISODES, opt_steps=JOINT_STEPS):
    """Same episodes AND same optimizer steps as the instrumented TCN run.

    Steps are spread evenly over the episode stream and taken on the full replay
    buffer of everything collected so far -- no extra environment interaction.
    """
    torch.manual_seed(seed)
    m = JS.Net(width, depth)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    bx, by, taken, returns = [], [], 0, []
    t0 = time.perf_counter()
    for i in range(episodes):
        with torch.no_grad():
            def act(v):
                return int(m(v).item() > 0)
        xs, ys, total = JS.collect(i, act)
        returns.append(total)
        bx.append(xs)
        by.append(ys)
        want = ((i + 1) * opt_steps) // episodes
        while taken < want:
            X, Y = torch.cat(bx), torch.cat(by)
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(X), Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            taken += 1
    return m, returns, time.perf_counter() - t0, taken


def main():
    set_threads()
    out = {}

    # -------- mixed at exactly 340 full-batch steps --------
    cfg = json.loads(Path("research/baselines/out/mixed_baseline.json").read_text())["selected"]
    out["mixed"] = {}
    for mode in ("answer", "signals"):
        c = cfg[mode]["config"]
        rows = []
        for s in range(3):
            m, loss, wall = MB.train(c["width"], c["depth"], c["lr"], MIXED_STEPS, mode, seed=s)
            rows.append({"seed": s, "final_loss": loss, "wall_seconds": wall, **MB.evaluate(m)})
            if s == 0:
                m0 = m
        out["mixed"][mode] = {"config": dict(c, steps=MIXED_STEPS), "seeds": rows,
                              **description_bits(m0)}

    # -------- joint at exactly 331 episodes and 194 optimizer steps --------
    out["joint_matched_episodes_and_steps"] = {}
    for width, depth, lr in ((32, 2, 0.04), (8, 2, 0.04), (32, 2, 0.1), (8, 2, 0.1), (32, 3, 0.04)):
        rows = []
        for s in range(3):
            m, ret, wall, taken = joint_matched(width, depth, lr, s)
            rows.append({"seed": s, "optimizer_steps": taken, "wall_seconds": wall,
                         "train_return_last32": sum(ret[-32:]) / 32,
                         "eval_mean_return_n16": JS.evaluate(m, n=16)[0],
                         "eval_mean_return_n256": JS.evaluate(m, n=256)[0]})
        out["joint_matched_episodes_and_steps"][f"w{width}_d{depth}_lr{lr}"] = rows

    # -------- lookup table at the same 331 episodes --------
    m, ret, wall = JS.train_table(JOINT_EPISODES)
    out["joint_table_331_episodes"] = {"wall_seconds": wall, "entries_seen": len(m["t"]),
                                       "eval_mean_return_n16": JS.eval_table(m, n=16)[0],
                                       "eval_mean_return_n256": JS.eval_table(m, n=256)[0],
                                       "description_bits": 32}

    Path("research/baselines/out/matched_budget.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
