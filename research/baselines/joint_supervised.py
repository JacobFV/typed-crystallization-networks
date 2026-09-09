"""Information-matched supervised baselines for the joint logic task.

WHY THIS FILE EXISTS.  Inspecting the frozen program produced by examples/joint.py
shows the policy decoder is hand-wired correct at initialisation:
    z = encode(goal_relation);  logit0 = -2*z + 1;  logit1 = 2*z - 1
so argmax(logits) == goal_relation for any z in {0,1}, and the trained constants
only rescale it (-2 -> -2.86, 2 -> 2.86, 1 -> 1.05, -1 -> -1.05).  The only thing
the run learns is two 16-way truth-table selections (both settle on truth_6 =
XOR), i.e. 8 bits of content, driven by the dense privileged probe/prediction
loss on probes['target'] and probes['gate'].  The REINFORCE term is not what
produces 4/4.

So the matched-information baseline is not "REINFORCE from scratch"; it is
"same observations, same 160 episodes, same privileged probe supervision,
same one-optimizer-step-per-episode schedule, then act greedily".  Three of
those are implemented here plus a linear control and a 32-entry lookup table.

WHAT EACH BASELINE SEES: the same 8-float observation vector as joint_baseline.py
(4 task bits, goal bit, previous-action one-hot, dt) and the same privileged
probes['target'] used as a per-step BCE label -- exactly the channel the TCN's
probe_weight=1 loss consumes.  Nothing else.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bench, description_bits, param_count, set_threads  # noqa: E402
import joint_baseline as JB  # noqa: E402


class Net(torch.nn.Module):
    """Predicts probes['target'] as one logit; the action is argmax over {False,True}."""

    def __init__(self, width, depth):
        super().__init__()
        layers, d = [], JB.OBS_DIM
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.body(x))


def collect(index, act):
    """Roll one training episode; return (obs, label) pairs and the achieved return."""
    host = JB.make_host(index, "train", JB.OBJECTIVES[index % len(JB.OBJECTIVES)])
    prev, xs, ys, total = 0, [], [], 0.0
    for _ in range(JB.HORIZON):
        view = host.view()
        v = JB.obs_vector(view, prev)
        xs.append(v)
        ys.append(host.records[-1].probes["target"].flat()[0])   # privileged probe label
        j = act(v)
        record = host.step((JB.ACTIONS[j],), JB.DT)
        total += sum(w.decoded for w in record.reward_components.values())
        prev = j
        if record.done:
            break
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32).unsqueeze(1), total


def train_supervised(width, depth, lr, episodes, seed, inner=1, replay=False):
    torch.manual_seed(seed)
    m = Net(width, depth)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    buf_x, buf_y, returns = [], [], []
    t0 = time.perf_counter()
    for i in range(episodes):
        with torch.no_grad():
            def act(v):
                return int(m(v).item() > 0)
        xs, ys, total = collect(i, act)
        returns.append(total)
        buf_x.append(xs)
        buf_y.append(ys)
        X = torch.cat(buf_x) if replay else xs
        Y = torch.cat(buf_y) if replay else ys
        for _ in range(inner):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(X), Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
    return m, returns, time.perf_counter() - t0, float(loss.detach())


@torch.no_grad()
def evaluate(m, start=10000, n=16):
    def act(v):
        return int(m(v).item() > 0)
    return JB.fixed_policy(lambda view, prev: act(JB.obs_vector(view, prev)), start=start, n=n)


def train_table(episodes, seed=0):
    """32-entry lookup keyed on the five observed bits (4 task bits + goal).

    No knowledge of which bits matter; counts from the same privileged label.
    Description size: 32 bits of table + the key layout."""
    counts = {}
    m = {"t": counts}

    def key(v):
        return sum(int(v[i] > 0.5) << i for i in range(5))

    def act(v):
        c = counts.get(key(v))
        return 0 if c is None else int(c[1] > c[0])
    returns = []
    t0 = time.perf_counter()
    for i in range(episodes):
        xs, ys, total = collect(i, act)
        returns.append(total)
        for v, y in zip(xs, ys):
            c = counts.setdefault(key(v), [0, 0])
            c[int(y.item() > 0.5)] += 1
    return m, returns, time.perf_counter() - t0


def eval_table(m, start=10000, n=16):
    counts = m["t"]

    def act(view, prev):
        v = JB.obs_vector(view, prev)
        k = sum(int(v[i] > 0.5) << i for i in range(5))
        c = counts.get(k)
        return 0 if c is None else int(c[1] > c[0])
    return JB.fixed_policy(act, start=start, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/baselines/out/joint_supervised.json")
    args = ap.parse_args()
    set_threads()
    report = {"sweeps": {}, "selected": {}, "table": {}, "budget_curve": {}}

    grid = [(w, d, lr) for w in (8, 32) for d in (0, 2, 3) for lr in (0.01, 0.04, 0.1)]
    report["tuning_grid"] = [{"width": w, "depth": d, "lr": lr} for w, d, lr in grid]

    for tag, inner, replay in (("onestep", 1, False), ("replay8", 8, True)):
        rows = []
        for w, d, lr in grid:
            m, ret, wall, loss = train_supervised(w, d, lr, 160, 0, inner, replay)
            e16, _ = evaluate(m, n=16)
            e256, _ = evaluate(m, n=256)
            rows.append({"width": w, "depth": d, "lr": lr, "wall_seconds": wall,
                         "final_bce": loss, "train_return_last32": sum(ret[-32:]) / 32,
                         "eval_mean_return_n16": e16, "eval_mean_return_n256": e256,
                         "params": param_count(m)})
        report["sweeps"][tag] = rows
        best = min(rows, key=lambda r: r["final_bce"])   # selected on the training objective
        seeds = []
        for s in range(5):
            m, ret, wall, loss = train_supervised(best["width"], best["depth"], best["lr"], 160, s, inner, replay)
            e16, _ = evaluate(m, n=16)
            e256, _ = evaluate(m, n=256)
            seeds.append({"seed": s, "wall_seconds": wall, "final_bce": loss,
                          "train_return_last32": sum(ret[-32:]) / 32,
                          "eval_mean_return_n16": e16, "eval_mean_return_n256": e256})
            if s == 0:
                m0 = m
        with torch.no_grad():
            host = JB.make_host(10000, "test", JB.OBJECTIVES[0])
            vec = [JB.obs_vector(host.view(), 0)]
            lat = bench(lambda t: m0(t), vec, scope="torch MLP forward, batch one, no_grad")
        report["selected"][tag] = {
            "config": {"width": best["width"], "depth": best["depth"], "lr": best["lr"],
                       "episodes": 160, "inner_steps": inner, "replay": replay},
            "selection_rule": "lowest final training BCE across the 18-point grid",
            "seeds": seeds,
            "eval_mean_return_n16_mean": sum(x["eval_mean_return_n16"] for x in seeds) / len(seeds),
            "eval_mean_return_n16_min": min(x["eval_mean_return_n16"] for x in seeds),
            "eval_mean_return_n256_mean": sum(x["eval_mean_return_n256"] for x in seeds) / len(seeds),
            "wall_seconds_mean": sum(x["wall_seconds"] for x in seeds) / len(seeds),
            **description_bits(m0), "latency": lat}

    # lookup table
    m, ret, wall = train_table(160)
    report["table"] = {"episodes": 160, "wall_seconds": wall,
                       "train_return_last32": sum(ret[-32:]) / 32,
                       "eval_mean_return_n16": eval_table(m, n=16)[0],
                       "eval_mean_return_n256": eval_table(m, n=256)[0],
                       "entries_seen": len(m["t"]),
                       "description_bits": 32,
                       "note": "one bit per key over 5 observed bits; unseen keys answer False"}

    # How many episodes does the REINFORCE-only MLP need?  (over-budget probe)
    for ep in (160, 640, 2560):
        best = {"width": 32, "depth": 3, "lr": 0.04}
        res = []
        for s in range(3):
            mm, hist, wall = JB.train_agent(best["width"], best["depth"], best["lr"], "aux", ep, s)
            res.append({"seed": s, "wall_seconds": wall,
                        "eval_mean_return_n16": JB.evaluate(mm, "aux", 10000, 16)[0],
                        "eval_mean_return_n256": JB.evaluate(mm, "aux", 10000, 256)[0]})
        report["budget_curve"][str(ep)] = res

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sweeps"}, indent=2))


if __name__ == "__main__":
    main()
