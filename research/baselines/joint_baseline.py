"""Matched-information baselines for examples/joint.py.

ENVIRONMENT: the real repo environment, `generators/logic` via `tcn.generation.Host`,
configuration {'depth':1,'table':6,'fixed_inputs':True,'horizon':4}, objectives
alternating {'invert':False}/{'invert':True} by episode index -- byte-for-byte the
same `TrainConfig` fields `examples/joint.py` builds.  Episodes 0..159 on
split='train' seed 0 for training; episodes 10000..10015 on split='test' for the
deterministic evaluation, exactly as `tcn/cli.py:joint` does.

WHAT THE BASELINE SEES per step (identical to the TCN Program's declared inputs):
    bits          4 floats   observations['bits'].flat()   (4 boolean task bits)
    goal          1 float    observations['goal'].flat()   (the objective bit)
    prev action   2 floats   one-hot of the previous action index (TCN declares
                             an 'action' input of type tuple[F,F]; its nodes do
                             not consume it, and neither does the reward -- it is
                             given to the MLP anyway so the input set is a
                             superset-free match)
    dt            1 float    the constant 1.0 the TCN 'dt' input receives
The generator ALSO publishes observations['program'] (the gate table).  Neither
the TCN nor these baselines read it.  latent_states['values'] and probes
{'target','gate'} are privileged; they never enter the observation vector.

SUPERVISION MODES
  reinforce  policy + value + entropy only.  Strictly less information than the
             TCN receives.
  aux        the same losses PLUS the same privileged auxiliary signals the TCN
             config declares: prediction of probes['target'] and probes['gate']
             at horizon 1 (weight 1.0) and a probe head on probes at horizon 0
             (weight 1.0).  This is the information-matched setting.

TRIVIAL REFERENCES: always-True, always-False, uniform random, and the exact
oracle answer computed from the observation vector alone
(bits[0] XOR bits[1]) XOR goal -- which is realizable from what the agent sees.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tcn.generation import Host, Action  # noqa: E402
from tcn.types import BOOL, Value  # noqa: E402
from common import bench, description_bits, param_count, set_threads  # noqa: E402

GEN = "logic"
CFG = {"depth": 1, "table": 6, "fixed_inputs": True, "horizon": 4}
OBJECTIVES = ({"invert": False}, {"invert": True})
ACTIONS = (Action("answer", arguments=(("value", Value.of(BOOL, False)),)),
           Action("answer", arguments=(("value", Value.of(BOOL, True)),)))
HORIZON, DT, DISCOUNT, SEED = 4, 1.0, 0.95, 0
POLICY_W, VALUE_W, ENTROPY_W, PRED_W, PROBE_W = 1.0, 0.5, 0.01, 1.0, 1.0
OBS_DIM = 4 + 1 + 2 + 1


def make_host(index, split, objective):
    return Host.create(GEN, seed=SEED, index=index, split=split,
                       configuration=CFG | {"horizon": HORIZON}, objective=objective)


def obs_vector(view, prev_action):
    bits = view.observations["bits"].flat()
    goal = view.observations["goal"].flat()
    onehot = [1.0 if i == prev_action else 0.0 for i in range(len(ACTIONS))]
    return torch.tensor(bits + goal + onehot + [DT], dtype=torch.float32)


def probe_truth(record):
    return torch.tensor([record.probes["target"].flat()[0], record.probes["gate"].flat()[0]],
                        dtype=torch.float32)


class PolicyValue(torch.nn.Module):
    def __init__(self, width, depth, aux):
        super().__init__()
        layers, d = [], OBS_DIM
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        self.body = torch.nn.Sequential(*layers)
        self.policy = torch.nn.Linear(d, len(ACTIONS))
        self.value = torch.nn.Linear(d, 1)
        self.aux = aux
        if aux:
            self.prediction = torch.nn.Linear(d, 2)   # probes['target'], probes['gate'] at t+1
            self.probe = torch.nn.Linear(d, 2)        # the same two at t

    def forward(self, x):
        h = self.body(x)
        out = {"policy": self.policy(h), "value": self.value(h).reshape(())}
        if self.aux:
            out["prediction"] = self.prediction(h)
            out["probe"] = self.probe(h)
        return out


def run_episode(model, index, split, train, mode):
    host = make_host(index, split, OBJECTIVES[index % len(OBJECTIVES)])
    prev, rows = 0, []
    for _ in range(HORIZON):
        view = host.view()
        out = model(obs_vector(view, prev))
        logits = out["policy"]
        allowed = torch.tensor([a.verb in view.available_actions for a in ACTIONS], dtype=torch.bool)
        masked = logits.masked_fill(~allowed, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked)
        choice = dist.sample() if train else masked.argmax()
        i = int(choice)
        # TCN re-runs the model with the chosen action bound before predicting.
        pred_out = model(obs_vector(view, i)) if mode == "aux" else None
        record = host.step((ACTIONS[i],), DT)
        rows.append({"logp": dist.log_prob(choice), "entropy": dist.entropy(),
                     "value": out["value"], "reward": float(sum(v.decoded for v in record.reward_components.values())),
                     "prediction": pred_out["prediction"] if pred_out else None,
                     "probe": pred_out["probe"] if pred_out else None})
        prev = i
        if record.done:
            break
    returns, g = [], 0.0
    for row in reversed(rows):
        g = row["reward"] + DISCOUNT * g
        returns.insert(0, g)
    actor = torch.stack([-r["logp"] * (ret - r["value"].detach()) for r, ret in zip(rows, returns)]).mean()
    value = torch.stack([(r["value"] - ret).square() for r, ret in zip(rows, returns)]).mean()
    entropy = torch.stack([r["entropy"] for r in rows]).mean()
    loss = POLICY_W * actor + VALUE_W * value - ENTROPY_W * entropy
    prediction_loss = torch.tensor(0.0)
    if mode == "aux":
        preds, probes = [], []
        for t, row in enumerate(rows):
            if t + 1 < len(host.records):
                preds.append((row["prediction"] - probe_truth(host.records[t + 1])).square().mean())
            probes.append((row["probe"] - probe_truth(host.records[t])).square().mean())
        prediction_loss = torch.stack(preds).mean()
        loss = loss + PRED_W * prediction_loss + PROBE_W * torch.stack(probes).mean()
    return loss, {"return": sum(r["reward"] for r in rows),
                  "prediction_loss": float(prediction_loss.detach())}


def train_agent(width, depth, lr, mode, episodes, seed):
    torch.manual_seed(seed)
    m = PolicyValue(width, depth, aux=(mode == "aux"))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    history = []
    t0 = time.perf_counter()
    for i in range(episodes):
        loss, metrics = run_episode(m, i, "train", True, mode)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        history.append(metrics)
    return m, history, time.perf_counter() - t0


@torch.no_grad()
def evaluate(model, mode, start=10000, n=16):
    returns = [run_episode(model, start + i, "test", False, mode)[1]["return"] for i in range(n)]
    return sum(returns) / len(returns), returns


# ---------------------------------------------------------------- trivial refs
def fixed_policy(fn, start=10000, n=16):
    """fn(view, prev) -> action index."""
    returns = []
    for i in range(n):
        host = make_host(start + i, "test", OBJECTIVES[i % len(OBJECTIVES)])
        prev, total = 0, 0.0
        for _ in range(HORIZON):
            view = host.view()
            j = fn(view, prev)
            record = host.step((ACTIONS[j],), DT)
            total += sum(v.decoded for v in record.reward_components.values())
            prev = j
            if record.done:
                break
        returns.append(total)
    return sum(returns) / len(returns), returns


def oracle_action(view, prev):
    bits = view.observations["bits"].flat()
    goal = view.observations["goal"].flat()[0]
    return int((bool(bits[0]) != bool(bits[1])) != bool(goal))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/baselines/out/joint_baseline.json")
    ap.add_argument("--episodes", type=int, default=160)
    args = ap.parse_args()
    set_threads()

    report = {"episodes": args.episodes, "observation_dim": OBS_DIM, "sweeps": {}, "selected": {},
              "trivial": {}}

    # Trivial references, on both the 16-episode repo protocol and a 256-episode set.
    for n, tag in ((16, "n16"), (256, "n256")):
        rng = torch.Generator().manual_seed(0)
        report["trivial"][tag] = {
            "always_false": fixed_policy(lambda v, p: 0, n=n)[0],
            "always_true": fixed_policy(lambda v, p: 1, n=n)[0],
            "uniform_random": fixed_policy(lambda v, p: int(torch.randint(0, 2, (1,), generator=rng)), n=n)[0],
            "exact_oracle": fixed_policy(oracle_action, n=n)[0],
        }
    report["trivial"]["n16_returns_always_true"] = fixed_policy(lambda v, p: 1, n=16)[1]

    grid = [(w, d, lr) for w in (8, 32) for d in (2, 3) for lr in (0.01, 0.04, 0.1)]
    report["tuning_grid"] = [{"width": w, "depth": d, "lr": lr} for w, d, lr in grid]
    for mode in ("reinforce", "aux"):
        rows = []
        for w, d, lr in grid:
            m, hist, wall = train_agent(w, d, lr, mode, args.episodes, seed=0)
            mean16, _ = evaluate(m, mode, 10000, 16)
            mean256, _ = evaluate(m, mode, 10000, 256)
            rows.append({"width": w, "depth": d, "lr": lr, "wall_seconds": wall,
                         "train_return_last32": sum(x["return"] for x in hist[-32:]) / 32,
                         "eval_mean_return_n16": mean16, "eval_mean_return_n256": mean256,
                         "params": param_count(m),
                         "final_prediction_loss": sum(x["prediction_loss"] for x in hist[-8:]) / 8})
        report["sweeps"][mode] = rows
        # Selection on TRAINING return only (no test-set peeking).
        best = max(rows, key=lambda r: r["train_return_last32"])
        # Seed robustness at the selected configuration.
        seeds = []
        for s in range(5):
            m, hist, wall = train_agent(best["width"], best["depth"], best["lr"], mode, args.episodes, seed=s)
            mean16, ret16 = evaluate(m, mode, 10000, 16)
            mean256, _ = evaluate(m, mode, 10000, 256)
            seeds.append({"seed": s, "wall_seconds": wall,
                          "train_return_last32": sum(x["return"] for x in hist[-32:]) / 32,
                          "eval_mean_return_n16": mean16, "eval_mean_return_n256": mean256,
                          "final_prediction_loss": sum(x["prediction_loss"] for x in hist[-8:]) / 8,
                          "initial_prediction_loss": sum(x["prediction_loss"] for x in hist[:8]) / 8})
            if s == 0:
                m0 = m
        with torch.no_grad():
            host = make_host(10000, "test", OBJECTIVES[0])
            vecs = [obs_vector(host.view(), 0)]
            lat = bench(lambda t: m0(t), vecs, scope="torch policy+value forward, batch one, no_grad")
        report["selected"][mode] = {
            "config": {"width": best["width"], "depth": best["depth"], "lr": best["lr"],
                       "episodes": args.episodes},
            "selection_rule": "highest mean training return over the last 32 episodes",
            "seeds": seeds,
            "eval_mean_return_n16_mean": sum(s["eval_mean_return_n16"] for s in seeds) / len(seeds),
            "eval_mean_return_n16_min": min(s["eval_mean_return_n16"] for s in seeds),
            "eval_mean_return_n16_max": max(s["eval_mean_return_n16"] for s in seeds),
            "eval_mean_return_n256_mean": sum(s["eval_mean_return_n256"] for s in seeds) / len(seeds),
            "wall_seconds_mean": sum(s["wall_seconds"] for s in seeds) / len(seeds),
            **description_bits(m0), "latency": lat}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"trivial": report["trivial"], "selected": {k: {kk: vv for kk, vv in v.items() if kk != "seeds"}
                                                                for k, v in report["selected"].items()}}, indent=2))


if __name__ == "__main__":
    main()
