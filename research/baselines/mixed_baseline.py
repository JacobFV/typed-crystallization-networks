"""Matched-information MLP baseline for examples/mixed.py.

TASK (from examples/mixed.py): inputs a:bool, b:bool, x:float.
Target answer = sin(xor(a,b) + x).  Fitting set = 4 (a,b) combinations x
4 values of x in {-.7,-.2,.3,.8} = 16 examples.  The repo test
tests/test_graph_learning.py::test_learning_and_progressive_crystallization
additionally checks a=True,b=False at x in {.13,-.43} against sin(1+x).

WHAT THE BASELINE SEES:  exactly [float(a), float(b), x]  -- the same three
typed inputs the TCN Program declares, in the same 16 examples, with no extra
data, no extra x values, and no structural hint.

Two supervision modes, because the TCN is *not* trained on `answer` alone:
  * `answer`  -- the MLP is fit to the answer only (weaker supervision than TCN).
  * `signals` -- the MLP has four heads fit to the same four declared signals
                 the TCN's Signal list supervises: logic (BCE on xor), conversion
                 (encode), algebra (z+x), answer (sin(z+x)).  This is the
                 information-matched setting.

Budget: the TCN run is 300 fit steps + up to 24 crystallizer rounds x 10 retrain
steps = <=540 full-batch Adam steps.  The headline MLP gets 540.  A 5000-step
run is also reported and clearly labelled as an over-budget courtesy.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from common import bench, description_bits, param_count, set_threads

TRAIN_X = [-0.7, -0.2, 0.3, 0.8]
AB = [(False, False), (False, True), (True, False), (True, True)]


def truth(a, b, x):
    z = float(a != b)
    u = z + x
    return {"logic": float(a != b), "conversion": z, "algebra": u, "answer": math.sin(u)}


def dataset(xs):
    rows = [(a, b, x) for a, b in AB for x in xs]
    inp = torch.tensor([[float(a), float(b), x] for a, b, x in rows], dtype=torch.float32)
    tgt = {k: torch.tensor([[truth(a, b, x)[k]] for a, b, x in rows], dtype=torch.float32)
           for k in ("logic", "conversion", "algebra", "answer")}
    return inp, tgt, rows


class MLP(torch.nn.Module):
    def __init__(self, width, depth, heads):
        super().__init__()
        layers, d = [], 3
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(d, heads)
        self.heads = heads

    def forward(self, x):
        return self.head(self.body(x))


def train(width, depth, lr, steps, mode, seed):
    torch.manual_seed(seed)
    heads = 4 if mode == "signals" else 1
    m = MLP(width, depth, heads)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    inp, tgt, _ = dataset(TRAIN_X)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        out = m(inp)
        if mode == "signals":
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0:1], tgt["logic"])
                    + (out[:, 1:2] - tgt["conversion"]).square().mean()
                    + (out[:, 2:3] - tgt["algebra"]).square().mean()
                    + (out[:, 3:4] - tgt["answer"]).square().mean())
        else:
            loss = (out - tgt["answer"]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
    wall = time.perf_counter() - t0
    return m, float(loss.detach()), wall


def answer_of(m, inp):
    out = m(inp)
    return out[:, -1:] if m.heads > 1 else out


@torch.no_grad()
def evaluate(m):
    res = {}
    inp, tgt, _ = dataset(TRAIN_X)
    res["train_answer_mse"] = float((answer_of(m, inp) - tgt["answer"]).square().mean())
    res["train_answer_maxabs"] = float((answer_of(m, inp) - tgt["answer"]).abs().max())
    # The two exact points the repo's own test checks.
    pts = torch.tensor([[1.0, 0.0, 0.13], [1.0, 0.0, -0.43]])
    want = torch.tensor([[math.sin(1 + 0.13)], [math.sin(1 - 0.43)]])
    err = (answer_of(m, pts) - want).abs()
    res["repo_test_points_maxabs"] = float(err.max())
    res["repo_test_points_pass_1e-6"] = bool(float(err.max()) <= 1e-6)
    res["repo_test_points_pass_1e-2"] = bool(float(err.max()) <= 1e-2)
    # Dense interpolation grid inside the training range of x.
    grid = [round(-0.7 + 0.01 * i, 3) for i in range(151)]
    grid = [v for v in grid if v not in TRAIN_X]
    inp, tgt, _ = dataset(grid)
    d = (answer_of(m, inp) - tgt["answer"]).abs()
    res["interp_maxabs"] = float(d.max())
    res["interp_rmse"] = float(d.square().mean().sqrt())
    # Extrapolation outside the fitted x range.
    ex = [round(-2.0 + 0.05 * i, 3) for i in range(81)]
    ex = [v for v in ex if v < -0.7 or v > 0.8]
    inp, tgt, _ = dataset(ex)
    d = (answer_of(m, inp) - tgt["answer"]).abs()
    res["extrap_maxabs"] = float(d.max())
    res["extrap_rmse"] = float(d.square().mean().sqrt())
    return res


def oracle_benchmark():
    """Hand-written exact reference: the four operations, in plain Python."""
    rows = [(a, b, x) for a, b in AB for x in TRAIN_X]

    def call(r):
        a, b, x = r
        return math.sin(float(a != b) + x)

    out = bench(call, rows, scope="hand-written Python oracle, batch one")
    out["params"] = 0
    return out


def numpy_export_benchmark(m):
    import numpy as np
    ws = [(l.weight.detach().numpy().astype(np.float32), l.bias.detach().numpy().astype(np.float32))
          for l in list(m.body) + [m.head] if isinstance(l, torch.nn.Linear)]
    inputs = [np.array([float(a), float(b), x], dtype=np.float32) for a, b in AB for x in TRAIN_X]

    def call(v):
        h = v
        for i, (w, b) in enumerate(ws):
            h = w @ h + b
            if i < len(ws) - 1:
                h = np.tanh(h)
        return h

    return bench(call, inputs, scope="numpy-exported MLP, batch one")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/baselines/out/mixed_baseline.json")
    args = ap.parse_args()
    set_threads()

    STEPS = 540  # matched to the TCN synthesis budget
    grid = [(w, d, lr) for w in (8, 16, 32) for d in (2, 3) for lr in (0.01, 0.03)]
    report = {"budget_steps": STEPS, "tuning_grid": [{"width": w, "depth": d, "lr": lr} for w, d, lr in grid],
              "sweeps": {}, "selected": {}}

    for mode in ("answer", "signals"):
        rows = []
        for w, d, lr in grid:
            m, loss, wall = train(w, d, lr, STEPS, mode, seed=0)
            ev = evaluate(m)
            rows.append({"width": w, "depth": d, "lr": lr, "final_loss": loss,
                         "wall_seconds": wall, **ev, "params": param_count(m)})
        report["sweeps"][mode] = rows
        # Model selection on the *fitting* objective only -- no held-out peeking.
        best = min(rows, key=lambda r: r["train_answer_mse"])
        m, loss, wall = train(best["width"], best["depth"], best["lr"], STEPS, mode, seed=0)
        ev = evaluate(m)
        with torch.no_grad():
            inputs = [torch.tensor([[float(a), float(b), x]], dtype=torch.float32)
                      for a, b in AB for x in TRAIN_X]
            lat = bench(lambda t: m(t), inputs, scope="torch MLP forward, batch one, no_grad")
            lat_np = numpy_export_benchmark(m)
        report["selected"][mode] = {
            "config": {"width": best["width"], "depth": best["depth"], "lr": best["lr"], "steps": STEPS},
            "final_loss": loss, "wall_seconds": wall, **ev,
            **description_bits(m), "latency_torch": lat, "latency_numpy": lat_np,
            "selection_rule": "lowest training-set answer MSE across the 12-point grid",
        }
        # Over-budget courtesy run at the same configuration.
        m2, loss2, wall2 = train(best["width"], best["depth"], best["lr"], 5000, mode, seed=0)
        report["selected"][mode]["over_budget_5000_steps"] = {
            "final_loss": loss2, "wall_seconds": wall2, **evaluate(m2)}

    report["oracle"] = oracle_benchmark()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["selected"], indent=2)[:4000])


if __name__ == "__main__":
    main()
