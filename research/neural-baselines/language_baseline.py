"""Matched neural baselines for FINDINGS §19 -- grammaticality from raw prompt bytes.

MATCHING CONDITIONS, stated so they can be checked rather than trusted:

* **Inputs.** The TCN program's only input is `observations['text']`, a
  `(length: u32, 128 x byte)` typed carrier. Every baseline here receives exactly
  those 129 numbers: the 128 raw bytes as token ids 0-255 and the length. No
  baseline ever sees `latent_states['construction']`, `probes['answer']`, the
  seed, the split, or the extracted `string` as an input.
* **Supervision.** The TCN path is *staged* on the privileged `construction`
  latent: stage A is supervised at 44 in-string positions with "is this byte an
  opening bracket", stage B on the yes/no answer. The `aux` arms below give the
  baseline the same privileged channel, as an auxiliary per-position head
  supervised on the running bracket depth derived from `construction` -- strictly
  more dense supervision than stage A had. The `plain` arms use the answer alone.
* **Splits.** Identical episode indices to `run_stage_b.py` / `final_eval.py`:
  train = the first 24 episodes of `dataset(900, seed0=0, split='train')` with
  length in {2,4,6}; validation = the next 120 of the same lengths; test = every
  episode of `dataset(1500, seed0=100000, split='test')` whose length is not in
  {2,4,6} -- 724 episodes at lengths 8-16, 0% string overlap with training.
  All of these are drawn with `hardening="none"`, the preserved pre-audit stream
  the §19 artifact was searched on; see `langdata.py` for why the current default
  stream cannot host these splits at all. The current default stream is reported
  as a *second* held-out set (400 episodes, lengths 10-22) that both methods face
  on identical terms.
* **Model selection.** Two protocols are reported separately: selection on
  training accuracy only (what the TCN search had -- it never looked at a
  validation set) and selection on the 120-episode seen-length validation split
  (a concession *to* the baseline). Neither ever touches the 724 test episodes.
* **Trivial reference.** The majority constant (0.548) and random (0.500) are
  printed beside every number.

Run: `.venv/bin/python research/neural-baselines/language_baseline.py`
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn

import common as harness
import langdata

torch.set_num_threads(1)
try:                                  # keep the host share honest: one thread only
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

CAP = 128
splits = langdata.splits


def featurise(episodes):
    """The 129 observable numbers, and nothing else."""
    tokens, lengths, labels, depths, mask = [], [], [], [], []
    for e in episodes:
        raw = e["text"].raw
        n = int(raw[0])
        byts = [int(b) for b in raw[1]]
        tokens.append(byts)
        lengths.append(n)
        labels.append(1.0 if e["label"] else 0.0)
        # privileged: running bracket depth of the string, aligned to its bytes in
        # the prompt. Derived from `construction`; used ONLY as a target.
        d, row, m = 0, [0.0] * CAP, [0.0] * CAP
        off = e["offset"]
        for i, ch in enumerate(e["string"]):
            d += 1 if ch == "(" else -1
            row[off + i] = float(d)
            m[off + i] = 1.0
        depths.append(row)
        mask.append(m)
    return {"tokens": torch.tensor(tokens, dtype=torch.long),
            "length": torch.tensor(lengths, dtype=torch.float32).unsqueeze(1),
            "label": torch.tensor(labels, dtype=torch.float32).unsqueeze(1),
            "depth": torch.tensor(depths, dtype=torch.float32),
            "depth_mask": torch.tensor(mask, dtype=torch.float32)}


# --- models ----------------------------------------------------------------
class ByteGRU(nn.Module):
    """A recurrent scan over the raw bytes. The one small architecture that can
    *express* a counter, so the one a competent practitioner reaches for here."""

    def __init__(self, embed=8, hidden=8, aux=False):
        super().__init__()
        self.embed = nn.Embedding(256, embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.aux = nn.Linear(hidden, 1) if aux else None

    def forward(self, tokens, length):
        h, _ = self.rnn(self.embed(tokens))
        idx = (length.squeeze(1).long() - 1).clamp(0, tokens.shape[1] - 1)
        last = h[torch.arange(h.shape[0]), idx]
        return self.head(last), (self.aux(h).squeeze(-1) if self.aux is not None else None)


class ByteCNN(nn.Module):
    """A 1-D convolutional reader with mean pooling -- a permutation-blind counter
    if it wants to be, and the standard small text classifier."""

    def __init__(self, embed=8, width=16, layers=2, aux=False):
        super().__init__()
        self.embed = nn.Embedding(256, embed)
        chans, mods = embed, []
        for _ in range(layers):
            mods += [nn.Conv1d(chans, width, 3, padding=1), nn.Tanh()]
            chans = width
        self.body = nn.Sequential(*mods)
        self.head = nn.Linear(chans + 1, 1)
        self.aux = nn.Conv1d(chans, 1, 1) if aux else None

    def forward(self, tokens, length):
        h = self.body(self.embed(tokens).transpose(1, 2))
        pooled = h.mean(dim=2)
        return self.head(torch.cat([pooled, length / CAP], dim=1)), \
            (self.aux(h).squeeze(1) if self.aux is not None else None)


class ByteMLP(nn.Module):
    """Positional MLP: embed each byte, flatten, two tanh layers."""

    def __init__(self, embed=4, hidden=32, aux=False):
        super().__init__()
        self.embed = nn.Embedding(256, embed)
        self.body = nn.Sequential(nn.Linear(CAP * embed + 1, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 1)
        self.aux = nn.Linear(hidden, CAP) if aux else None

    def forward(self, tokens, length):
        e = self.embed(tokens).reshape(tokens.shape[0], -1)
        h = self.body(torch.cat([e, length / CAP], dim=1))
        return self.head(h), (self.aux(h) if self.aux is not None else None)


class TinyTransformer(nn.Module):
    def __init__(self, d=16, heads=2, layers=1, aux=False):
        super().__init__()
        self.embed = nn.Embedding(256, d)
        self.pos = nn.Embedding(CAP, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True,
                                           dropout=0.0, activation="gelu")
        self.body = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d + 1, 1)
        self.aux = nn.Linear(d, 1) if aux else None

    def forward(self, tokens, length):
        p = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        h = self.body(self.embed(tokens) + self.pos(p))
        return self.head(torch.cat([h.mean(dim=1), length / CAP], dim=1)), \
            (self.aux(h).squeeze(-1) if self.aux is not None else None)


class CountingHint(nn.Module):
    """DECLARED STRUCTURAL HINT, not a matched baseline.

    Bag-of-bytes counts hand the model the aggregation the TCN had to discover
    (which byte, at which offset, accumulated how). It is reported separately and
    labelled, in the same spirit as `research/baselines/RESULTS.md` §6 refusing
    Fourier features on the mixed fixture -- except here it is included *because*
    the size of the gap it closes is the interesting number.
    """

    def __init__(self, aux=False):
        super().__init__()
        self.head = nn.Linear(257, 1)
        self.aux = None

    def forward(self, tokens, length):
        counts = torch.zeros(tokens.shape[0], 256)
        counts.scatter_add_(1, tokens, torch.ones_like(tokens, dtype=torch.float32))
        return self.head(torch.cat([counts, length / CAP], dim=1)), None


ZOO = {
    "gru_8": lambda aux: ByteGRU(8, 8, aux),
    "gru_16": lambda aux: ByteGRU(8, 16, aux),
    "gru_32": lambda aux: ByteGRU(16, 32, aux),
    "cnn_16x2": lambda aux: ByteCNN(8, 16, 2, aux),
    "cnn_32x3": lambda aux: ByteCNN(8, 32, 3, aux),
    "mlp_32": lambda aux: ByteMLP(4, 32, aux),
    "mlp_64": lambda aux: ByteMLP(8, 64, aux),
    "transformer_d16": lambda aux: TinyTransformer(16, 2, 1, aux),
    "transformer_d32": lambda aux: TinyTransformer(32, 4, 2, aux),
}


# --- training --------------------------------------------------------------
def train_one(name, aux, data, seed, steps=1500, lr=0.01, aux_weight=1.0, batch=64):
    """Full batch when the budget is 24 examples; minibatches of `batch` above that."""
    torch.manual_seed(seed)
    model = ZOO[name](aux)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    n = data["tokens"].shape[0]
    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    for _ in range(steps):
        idx = (torch.arange(n) if n <= batch
               else torch.randint(0, n, (batch,), generator=generator))
        opt.zero_grad()
        logit, aux_out = model(data["tokens"][idx], data["length"][idx])
        loss = bce(logit, data["label"][idx])
        if aux and aux_out is not None:
            m = data["depth_mask"][idx]
            loss = loss + aux_weight * (((aux_out - data["depth"][idx]) ** 2) * m).sum() \
                / m.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model, {"train_seconds": time.perf_counter() - started, "final_loss": float(loss),
                   "steps": steps, "lr": lr, "batch": min(batch, n),
                   "examples_consumed": steps * min(batch, n)}


@torch.no_grad()
def accuracy(model, data):
    logit, _ = model(data["tokens"], data["length"])
    pred = (logit > 0).float()
    return float((pred == data["label"]).float().mean())


# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--lrs", type=float, nargs="*", default=[0.01],
                    help="learning-rate grid; the arm reported is the best on TRAINING "
                         "accuracy, never on the held-out set")
    ap.add_argument("--over-seeds", type=int, default=2)
    ap.add_argument("--over-models", nargs="*",
                    default=["gru_16", "cnn_32x3", "mlp_32", "transformer_d16"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--tag", default="language")
    args = ap.parse_args()

    print("building episodes ...", flush=True)
    sp = splits()
    feats = {k: featurise(v) for k, v in sp.items()}
    ref = {k: harness.majority_reference([bool(e["label"]) for e in v]) for k, v in sp.items()}
    print(json.dumps({k: {"n": len(v)} for k, v in sp.items()}), flush=True)
    print(json.dumps(ref["test"]), flush=True)

    report = {"splits": {k: {"n": len(v), "lengths": sorted({e["length"] for e in v}),
                            "depths": sorted({e["depth"] for e in v}),
                            "distinct_strings": len({e["string"] for e in v})}
                         for k, v in sp.items()},
              "trivial_reference": ref, "arms": []}

    budgets = [("matched_24", "train", args.seeds, args.models or list(ZOO)),
               ("over_budget_461", "over_budget_train", args.over_seeds, args.over_models)]
    for budget_name, key, n_seeds, models in budgets:
        data = feats[key]
        for aux in (False, True):
            for name in models:
                rows = []
                for seed in range(n_seeds):
                  for lr in args.lrs:
                    model, info = train_one(name, aux, data, seed, steps=args.steps, lr=lr)
                    row = {"seed": seed,
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "train_acc": accuracy(model, data),
                           "val_acc": accuracy(model, feats["val"]),
                           "test_acc": accuracy(model, feats["test"]),
                           "current_stream_acc": accuracy(model, feats["current_stream_test"]),
                           **info}
                    rows.append(row)
                    print(f"  {budget_name:16s} aux={int(aux)} {name:18s} seed {seed}: "
                          f"train {row['train_acc']:.3f} val {row['val_acc']:.3f} "
                          f"test {row['test_acc']:.3f} ({row['parameters']} params, "
                          f"{row['train_seconds']:.1f}s)", flush=True)
                best_lr = max(args.lrs, key=lambda l: statistics.median(
                    r["train_acc"] for r in rows if r["lr"] == l))
                rows = [r for r in rows if r["lr"] == best_lr]
                report["arms"].append({
                    "budget": budget_name, "aux": aux, "model": name, "lr": best_lr,
                    "lr_grid": list(args.lrs),
                    "parameters": rows[0]["parameters"],
                    "train_acc_median": statistics.median(r["train_acc"] for r in rows),
                    "val_acc_median": statistics.median(r["val_acc"] for r in rows),
                    "test_acc_median": statistics.median(r["test_acc"] for r in rows),
                    "test_acc_max": max(r["test_acc"] for r in rows),
                    "test_acc_min": min(r["test_acc"] for r in rows),
                    "current_stream_acc_median": statistics.median(
                        r["current_stream_acc"] for r in rows),
                    "train_seconds_median": statistics.median(r["train_seconds"] for r in rows),
                    "rows": rows})
        harness.dump(args.tag, report)

    # the declared structural-hint arm, reported apart from the matched ones
    hint_rows = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = CountingHint()
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        bce = nn.BCEWithLogitsLoss()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            opt.zero_grad()
            logit, _ = model(feats["train"]["tokens"], feats["train"]["length"])
            loss = bce(logit, feats["train"]["label"])
            loss.backward()
            opt.step()
        hint_rows.append({"seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                          "train_acc": accuracy(model, feats["train"]),
                          "val_acc": accuracy(model, feats["val"]),
                          "test_acc": accuracy(model, feats["test"]),
                          "current_stream_acc": accuracy(model, feats["current_stream_test"]),
                          "train_seconds": time.perf_counter() - t0})
        print(f"  structural-hint counting seed {seed}: test {hint_rows[-1]['test_acc']:.3f}",
              flush=True)
    report["structural_hint_counting"] = {
        "declared": "bag-of-256-byte-counts + length -> linear. Hands the model the "
                    "aggregation stage A had to discover. NOT a matched baseline.",
        "parameters": hint_rows[0]["parameters"],
        "test_acc_median": statistics.median(r["test_acc"] for r in hint_rows),
        "current_stream_acc_median": statistics.median(
            r["current_stream_acc"] for r in hint_rows),
        "rows": hint_rows}

    harness.dump(args.tag, report)

    # --- best matched arm, selected two ways, then priced --------------------
    matched = [a for a in report["arms"] if a["budget"] == "matched_24"]
    by_train = max(matched, key=lambda a: (a["train_acc_median"], -a["parameters"]))
    by_val = max(matched, key=lambda a: (a["val_acc_median"], -a["parameters"]))
    report["selection"] = {
        "on_training_accuracy_only": {k: by_train[k] for k in
                                      ("model", "aux", "parameters", "train_acc_median",
                                       "val_acc_median", "test_acc_median",
                                       "current_stream_acc_median")},
        "on_seen_length_validation": {k: by_val[k] for k in
                                      ("model", "aux", "parameters", "train_acc_median",
                                       "val_acc_median", "test_acc_median",
                                       "current_stream_acc_median")},
        "best_test_over_every_matched_arm_and_seed": max(
            (r["test_acc"] for a in matched for r in a["rows"]))}
    harness.dump(args.tag, report)
    print(json.dumps(report["selection"], indent=1))


if __name__ == "__main__":
    main()
