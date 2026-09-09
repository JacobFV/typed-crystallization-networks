"""Matched neural baseline for FINDINGS §23 -- a policy acting in a live OS.

THE TASK. `/home/agent/task.txt` holds `<name> = <digit>`; the objective is met
when the file contains `str(digit+1)`. At tick 0 the terminal shows only the
setup write's JSON, which carries the file's *length*, not its content, so the
policy must read first, then write a value computed from what it saw, then stop.
Max return is 2 (`horizon - 1`).

MATCHING CONDITIONS.

* **Inputs.** `observations['terminal']` -- a `(length: u32, 4096 x byte)` typed
  text -- and the previous action's one-hot, which is exactly the pair the frozen
  program's two input ports carry (`terminal`, `action`). The baseline reads the
  first `PREFIX` bytes of the terminal; `assert_padding_is_constant` verifies on
  every train and test episode that every byte beyond `PREFIX` is 0, so this is a
  restriction that discards nothing (and if it ever fails, the run aborts). No
  baseline sees `probes`, `latent_states`, the document string, the digit, the
  file name, the episode index or the split.
* **Supervision.** `task.examples_from`'s `reference` (the verb index of the
  scripted trajectory) and `reference_byte` (the byte to write) -- byte for byte
  the two targets `search_run.py` enumerated the TCN program against.
* **Splits.** `task.TRAIN_DOCUMENTS` (5 documents, 3 ticks each = 15 decision
  points, 5 of them write decisions) for fitting; `task.TEST_DOCUMENTS` (10
  documents, names and digits never seen) for the closed loop, run at
  `index=1000+i, split='test'`, exactly as `closed_loop.run_agent` does.
* **Quality is measured in the live OS**, not on a proxy: the chosen action is
  executed by the shipped kernel and the reward is the shipped reward.
* **Cost is measured program-only.** Per-step wall clock separates the policy's
  own forward pass from the ~968 ms kernel round trip, which is charged to
  neither method.
* **Trivial reference.** `closed_loop.py`'s own baselines -- always write "5",
  read then write the modal digit, uniform random verb and digit -- re-run here
  on the same 10 documents.

Run: `.venv/bin/python research/neural-baselines/computer_baseline.py`
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "computer-capability"))

import torch
import torch.nn as nn

import common as harness
import task as T
from tcn.generation import Action, read_text, text_value

torch.set_num_threads(1)

# The typed carrier is 4,096 bytes. `assert_padding_is_constant` verifies on every
# train and test episode that no byte at or beyond `PREFIX` is non-zero -- the
# longest terminal this task produces is 45 bytes -- so reading a 64-byte window
# discards no observable information and the run aborts if that ever stops being
# true. It is a restriction of the baseline's input, never an enlargement.
PREFIX = 64
VERBS = T.TEMPLATES          # ('wait', 'read', 'write')


# --- data ------------------------------------------------------------------
def collect(documents, horizon=3):
    """Scripted reference trajectories through the live kernel, as `task.collect`."""
    episodes = []
    for name, digit in documents:
        host, actions = T.scripted_episode(name, digit, horizon=horizon)
        rows = T.examples_from(host, actions)
        episodes.append({"name": name, "digit": digit, "rows": rows})
    return episodes


def assert_padding_is_constant(episodes):
    """The truncation to `PREFIX` bytes must discard nothing. Checked, not assumed."""
    worst = 0
    for e in episodes:
        for row in e["rows"]:
            from tcn.types import Value
            terminal = Value.from_dict(row["terminal"])
            raw = terminal.raw
            worst = max(worst, int(raw[0]))
            tail = [int(b) for b in raw[1][PREFIX:]]
            assert set(tail) <= {0}, "terminal carries content past PREFIX"
    return {"max_terminal_length": worst, "prefix_bytes_used": PREFIX,
            "bytes_beyond_prefix_all_zero": True}


def featurise(episodes):
    from tcn.types import Value
    tokens, lengths, previous, verb, byte, has_byte = [], [], [], [], [], []
    for e in episodes:
        for row in e["rows"]:
            terminal = Value.from_dict(row["terminal"])
            tokens.append([int(b) for b in terminal.raw[1][:PREFIX]])
            lengths.append(int(terminal.raw[0]))
            previous.append([1.0 if j == row["previous"] else 0.0 for j in range(3)])
            verb.append(row["reference"])
            byte.append(row["reference_byte"] if row["reference_byte"] is not None else 0)
            has_byte.append(row["reference_byte"] is not None)
    return {"tokens": torch.tensor(tokens, dtype=torch.long),
            "length": torch.tensor(lengths, dtype=torch.float32).unsqueeze(1),
            "previous": torch.tensor(previous, dtype=torch.float32),
            "verb": torch.tensor(verb, dtype=torch.long),
            "byte": torch.tensor(byte, dtype=torch.float32),
            "has_byte": torch.tensor(has_byte, dtype=torch.bool)}


def observe(host, previous_index):
    """The live counterpart of `featurise`, from a real record."""
    terminal = host.records[-1].observations["terminal"].value
    raw = terminal.raw
    return {"tokens": torch.tensor([[int(b) for b in raw[1][:PREFIX]]], dtype=torch.long),
            "length": torch.tensor([[float(raw[0])]]),
            "previous": torch.tensor([[1.0 if j == previous_index else 0.0
                                       for j in range(3)]])}


# --- models ----------------------------------------------------------------
class Backbone(nn.Module):
    """Common head structure: 3 verb logits and one byte output.

    `byte_mode='reg'` regresses the byte value (so `+1` is expressible as an
    affine map and the two digits absent from training are reachable);
    `byte_mode='cls'` classifies over 256 bytes (so they are not). Both are
    reported -- which one a practitioner picks is exactly the design decision the
    TCN's `shift` node made by choosing `add(value, u1)` over `identity`.
    """

    def __init__(self, feature_dim, byte_mode="reg"):
        super().__init__()
        self.byte_mode = byte_mode
        self.verb = nn.Linear(feature_dim + 3, 3)
        self.byte = nn.Linear(feature_dim, 1 if byte_mode == "reg" else 256)

    def heads(self, feature, previous):
        return self.verb(torch.cat([feature, previous], dim=1)), self.byte(feature)

    def decode_byte(self, out):
        if self.byte_mode == "reg":
            return int(round(float(out.reshape(-1)[0]) * 255))
        return int(out.reshape(-1).argmax())


class TerminalGRU(Backbone):
    def __init__(self, embed=8, hidden=16, byte_mode="reg"):
        super().__init__(hidden, byte_mode)
        self.embed = nn.Embedding(256, embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)

    def forward(self, tokens, length, previous):
        h, _ = self.rnn(self.embed(tokens))
        idx = (length.squeeze(1).long() - 1).clamp(0, tokens.shape[1] - 1)
        return self.heads(h[torch.arange(h.shape[0]), idx], previous)


class TerminalCNN(Backbone):
    def __init__(self, embed=8, width=32, byte_mode="reg"):
        super().__init__(width, byte_mode)
        self.embed = nn.Embedding(256, embed)
        self.body = nn.Sequential(nn.Conv1d(embed, width, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(width, width, 3, padding=1), nn.ReLU())

    def forward(self, tokens, length, previous):
        h = self.body(self.embed(tokens).transpose(1, 2))
        mask = (torch.arange(tokens.shape[1]).unsqueeze(0) < length).unsqueeze(1)
        h = h.masked_fill(~mask, -1e9).max(dim=2).values
        return self.heads(h, previous)


class AbsoluteMLP(Backbone):
    """Reads a fixed window of absolute byte addresses -- the family the TCN's
    address pool contained and its search had to reject, because the training
    documents have five different lengths."""

    def __init__(self, window=48, embed=8, hidden=32, byte_mode="reg"):
        super().__init__(hidden, byte_mode)
        self.window = window
        self.embed = nn.Embedding(256, embed)
        self.body = nn.Sequential(nn.Linear(window * embed + 1, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, tokens, length, previous):
        e = self.embed(tokens[:, :self.window]).reshape(tokens.shape[0], -1)
        return self.heads(self.body(torch.cat([e, length / PREFIX], dim=1)), previous)


class RelativeHintMLP(Backbone):
    """DECLARED STRUCTURAL HINT, not a matched baseline: it is *given* the
    computed address `length - k` that the TCN's `pos` node had to search for,
    among 16 candidates, against 15 constant alternatives."""

    def __init__(self, taps=4, embed=8, hidden=32, byte_mode="reg"):
        super().__init__(hidden, byte_mode)
        self.taps = taps
        self.embed = nn.Embedding(256, embed)
        self.body = nn.Sequential(nn.Linear(taps * embed + 1, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, tokens, length, previous):
        idx = (length.long() - 1 - torch.arange(self.taps).unsqueeze(0)).clamp(0, tokens.shape[1] - 1)
        e = self.embed(torch.gather(tokens, 1, idx)).reshape(tokens.shape[0], -1)
        return self.heads(self.body(torch.cat([e, length / PREFIX], dim=1)), previous)


ZOO = {
    "gru_h16": lambda m: TerminalGRU(8, 16, m),
    "gru_h32": lambda m: TerminalGRU(8, 32, m),
    "cnn_w32": lambda m: TerminalCNN(8, 32, m),
    "mlp_absolute_w48": lambda m: AbsoluteMLP(48, 8, 32, m),
}
HINT_ZOO = {"mlp_relative_hint": lambda m: RelativeHintMLP(4, 8, 32, m)}


def train_one(factory, mode, data, seed, steps=2000, lr=5e-3):
    torch.manual_seed(seed)
    model = factory(mode)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    started = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        verb_logits, byte_out = model(data["tokens"], data["length"], data["previous"])
        loss = ce(verb_logits, data["verb"])
        m = data["has_byte"]
        if m.any():
            if mode == "reg":
                loss = loss + ((byte_out[m].reshape(-1) - data["byte"][m] / 255.) ** 2).mean()
            else:
                loss = loss + ce(byte_out[m], data["byte"][m].long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model, {"train_seconds": time.perf_counter() - started,
                   "final_loss": float(loss.detach()), "steps": steps, "lr": lr,
                   "parameters": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def supervised_accuracy(model, data):
    verb_logits, byte_out = model(data["tokens"], data["length"], data["previous"])
    verb_ok = float((verb_logits.argmax(1) == data["verb"]).float().mean())
    m = data["has_byte"]
    if model.byte_mode == "reg":
        got = (byte_out[m].reshape(-1) * 255).round()
    else:
        got = byte_out[m].argmax(1).float()
    byte_ok = float((got == data["byte"][m]).float().mean())
    return {"verb_accuracy": verb_ok, "byte_accuracy": byte_ok}


# --- closed loop -----------------------------------------------------------
@torch.no_grad()
def rollout(model, documents, horizon=3, path=T.TASK_PATH):
    """The model's chosen action executed by the shipped kernel, reward as shipped."""
    rows, policy_ms = [], []
    for i, (name, digit) in enumerate(documents):
        host = T.host(name, digit, horizon, objective_path=path, seed=0, index=1000 + i,
                      split="test")
        previous = 0
        actions, written = [], []
        for _ in range(horizon):
            if host.records[-1].done:
                break
            obs = observe(host, previous)
            started = time.perf_counter_ns()
            verb_logits, byte_out = model(obs["tokens"], obs["length"], obs["previous"])
            index = int(verb_logits.argmax())
            byte = model.decode_byte(byte_out)
            policy_ms.append((time.perf_counter_ns() - started) / 1e6)
            verb = VERBS[index]
            if verb == "wait":
                action = T.wait_action()
            elif verb == "read":
                action = T.read_action()
            else:
                text = chr(byte) if 0 <= byte < 256 else ""
                action = T.act("write", path=path, text=text)
                written.append(text)
            host.step((action,))
            actions.append(verb)
            previous = index
        total = sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records)
        rows.append({"document": T.document(name, digit), "target": T.target(digit),
                     "return": total, "actions": actions, "written": written})
    return {"episodes": len(rows), "mean_return": sum(r["return"] for r in rows) / len(rows),
            "max_return": horizon - 1,
            "solved": sum(r["return"] == horizon - 1 for r in rows),
            "policy_ms_median": statistics.median(policy_ms) if policy_ms else None,
            "rows": rows}


def trivial_references(documents, horizon=3, path=T.TASK_PATH):
    """`closed_loop.py`'s own baselines, re-run on the same 10 held-out documents."""
    def scripted(choose, label):
        rows = []
        for i, (name, digit) in enumerate(documents):
            host = T.host(name, digit, horizon, objective_path=path, seed=0, index=1000 + i,
                          split="test")
            rng = random.Random(1000 + i)
            for tick in range(horizon):
                if host.records[-1].done:
                    break
                host.step((choose(tick, rng),))
            rows.append({"document": T.document(name, digit),
                         "return": sum(sum(v.decoded for v in r.reward_components.values())
                                       for r in host.records)})
        return {"label": label, "episodes": len(rows),
                "mean_return": sum(r["return"] for r in rows) / len(rows),
                "max_return": horizon - 1,
                "solved": sum(r["return"] == horizon - 1 for r in rows)}

    def uniform(tick, rng):
        verb = rng.choice(["wait", "read", "write"])
        if verb == "wait":
            return T.wait_action()
        if verb == "read":
            return T.read_action()
        return T.act("write", path=path, text=str(rng.randrange(10)))

    return [scripted(lambda t, r: T.act("write", path=path, text="5"), 'always write "5"'),
            scripted(lambda t, r: T.read_action() if t == 0
                     else T.act("write", path=path, text="5"),
                     'read then write the modal digit "5"'),
            scripted(uniform, "uniform random verb, random digit"),
            scripted(lambda t, r: T.wait_action(), "always wait")]


# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--live-arms", type=int, default=3)
    ap.add_argument("--tag", default="computer")
    args = ap.parse_args()

    print("collecting scripted trajectories through the live kernel ...", flush=True)
    started = time.perf_counter()
    train_eps = collect(T.TRAIN_DOCUMENTS)
    test_eps = collect(T.TEST_DOCUMENTS)
    collect_seconds = time.perf_counter() - started
    padding = assert_padding_is_constant(train_eps + test_eps)
    train = featurise(train_eps)
    held = featurise(test_eps)
    print(json.dumps(padding), flush=True)

    report = {"padding_check": padding,
              "budget": {"train_documents": len(T.TRAIN_DOCUMENTS),
                         "train_decision_points": int(train["verb"].shape[0]),
                         "train_write_decisions": int(train["has_byte"].sum()),
                         "collect_seconds": collect_seconds},
              "supervised_trivial_reference": {
                  "constant_verb_accuracy": max(
                      float((train["verb"] == v).float().mean()) for v in range(3)),
                  "constant_byte_accuracy_train": float(
                      max((train["byte"][train["has_byte"]] == b).float().mean()
                          for b in train["byte"][train["has_byte"]].unique())),
                  "random_verb_accuracy": 1 / 3, "random_byte_accuracy": 1 / 256},
              "arms": []}

    # A kernel step costs about a second on this host, and the trivial references
    # do not depend on any model, so they are measured once and cached. The cache
    # is a plain JSON in `out/`; delete it to re-measure.
    cache = harness.OUT / "computer_trivial_cache.json"
    if cache.exists():
        print(f"trivial closed-loop references: reusing {cache.name}", flush=True)
        report["trivial_reference_closed_loop"] = json.loads(cache.read_text())
    else:
        print("trivial closed-loop references (live kernel) ...", flush=True)
        report["trivial_reference_closed_loop"] = trivial_references(T.TEST_DOCUMENTS)
        cache.write_text(json.dumps(report["trivial_reference_closed_loop"], indent=1))
    print(json.dumps([{k: v for k, v in r.items()} for r in
                      report["trivial_reference_closed_loop"]], indent=1), flush=True)
    harness.dump(args.tag, report)

    # Phase 1: fit every arm and score it offline. A kernel step costs about a
    # second, so the live loop is spent only where it can change a conclusion.
    trained = {}
    for label, zoo in (("matched", ZOO), ("structural_hint", HINT_ZOO)):
        for name, factory in zoo.items():
            for mode in ("reg", "cls"):
                rows = []
                for seed in range(args.seeds):
                    model, info = train_one(factory, mode, train, seed, steps=args.steps)
                    fit = supervised_accuracy(model, train)
                    off = supervised_accuracy(model, held)
                    trained[(label, name, mode, seed)] = model
                    rows.append({"seed": seed, **info, "train": fit,
                                 "heldout_supervised": off})
                    print(f"  {label:15s} {name:18s} {mode} seed {seed}: "
                          f"params {info['parameters']} train verb "
                          f"{fit['verb_accuracy']:.2f} byte {fit['byte_accuracy']:.2f} | "
                          f"held verb {off['verb_accuracy']:.2f} byte "
                          f"{off['byte_accuracy']:.2f}", flush=True)
                report["arms"].append({
                    "kind": label, "model": name, "byte_head": mode,
                    "parameters": rows[0]["parameters"],
                    "train_verb_accuracy_median": statistics.median(
                        r["train"]["verb_accuracy"] for r in rows),
                    "train_byte_accuracy_median": statistics.median(
                        r["train"]["byte_accuracy"] for r in rows),
                    "heldout_verb_accuracy_median": statistics.median(
                        r["heldout_supervised"]["verb_accuracy"] for r in rows),
                    "heldout_byte_accuracy_median": statistics.median(
                        r["heldout_supervised"]["byte_accuracy"] for r in rows),
                    "train_seconds_median": statistics.median(r["train_seconds"] for r in rows),
                    "rows": rows})
                harness.dump(args.tag, report)

    # Phase 2: the live kernel, for the arms selected on TRAINING FIT ONLY -- the
    # information the TCN search had. The held-out closed loop is never used to
    # choose an arm, only to report one.
    def score(arm):
        return (arm["train_verb_accuracy_median"], arm["train_byte_accuracy_median"],
                -arm["parameters"])

    matched = sorted([a for a in report["arms"] if a["kind"] == "matched"],
                     key=score, reverse=True)
    hint = sorted([a for a in report["arms"] if a["kind"] == "structural_hint"],
                  key=score, reverse=True)
    chosen = matched[:args.live_arms] + hint[:1]
    print(f"live closed loop for {len(chosen)} arms x {args.seeds} seeds ...", flush=True)
    for arm in chosen:
        live_rows = []
        for row in arm["rows"]:
            model = trained[(arm["kind"], arm["model"], arm["byte_head"], row["seed"])]
            live = rollout(model, T.TEST_DOCUMENTS)
            row["closed_loop"] = {k: v for k, v in live.items() if k != "rows"}
            row["closed_loop_rows"] = live["rows"]
            live_rows.append(live)
            print(f"  LIVE {arm['model']:18s} {arm['byte_head']} seed {row['seed']}: "
                  f"{live['solved']}/10 solved, mean return {live['mean_return']:.2f}, "
                  f"policy {live['policy_ms_median']:.3f} ms", flush=True)
        arm["solved_median"] = statistics.median(r["solved"] for r in live_rows)
        arm["solved_max"] = max(r["solved"] for r in live_rows)
        arm["mean_return_median"] = statistics.median(r["mean_return"] for r in live_rows)
        arm["policy_ms_median"] = statistics.median(r["policy_ms_median"] for r in live_rows)
        harness.dump(args.tag, report)

    best = matched[0]
    report["selected_on_training_fit"] = {k: best.get(k) for k in
                                          ("model", "byte_head", "parameters",
                                           "train_verb_accuracy_median",
                                           "train_byte_accuracy_median",
                                           "heldout_byte_accuracy_median",
                                           "solved_median", "mean_return_median",
                                           "policy_ms_median")}
    report["best_closed_loop_over_every_matched_arm_run_live"] = max(
        (a["solved_max"] for a in matched if "solved_max" in a), default=None)
    harness.dump(args.tag, report)
    print(json.dumps(report["selected_on_training_fit"], indent=1))


if __name__ == "__main__":
    main()
