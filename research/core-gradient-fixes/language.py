"""The language track's two gradient benchmarks, before and after the surrogate fix.

Section 19 records gradient descent conforming in **0 of 44 runs** on the spaces
enumeration settles, with the cause measured: `lt`'s surrogate gradient is exactly
0.0 at delta >= 17 while the task operates near delta 48, and the derived `eq`
fix could not be applied because one temperature served both the surrogate and
the 256-way choice softmax.

Both arms run the *default* node temperatures. `before` is the shipped default
with nothing turned on; `after` calls `SoftProgram.scale_surrogates()`, which
widens `eq` to its declared carrier without touching the 256-way choice softmax.
The track's own `eq256` / `index0.1` arms are deliberately *not* rerun: they set
`SoftProgram.temperatures`, which is the choice temperature, so they no longer
mean what they meant when they were written.

MEASUREMENT WARNING. `SoftProgram` zero-initializes every choice logit, so seeds
vary nothing on their own; both runners add explicit initialization noise (0.01)
and the same seeds are used in both arms.

Run:  .venv/bin/python research/core-gradient-fixes/language.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "language-capability"))

import torch

from tcn.learning import SoftProgram, tensor
from tcn.search import space_size

EVALUATIONS = [0]


def anneal_fit(program, registry, signals, ex, seed, steps=400, lr=.05, noise=.01, floor=.05):
    """The track's `fit`, with the SURROGATE temperature annealed and the choice left alone.

    This arm exists only because the two temperatures are now separate. It is the
    crystallizer's own schedule -- geometric down to the 0.05 floor -- applied to
    the relaxation, so `eq` starts at the full carrier width (256) and finishes at
    12.8, where the constant-selection minimum is measured to be exact again. The
    candidate softmax is untouched throughout, which is precisely what could not
    be done before.
    """
    torch.manual_seed(seed); torch.set_num_threads(1)
    model = SoftProgram(program, registry)
    model.scale_surrogates()
    with torch.no_grad():
        for p in model.choices: p.add_(torch.randn_like(p) * noise)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([tensor(e['inputs'][k]) for e in ex]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e['targets'][s.target]) for e in ex]) for s in signals}
    grads = None
    for step in range(steps):
        tau = floor ** (step / max(1, steps - 1))
        for k in model.surrogate_scale: model.surrogate_scale[k] = tau
        opt.zero_grad()
        _, _, trace = model(inputs, return_trace=True)
        loss = model.probe_loss(trace, targets, signals)
        loss.backward()
        if step == 0:
            grads = {n.name: model.choices[i].grad.detach().clone() for i, n in enumerate(program.nodes)}
        opt.step()
    return model, model.selections(), float(loss.detach()), grads


def scaled_fit(program, registry, signals, ex, seed, steps=400, lr=.05, noise=.01, scaled=False):
    """The track's `fit`, with the carrier scaling switched on or off."""
    torch.manual_seed(seed); torch.set_num_threads(1)
    model = SoftProgram(program, registry)
    if scaled: model.scale_surrogates()
    with torch.no_grad():
        for p in model.choices: p.add_(torch.randn_like(p) * noise)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([tensor(e['inputs'][k]) for e in ex]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e['targets'][s.target]) for e in ex]) for s in signals}
    grads = None
    for step in range(steps):
        opt.zero_grad()
        _, _, trace = model(inputs, return_trace=True)
        loss = model.probe_loss(trace, targets, signals)
        loss.backward()
        if step == 0:
            grads = {n.name: model.choices[i].grad.detach().clone() for i, n in enumerate(program.nodes)}
        opt.step()
    return model, model.selections(), float(loss.detach()), grads


def stage_a(seeds=8, steps=400):
    import common, scaffolds
    from run_stage_a import examples
    from run_stage_a_grad import exact_accuracy, REFERENCE
    pool = common.dataset(600, seed0=0, split='train')
    train_eps = [e for e in pool if e['length'] in (2, 4, 6)][:12]
    program, registry, signals = scaffolds.stage_a()
    ex = examples(train_eps)
    out = {"space_size": space_size(program), "positions": len(ex),
           "reference_selection": REFERENCE, "arms": {}}
    print(f"== stage A: {out['space_size']} programs, reference {REFERENCE} ==", flush=True)
    for name in ("before", "after", "after+annealed surrogate"):
        t0 = time.perf_counter(); runs = []
        for s in range(seeds):
            if name.endswith("annealed surrogate"):
                model, sel, loss, grads = anneal_fit(program, registry, signals, ex, s, steps=steps)
            else:
                model, sel, loss, grads = scaled_fit(program, registry, signals, ex, s,
                                                     steps=steps, scaled=(name == "after"))
            EVALUATIONS[0] += steps * len(program.nodes)
            g = grads['open'].abs()
            runs.append({"seed": s, "loss": loss, "selection": sel,
                         "matches_reference": sel == REFERENCE,
                         "open_byte": sel['open'], "base": sel['base'],
                         "exact_train_accuracy": exact_accuracy(program, sel, registry, ex),
                         "eq_candidates_with_zero_gradient": int((g == 0).sum()),
                         "max_eq_choice_gradient": float(g.max()),
                         "base_choice_gradient_max": float(grads['base'].abs().max())})
        out["arms"][name] = {"seconds": round(time.perf_counter() - t0, 1),
                             "conformant": sum(r["matches_reference"] for r in runs),
                             "seeds": seeds,
                             "exact_train_accuracy_mean": sum(r["exact_train_accuracy"] for r in runs) / len(runs),
                             "max_eq_choice_gradient_mean": sum(r["max_eq_choice_gradient"] for r in runs) / len(runs),
                             "bytes_chosen": sorted({r["open_byte"] for r in runs}),
                             "bases_chosen": sorted({r["base"] for r in runs}),
                             "runs": runs}
        a = out["arms"][name]
        print(f"  {name:7s} conformant {a['conformant']}/{seeds}  exact-acc "
              f"{a['exact_train_accuracy_mean']:.3f}  eq choice grad "
              f"{a['max_eq_choice_gradient_mean']:.3e}  bytes {a['bytes_chosen']}", flush=True)
    return out


def stage_b(seeds=6, steps=300, n_train=24):
    import common, scaffolds
    from run_stage_b import build_module, examples, accuracy, baselines
    module, registry, _ = build_module()
    program, signals = scaffolds.stage_b(module, registry)
    pool = common.dataset(900, seed0=0, split='train')
    train_eps = [e for e in pool if e['length'] in (2, 4, 6)][:n_train]
    seen_held = [e for e in pool if e['length'] in (2, 4, 6)][n_train:n_train + 120]
    unseen = [e for e in common.dataset(1500, seed0=100000, split='test') if e['length'] not in (2, 4, 6)]
    ex = examples(train_eps)
    inputs = {'text': torch.stack([tensor(e['inputs']['text']) for e in ex])}
    targets = {'answer': torch.stack([tensor(e['targets']['answer']) for e in ex])}
    out = {"space_size": space_size(program),
           "baselines": {"heldout_unseen_lengths": baselines(unseen)}, "arms": {}}
    print(f"== stage B: {out['space_size']} programs ==", flush=True)
    for name, scaled in (("before", False), ("after", True)):
        t0 = time.perf_counter(); runs = []
        for s in range(seeds):
            torch.manual_seed(s); torch.set_num_threads(1)
            model = SoftProgram(program, registry)
            if scaled: model.scale_surrogates()
            with torch.no_grad():
                for p in model.choices: p.add_(torch.randn_like(p) * .01)
            opt = torch.optim.Adam(model.parameters(), lr=.05)
            first = None
            for step in range(steps):
                opt.zero_grad()
                _, _, trace = model(inputs, return_trace=True)
                EVALUATIONS[0] += len(program.nodes)
                loss = model.probe_loss(trace, targets, signals)
                loss.backward()
                if step == 0:
                    first = {k: float(model.choices[i].grad.abs().max())
                             for i, k in enumerate(n.name for n in program.nodes)
                             if k in ('symbols', 'plus', 'minus', 'answer')}
                opt.step()
            sel = model.selections()
            tr, _ = accuracy(program, sel, registry, train_eps)
            un, _ = accuracy(program, sel, registry, unseen)
            runs.append({"seed": s, "loss": float(loss.detach()),
                         "selection": {k: sel[k] for k in ('symbols', 'plus', 'minus', 'answer')},
                         "exact_train_accuracy": tr, "heldout_unseen_lengths": un,
                         "first_step_choice_gradients": first})
            print(f"    seed {s}: train {tr:.3f} unseen {un:.3f}", flush=True)
        out["arms"][name] = {"seconds": round(time.perf_counter() - t0, 1), "seeds": seeds,
                             "exact_train_conformant": sum(r["exact_train_accuracy"] == 1.0 for r in runs),
                             "mean_heldout_unseen": sum(r["heldout_unseen_lengths"] for r in runs) / len(runs),
                             "best_heldout_unseen": max(r["heldout_unseen_lengths"] for r in runs),
                             "runs": runs}
        a = out["arms"][name]
        print(f"  {name:7s} exact-train conformant {a['exact_train_conformant']}/{seeds}  "
              f"mean unseen {a['mean_heldout_unseen']:.4f} (majority "
              f"{out['baselines']['heldout_unseen_lengths']['majority_constant']:.4f})", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    torch.set_num_threads(1)
    report = {"stage_a": stage_a(), "stage_b": stage_b()}
    report["node_evaluations"] = EVALUATIONS[0]
    report["wall_seconds"] = round(time.perf_counter() - t0, 2)
    (HERE / "language.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"node_evaluations": report["node_evaluations"],
                      "wall_seconds": report["wall_seconds"]}))
