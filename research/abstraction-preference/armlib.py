"""The retest's arms, with the description-cost term switchable.

`common.search` from `../recursive-abstraction-retest` is reproduced here with one
addition -- `mdl_weight * SoftProgram.description_cost()` in the training
objective -- so "with preference" and "without preference" differ in exactly one
term and nothing else. Everything else (initialization noise, learning rate,
anytime conformance criterion, stop-at-first-success) is unchanged, so the
`mdl_weight = 0` column is directly comparable to the retest's numbers.
"""
from __future__ import annotations

import sys, time
import torch

sys.path.insert(0, '../recursive-abstraction-retest')
import common
from tcn.learning import SoftProgram, tensor


def describe(hardened, registry):
    """Size, cost and shape of a discovered program, measured after pruning."""
    p = hardened.pruned()
    calls = [n for n in p.nodes if n.candidates[n.selected or 0].operator.name.startswith('module:')]
    return {'live_nodes': len(p.nodes), 'module_calls': len(calls),
            'description_bits': p.description_bits(registry),
            'description_bits_unpruned': hardened.description_bits(registry),
            'execution_cost': p.execution_cost(registry),
            'program': [f"{n.name} = {n.candidates[n.selected or 0].operator.name[:14]}"
                        f"({', '.join(n.candidates[n.selected or 0].sources)})" for n in p.nodes]}


def run(program, examples, signals, registry, seed, mdl_weight=0., steps=300, lr=0.05,
        eval_every=10, init_noise=0.5, stop_on_success=True):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = SoftProgram(program, registry)
    with torch.no_grad():
        for p in model.choices:
            p.add_(torch.randn_like(p) * init_noise)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([tensor(e['inputs'][k]) for e in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e['targets'][s.target]) for e in examples]) for s in signals}

    def task_loss():
        _, _, trace = model(inputs, return_trace=True)
        return model.probe_loss(trace, targets, signals)

    first, found, curve = None, None, []
    t0 = time.perf_counter(); last = 0
    for step in range(steps):
        opt.zero_grad()
        task = task_loss()
        loss = task + 0.001 * (step / max(1, steps)) * model.entropy()
        if mdl_weight:
            loss = loss + mdl_weight * model.description_cost()
        if loss.requires_grad:
            loss.backward(); opt.step()
        last = step
        if step % eval_every == 0 or step == steps - 1:
            exported = model.export()
            ok = common.conformant(exported, examples, signals, registry)
            curve.append({'step': step, 'task_loss': float(task.detach()),
                          'description_bits': float(model.description_cost().detach()),
                          'conformant': ok})
            if ok and first is None:
                first = step; found = describe(exported, registry)
                if stop_on_success: break
    wall = time.perf_counter() - t0
    exported = model.export()
    final_ok = common.conformant(exported, examples, signals, registry)
    return dict(seed=seed, mdl_weight=mdl_weight, first_conformant_step=first,
                steps_run=last + 1, wall_seconds=wall, seconds_per_step=wall / max(1, last + 1),
                final_task_loss=float(task_loss().detach()),
                final_description_bits=float(model.description_cost().detach()),
                conformant=bool(final_ok or first is not None), final_conformant=bool(final_ok),
                found=found if found is not None else (describe(exported, registry) if final_ok else None),
                final=describe(exported, registry), curve=curve)
