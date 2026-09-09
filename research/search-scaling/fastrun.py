"""Same experiment protocol as harness.run_one, evaluated with the validated
vectorised forward. research/search-scaling/check_fast.py proves step-for-step
equivalence with tcn.learning.SoftProgram (max |loss diff| < 2e-6, identical argmax).
"""
from __future__ import annotations
import json, math, os, sys, time
from dataclasses import asdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import (N_IN, N_ROWS, POOLS, RunResult, sample_target, build_program,
                     selection_mask, relevant_inputs)
import fast

BITS = torch.stack([torch.tensor([float((r >> j) & 1) for r in range(N_ROWS)])
                    for j in range(N_IN)])


def mask_to_vec(m):
    return torch.tensor([float((m >> r) & 1) for r in range(N_ROWS)])


def run_one(cfg, seed):
    torch.set_num_threads(1)
    t0 = time.time()
    depth = cfg['depth']
    n_nodes = cfg.get('n_nodes', depth)
    tables = POOLS[cfg.get('pool', 'T16')]
    window = cfg.get('window')
    tindex = cfg.get('target_index', 0) + (0 if cfg.get('fixed_target') else 1000 * seed)
    gates, mask, gate_masks, _ = sample_target(
        depth, cfg.get('target_seed', 0), tindex,
        tables=tables if cfg.get('restrict_target', True) else None,
        window=window, min_relevant=cfg.get('min_relevant', 0))
    program, _ = build_program(n_nodes, tables, cfg['wiring'], gates, window)
    plan, _ = fast.compile_program(program)
    sizes = [len(n.candidates) for n in program.nodes]
    log2_space = sum(math.log2(s) for s in sizes)

    dense = cfg.get('supervision') == 'dense' and n_nodes == depth
    sup = [(n_nodes - 1, mask_to_vec(mask), 1.)]
    if dense:
        sup = [(j, mask_to_vec(gate_masks[N_IN + j]), cfg.get('probe_weight', 1.))
               for j in range(n_nodes - 1)] + sup

    torch.manual_seed(seed)
    logits = [torch.nn.Parameter(torch.randn(s) * cfg.get('init_scale', .1)) for s in sizes]
    opt = torch.optim.Adam(logits, lr=cfg.get('lr', .05))
    steps = cfg.get('steps', 500)
    tau0, tau1 = cfg.get('tau0', 1.), cfg.get('tau1', .1)
    first = None
    trace = []
    tau = tau0
    for step in range(steps):
        tau = (tau0 * (tau1 / tau0) ** (step / max(1, steps - 1))
               if cfg.get('anneal', True) else tau0)
        taus = [tau] * len(logits)
        opt.zero_grad()
        outs = fast.forward(plan, logits, taus, BITS)
        loss = sum(w * torch.nn.functional.binary_cross_entropy(
                       outs[i].clamp(1e-6, 1 - 1e-6), t) for i, t, w in sup)
        total = (loss + cfg.get('entropy_weight', .001) * (step / max(1, steps))
                 * fast.entropy(logits, taus))
        total.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(logits, 1e9))
        opt.step()
        if step % cfg.get('check_every', 5) == 0 or step == steps - 1:
            sel = {n.name: int(p.argmax()) for n, p in zip(program.nodes, logits)}
            ok = selection_mask(program, sel) == mask
            if ok and first is None:
                first = step
            if step % cfg.get('log_every', 20) == 0 or step == steps - 1:
                ent = fast.per_node_entropy(logits, taus)
                trace.append({'step': step, 'loss': float(loss.detach()),
                              'norm_entropy': sum(e / math.log(s) for e, s in zip(ent, sizes)) / len(sizes),
                              'grad_norm': gn, 'exact': ok})
    sel = {n.name: int(p.argmax()) for n, p in zip(program.nodes, logits)}
    got = selection_mask(program, sel)
    ent = fast.per_node_entropy(logits, [tau] * len(logits))
    return RunResult(cfg, seed, got == mask, first, float(loss.detach()),
                     sum(e / math.log(s) for e, s in zip(ent, sizes)) / len(sizes),
                     time.time() - t0, sizes, log2_space,
                     [sel[n.name] for n in program.nodes], mask, got,
                     bin((got ^ mask) & ((1 << N_ROWS) - 1)).count('1'), trace,
                     relevant_inputs(mask), None)


def _job(args):
    cfg, seed = args
    try:
        return asdict(run_one(cfg, seed))
    except Exception as e:
        return {'config': cfg, 'seed': seed, 'error': f'{type(e).__name__}: {e}'}


def sweep(jobs, workers=8, out=None):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    if out:
        json.dump(rows, open(out, 'w'))
    return rows
