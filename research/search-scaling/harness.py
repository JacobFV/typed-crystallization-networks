"""Difficulty sweep for typed boolean-program synthesis on the `logic` generator.

Nothing under tcn/ or generators/ is modified; this only composes the public API.
"""
from __future__ import annotations
import itertools, json, math, os, sys, time
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate, Signal
from tcn.learning import SoftProgram, tensor

N_IN = 4
N_ROWS = 1 << N_IN            # 16 input assignments over 4 bits
FULL = (1 << N_ROWS) - 1      # 16-bit truth-table mask over those assignments

# --- semantic names for the 16 two-input tables (index i = 2*a + b) ---
TABLE_NAMES = {0:'false',1:'nor',2:'a<b',3:'notA',4:'a>b',5:'notB',6:'xor',7:'nand',
               8:'and',9:'xnor',10:'B',11:'a<=b',12:'A',13:'a>=b',14:'or',15:'true'}
POOLS = {
    'T2':  (6, 8),                              # xor, and
    'T4':  (1, 6, 8, 14),                       # nor, xor, and, or
    'T8':  (1, 6, 7, 8, 9, 10, 12, 14),
    'T16': tuple(range(16)),
}

# ---------------------------------------------------------------- target masks
INPUT_MASKS = tuple(sum(1 << m for m in range(N_ROWS) if (m >> j) & 1) for j in range(N_IN))

def apply_table(am, bm, table):
    na, nb = FULL ^ am, FULL ^ bm
    parts = (na & nb, na & bm, am & nb, am & bm)
    out = 0
    for i, p in enumerate(parts):
        if (table >> i) & 1:
            out |= p
    return out & FULL

def circuit_mask(gates):
    vals = list(INPUT_MASKS)
    for a, b, t in gates:
        vals.append(apply_table(vals[a], vals[b], t))
    return vals[-1], vals

def relevant_inputs(mask):
    n = 0
    for j in range(N_IN):
        flip = 0
        for m in range(N_ROWS):
            if ((mask >> m) & 1) != ((mask >> (m ^ (1 << j))) & 1):
                flip = 1
        n += flip
    return n

def sample_target(depth, seed, index, tables=None, window=None, max_tries=60000,
                  min_relevant=0):
    """Draw a real `logic` episode; reject circuits outside the restricted pool."""
    from tcn.generation import Host
    tries = 0
    while tries < max_tries:
        host = Host.create('logic', seed=seed, index=index + tries * 7919,
                           configuration={'depth': depth})
        gates = [tuple(g) for g in host.state['gates']]
        tries += 1
        if tables is not None and any(g[2] not in tables for g in gates):
            continue
        if window is not None and any(min(g[0], g[1]) < N_IN + j - window for j, g in enumerate(gates)):
            continue
        mask, vals = circuit_mask(gates)
        if min_relevant and relevant_inputs(mask) < min_relevant:
            continue
        return gates, mask, vals, tries
    raise RuntimeError('no target satisfied the restriction')

# ---------------------------------------------------------------- the scaffold
def build_program(n_nodes, tables, wiring, target_gates=None, window=None):
    r = Registry()
    inputs = tuple((f'b{j}', BOOL) for j in range(N_IN))
    names = [f'b{j}' for j in range(N_IN)]
    nodes = []
    for j in range(n_nodes):
        if wiring == 'supplied':
            pairs = [(names[target_gates[j][0]], names[target_gates[j][1]])]
        else:
            pool = names if window is None else names[max(0, len(names) - window):]
            pairs = list(itertools.product(pool, repeat=2))
        cands = tuple(Candidate(r.resolve(f'truth_{t}', (BOOL, BOOL)), p)
                      for p in pairs for t in tables)
        nodes.append(Node(f'g{j}', BOOL, cands, 'core', j + 1))
        names.append(f'g{j}')
    program = Program(inputs, tuple(nodes), (('y', f'g{n_nodes-1}'),)).validate(r)
    return program, r

def examples_for(mask, gate_masks=None, probe_nodes=()):
    exs = []
    for m in range(N_ROWS):
        ins = {f'b{j}': Value.of(BOOL, bool((m >> j) & 1)) for j in range(N_IN)}
        tg = {'y': Value.of(BOOL, bool((mask >> m) & 1))}
        for name, gm in zip(probe_nodes, gate_masks or ()):
            tg[name] = Value.of(BOOL, bool((gm >> m) & 1))
        exs.append({'inputs': ins, 'targets': tg})
    return exs

# ------------------------------------------------------- fast discrete readout
def selection_mask(program, selections):
    vals = {f'b{j}': INPUT_MASKS[j] for j in range(N_IN)}
    for n in program.nodes:
        c = n.candidates[selections[n.name]]
        t = int(c.operator.name[6:])
        vals[n.name] = apply_table(vals[c.sources[0]], vals[c.sources[1]], t)
    return vals[program.outputs[0][1]]

# ------------------------------------------------------------------- one run
@dataclass
class RunResult:
    config: dict
    seed: int
    success: bool
    first_success_step: int | None
    final_loss: float
    final_entropy: float          # mean normalised choice entropy at the end
    seconds: float
    n_candidates: list
    log2_space: float
    selections: list
    target_mask: int
    final_mask: int
    hamming: int                  # rows wrong out of 16
    trace: list = field(default_factory=list)
    relevant: int = 0
    verified: bool | None = None

def run_one(cfg, seed):
    torch.set_num_threads(1)
    t0 = time.time()
    depth = cfg['depth']; n_nodes = cfg.get('n_nodes', depth)
    tables = POOLS[cfg.get('pool', 'T16')]
    window = cfg.get('window')
    tindex = cfg.get('target_index', 0) + (0 if cfg.get('fixed_target') else 1000 * seed)
    gates, mask, gate_masks, _ = sample_target(depth, cfg.get('target_seed', 0), tindex,
                                               tables=tables if cfg.get('restrict_target', True) else None,
                                               window=window,
                                               min_relevant=cfg.get('min_relevant', 0))
    program, registry = build_program(n_nodes, tables, cfg['wiring'], gates, window)
    probe_nodes = ()
    if cfg.get('supervision') == 'dense' and n_nodes == depth:
        probe_nodes = tuple(f'g{j}' for j in range(n_nodes - 1))
    exs = examples_for(mask, gate_masks[N_IN:N_IN + n_nodes - 1], probe_nodes)
    signals = tuple(Signal(f'g{j}', f'g{j}', ('core',), BOOL, 'bce', weight=cfg.get('probe_weight', 1.))
                    for j in range(n_nodes - 1) if f'g{j}' in probe_nodes)
    signals = signals + (Signal(f'g{n_nodes-1}', 'y', ('core',), BOOL, 'bce'),)

    torch.manual_seed(seed)
    model = SoftProgram(program, registry)
    with torch.no_grad():
        for p in model.choices:
            p.normal_(0., cfg.get('init_scale', .1))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', .05))
    inputs = {k: torch.stack([tensor(e['inputs'][k]) for e in exs]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e['targets'][s.target]) for e in exs]) for s in signals}
    steps = cfg.get('steps', 400)
    tau0, tau1 = cfg.get('tau0', 1.), cfg.get('tau1', .1)
    anneal = cfg.get('anneal', True)
    sizes = [len(n.candidates) for n in program.nodes]
    log2_space = sum(math.log2(s) for s in sizes)

    first = None; trace = []
    for step in range(steps):
        if anneal:
            tau = tau0 * (tau1 / tau0) ** (step / max(1, steps - 1))
            for n in program.nodes:
                model.temperatures[n.name] = tau
        opt.zero_grad()
        _, _, tr = model(inputs, return_trace=True)
        loss = model.probe_loss(tr, targets, signals)
        total = loss + cfg.get('entropy_weight', .001) * (step / max(1, steps)) * model.entropy()
        total.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
        opt.step()
        if step % cfg.get('check_every', 5) == 0 or step == steps - 1:
            sel = model.selections()
            got = selection_mask(program, sel)
            ok = got == mask
            if ok and first is None:
                first = step
            if step % cfg.get('log_every', 20) == 0 or step == steps - 1:
                ent = [float(-(q * q.clamp_min(1e-12).log()).sum())
                       for q in (torch.softmax(p / model.temperatures[n.name], 0)
                                 for n, p in zip(program.nodes, model.choices))]
                trace.append({'step': step, 'loss': float(loss.detach()),
                              'norm_entropy': sum(e / math.log(s) for e, s in zip(ent, sizes)) / len(sizes),
                              'grad_norm': gn, 'exact': ok})
    sel = model.selections()
    got = selection_mask(program, sel)
    ham = bin((got ^ mask) & FULL).count('1')
    ent = [float(-(q * q.clamp_min(1e-12).log()).sum())
           for q in (torch.softmax(p / model.temperatures[n.name], 0)
                     for n, p in zip(program.nodes, model.choices))]
    verified = None
    if cfg.get('verify'):
        exact = model.export()
        verified = True
        for e in exs:
            out, _ = exact.run(e['inputs'], registry=registry)
            if out['y'].decoded != e['targets']['y'].decoded:
                verified = False
    return RunResult(cfg, seed, got == mask, first, float(loss.detach()),
                     sum(e / math.log(s) for e, s in zip(ent, sizes)) / len(sizes),
                     time.time() - t0, sizes, log2_space,
                     [sel[n.name] for n in program.nodes], mask, got, ham, trace,
                     relevant_inputs(mask), verified)

def _job(args):
    cfg, seed = args
    try:
        return asdict(run_one(cfg, seed))
    except Exception as e:
        return {'config': cfg, 'seed': seed, 'error': f'{type(e).__name__}: {e}'}

def sweep(jobs, workers=18, out=None):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    if out:
        with open(out, 'w') as f:
            json.dump(rows, f)
    return rows
