import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import build_program, sample_target, POOLS, N_IN, N_ROWS, examples_for
from tcn.learning import SoftProgram, tensor
import fast

def check(depth, wiring, seed, steps=40):
    gates, mask, gm, _ = sample_target(depth, 0, 1000 * seed)
    program, registry = build_program(depth, POOLS['T16'], wiring, gates)
    exs = examples_for(mask)
    from tcn.graph import Signal
    from tcn.types import BOOL
    signals = (Signal(f'g{depth-1}', 'y', ('core',), BOOL, 'bce'),)
    torch.manual_seed(seed)
    m = SoftProgram(program, registry)
    with torch.no_grad():
        for p in m.choices: p.normal_(0., .1)
    inputs = {k: torch.stack([tensor(e['inputs'][k]) for e in exs]) for k, _ in program.inputs}
    targets = {'y': torch.stack([tensor(e['targets']['y']) for e in exs])}
    opt = torch.optim.Adam(m.parameters(), lr=.05)

    plan, ports = fast.compile_program(program)
    logits = [torch.nn.Parameter(p.detach().clone()) for p in m.choices]
    fin = torch.stack([torch.tensor([float((r >> j) & 1) for r in range(N_ROWS)])
                       for j in range(N_IN)])
    ftar = targets['y'].squeeze(-1)
    fopt = torch.optim.Adam(logits, lr=.05)
    worst = 0.
    for step in range(steps):
        tau = 1.0 * (0.1 / 1.0) ** (step / max(1, steps - 1))
        for n in program.nodes: m.temperatures[n.name] = tau
        taus = [tau] * len(logits)
        opt.zero_grad(); _, _, tr = m(inputs, return_trace=True)
        L1 = m.probe_loss(tr, targets, signals) + .001 * (step / steps) * m.entropy()
        L1.backward(); opt.step()
        fopt.zero_grad()
        outs = fast.forward(plan, logits, taus, fin)
        pred = outs[-1].clamp(1e-6, 1 - 1e-6)
        L2 = torch.nn.functional.binary_cross_entropy(pred, ftar) + \
             .001 * (step / steps) * fast.entropy(logits, taus)
        L2.backward(); fopt.step()
        worst = max(worst, abs(float(L1) - float(L2)))
    sel1 = [int(p.argmax()) for p in m.choices]
    sel2 = [int(p.argmax()) for p in logits]
    return worst, sel1 == sel2, sel1, sel2

if __name__ == '__main__':
    ok = True
    for depth, wiring in [(1,'free'),(2,'free'),(3,'free'),(4,'free'),(3,'supplied'),(4,'supplied')]:
        for seed in (0, 1, 2):
            w, same, s1, s2 = check(depth, wiring, seed)
            print(f'depth={depth} {wiring} seed={seed}: max |loss diff| = {w:.3e}, '
                  f'identical argmax = {same}')
            ok &= (w < 1e-5 and same)
    print('EQUIVALENT' if ok else 'MISMATCH')
