"""Stage A by gradient descent, on the identical candidate space enumeration swept.

`SoftProgram` zero-initializes every choice logit (finding P2), so seeds do not
perturb synthesis at all; explicit initialization noise is added here, and said
so, exactly as track 1 did. Arms vary only the node temperatures, which is the
one place a caller can change a relaxation without touching `tcn/`:

  shipped      tau = 1 everywhere (as `SoftProgram.__init__` sets it)
  eq256        tau = 2^bits = 256 at the `eq` node (section 16's derived fix)
  index0.1     tau = 0.1 at the `index` node (sharpens section 16's M3 kernel)
  both         the two together

`tcn/` is unmodified; `SoftProgram.temperatures` is a public per-node dict.
"""
from __future__ import annotations
import sys, os, json, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import common, scaffolds
from run_stage_a import examples
from tcn.learning import SoftProgram, tensor
from tcn.search import space_size

REFERENCE = {'bytes': 0, 'base': 14, 'addr': 0, 'byte': 0, 'open': 40}

def fit(program, registry, signals, ex, taus, seed, steps=400, lr=.05, noise=.01):
    torch.manual_seed(seed); torch.set_num_threads(1)
    model = SoftProgram(program, registry)
    with torch.no_grad():
        for p in model.choices: p.add_(torch.randn_like(p) * noise)
    model.temperatures.update(taus)
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
            grads = {n.name: model.choices[i].grad.detach().clone()
                     for i, n in enumerate(program.nodes)}
        opt.step()
    sel = model.selections()
    return model, sel, float(loss.detach()), grads

def exact_accuracy(program, selections, registry, ex):
    ok = 0
    for x in ex:
        try:
            out, _ = program.run(x['inputs'], registry=registry, selections=selections)
        except Exception:
            continue
        ok += out['open'].decoded == x['targets']['open'].decoded
    return ok / max(1, len(ex))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--out', default=os.path.join(HERE, 'stage_a_grad.json'))
    a = ap.parse_args()
    pool = common.dataset(600, seed0=0, split='train')
    train_eps = [e for e in pool if e['length'] in (2, 4, 6)][:12]
    program, registry, signals = scaffolds.stage_a()
    ex = examples(train_eps)
    arms = {'shipped': {}, 'eq256': {'open': 256.}, 'index0.1': {'byte': .1},
            'both': {'open': 256., 'byte': .1}}
    report = {'space_size': space_size(program), 'positions': len(ex),
              'reference_selection': REFERENCE, 'arms': {}}
    for name, taus in arms.items():
        runs = []
        t0 = time.perf_counter()
        for s in range(a.seeds):
            model, sel, loss, grads = fit(program, registry, signals, ex, taus, s, steps=a.steps)
            g = grads['open'].abs()
            runs.append({'seed': s, 'loss': loss, 'selection': sel,
                         'matches_reference': sel == REFERENCE,
                         'open_byte': sel['open'], 'base': sel['base'],
                         'exact_train_accuracy': exact_accuracy(program, sel, registry, ex),
                         'eq_candidates_with_zero_gradient': int((g == 0).sum()),
                         'eq_candidates_total': int(g.numel()),
                         'max_eq_choice_gradient': float(g.max()),
                         'base_choice_gradient_max': float(grads['base'].abs().max())})
        report['arms'][name] = {'temperatures': taus, 'seconds': time.perf_counter() - t0,
                                'conformant': sum(r['matches_reference'] for r in runs),
                                'exact_train_accuracy_mean': sum(r['exact_train_accuracy'] for r in runs) / len(runs),
                                'runs': runs}
        print(name, report['arms'][name]['conformant'], '/', a.seeds,
              'acc', round(report['arms'][name]['exact_train_accuracy_mean'], 3),
              'zero-grad eq candidates', runs[0]['eq_candidates_with_zero_gradient'],
              'bytes chosen', sorted({r['open_byte'] for r in runs}))
    json.dump(report, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
