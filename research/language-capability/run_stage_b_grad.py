"""Stage B by gradient descent, on the identical space the sweep enumerates.

The frozen stage-A module is a `gradient="none"` boundary, so its nodes execute
exactly; everything the search chooses here (`sub`, `identity`, `mux`, `lt`,
`eq`/`ge`/`le` over a signed integer carrier) is differentiable. Initialization
noise is added explicitly because `SoftProgram` zero-initializes logits.
"""
from __future__ import annotations
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import common, scaffolds
from run_stage_b import build_module, examples, accuracy, baselines
from tcn.learning import SoftProgram, tensor
from tcn.search import space_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=6); ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--n-train', type=int, default=24); ap.add_argument('--lr', type=float, default=.05)
    ap.add_argument('--out', default=os.path.join(HERE, 'stage_b_grad.json'))
    ap.add_argument('--tau-lt', type=float, default=0.)   # widen the `lt` surrogate
    ap.add_argument('--tau-eq', type=float, default=0.)   # widen the answer-rule `eq`
    a = ap.parse_args()
    module, registry, _ = build_module()
    program, signals = scaffolds.stage_b(module, registry)
    pool = common.dataset(900, seed0=0, split='train')
    train_eps = [e for e in pool if e['length'] in (2, 4, 6)][:a.n_train]
    seen_held = [e for e in pool if e['length'] in (2, 4, 6)][a.n_train:a.n_train + 120]
    test_pool = common.dataset(1500, seed0=100000, split='test')
    unseen = [e for e in test_pool if e['length'] not in (2, 4, 6)]
    ex = examples(train_eps)
    inputs = {'text': torch.stack([tensor(e['inputs']['text']) for e in ex])}
    targets = {'answer': torch.stack([tensor(e['targets']['answer']) for e in ex])}
    runs = []
    for s in range(a.seeds):
        torch.manual_seed(s); torch.set_num_threads(1)
        model = SoftProgram(program, registry)
        with torch.no_grad():
            for p in model.choices: p.add_(torch.randn_like(p) * .01)
        if a.tau_lt: model.temperatures.update({f'in{i}': a.tau_lt for i in range(16)})
        if a.tau_eq: model.temperatures['answer'] = a.tau_eq
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        t0 = time.perf_counter(); first_grads = None
        for step in range(a.steps):
            opt.zero_grad()
            _, _, trace = model(inputs, return_trace=True)
            loss = model.probe_loss(trace, targets, signals)
            loss.backward()
            if step == 0:
                first_grads = {k: float(model.choices[i].grad.abs().max())
                               for i, k in enumerate(n.name for n in program.nodes)
                               if k in ('symbols', 'plus', 'minus', 'answer')}
            opt.step()
        sel = model.selections()
        tr_acc, _ = accuracy(program, sel, registry, train_eps)
        sn_acc, _ = accuracy(program, sel, registry, seen_held)
        un_acc, per_len = accuracy(program, sel, registry, unseen)
        runs.append({'seed': s, 'loss': float(loss.detach()), 'seconds': time.perf_counter() - t0,
                     'selection': {k: sel[k] for k in ('symbols', 'plus', 'minus', 'answer')},
                     'chosen': {'symbols_sub': program.nodes[1].candidates[sel['symbols']].sources[1],
                                'plus': program.nodes[2].candidates[sel['plus']].sources[0],
                                'minus': program.nodes[3].candidates[sel['minus']].sources[0],
                                'rule': [program.nodes[-1].candidates[sel['answer']].operator.name,
                                         program.nodes[-1].candidates[sel['answer']].sources[1]]},
                     'exact_train_accuracy': tr_acc, 'heldout_seen_lengths': sn_acc,
                     'heldout_unseen_lengths': un_acc, 'per_length': per_len,
                     'first_step_choice_gradients': first_grads})
        print(runs[-1]['seed'], runs[-1]['chosen'], round(tr_acc, 3), round(sn_acc, 3), round(un_acc, 3),
              first_grads, f"{runs[-1]['seconds']:.0f}s")
    report = {'space_size': space_size(program), 'steps': a.steps, 'seeds': a.seeds,
              'tau_lt': a.tau_lt, 'tau_eq': a.tau_eq,
              'train_episodes': len(train_eps),
              'exact_train_conformant': sum(r['exact_train_accuracy'] == 1.0 for r in runs),
              'mean_heldout_unseen': sum(r['heldout_unseen_lengths'] for r in runs) / len(runs),
              'baselines': {'heldout_seen_lengths': baselines(seen_held),
                            'heldout_unseen_lengths': baselines(unseen)}, 'runs': runs}
    json.dump(report, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=1))

if __name__ == '__main__':
    main()
