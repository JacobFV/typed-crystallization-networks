"""Does the steepest-descent direction pick the reference candidate at all?

The address-wall track measured this as the sharp form of the question: at
initialization, where does -dL/dlogit rank the reference candidate against the
chance rate of a uniform pick? Reported here for the value choice (which byte
denotes an opening bracket, 256 candidates) and the address choice (where the
symbol field starts, 41 candidates), on the same forward pass.
"""
from __future__ import annotations
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import common, scaffolds
from run_stage_a import examples
from tcn.learning import SoftProgram, tensor

REF = {'base': 14, 'open': 40}

def measure(taus, seeds=8):
    pool = common.dataset(600, seed0=0, split='train')
    train = [e for e in pool if e['length'] in (2, 4, 6)][:12]
    program, registry, signals = scaffolds.stage_a()
    ex = examples(train)
    inputs = {k: torch.stack([tensor(e['inputs'][k]) for e in ex]) for k, _ in program.inputs}
    targets = {'open': torch.stack([tensor(e['targets']['open']) for e in ex])}
    picked = {'base': 0, 'open': 0}; ranks = {'base': [], 'open': []}
    for s in range(seeds):
        torch.manual_seed(s); torch.set_num_threads(1)
        model = SoftProgram(program, registry)
        with torch.no_grad():
            for p in model.choices: p.add_(torch.randn_like(p) * .01)
        model.temperatures.update(taus)
        _, _, trace = model(inputs, return_trace=True)
        model.probe_loss(trace, targets, signals).backward()
        for i, n in enumerate(program.nodes):
            if n.name not in REF: continue
            g = -model.choices[i].grad.detach()          # steepest-descent direction
            order = torch.argsort(g, descending=True).tolist()
            r = order.index(REF[n.name])
            ranks[n.name].append(r); picked[n.name] += int(r == 0)
    return {k: {'picked_reference': picked[k] / seeds, 'chance': 1 / len(program.nodes[[n.name for n in program.nodes].index(k)].candidates),
                'mean_rank': sum(ranks[k]) / seeds, 'candidates': len(program.nodes[[n.name for n in program.nodes].index(k)].candidates)}
            for k in REF}

if __name__ == '__main__':
    out = {name: measure(taus) for name, taus in
           (('shipped', {}), ('eq256', {'open': 256.}), ('index0.1', {'byte': .1}),
            ('both', {'open': 256., 'byte': .1}))}
    json.dump(out, open(os.path.join(HERE, 'descent_direction.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
