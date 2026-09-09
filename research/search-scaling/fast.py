"""Vectorised equivalent of SoftProgram's forward for boolean truth-table scaffolds.

Same semantics as tcn.learning.relaxed(truth_k) + softmax(logits/tau) mixing, but all
candidates of a node are evaluated as one batched tensor op instead of a Python loop.
`validate_equivalence` checks it against the real SoftProgram step for step.
"""
import math
import torch
from harness import N_IN, POOLS

def compile_program(program):
    ports = {}
    for j in range(N_IN):
        ports[f'b{j}'] = j
    plan = []
    for n in program.nodes:
        ia, ib, coef = [], [], []
        for c in n.candidates:
            ia.append(ports[c.sources[0]]); ib.append(ports[c.sources[1]])
            k = int(c.operator.name[6:])
            coef.append([float((k >> i) & 1) for i in range(4)])
        ports[n.name] = len(ports)
        plan.append((torch.tensor(ia), torch.tensor(ib), torch.tensor(coef)))
    return plan, ports

def forward(plan, logits, taus, inputs):
    """inputs: (N_IN, B) float tensor of 0/1. Returns list of node activations (B,)."""
    vals = [inputs[j] for j in range(N_IN)]
    outs = []
    for (ia, ib, coef), lg, tau in zip(plan, logits, taus):
        V = torch.stack(vals)                    # (P, B)
        a = V[ia]; b = V[ib]                     # (K, B)
        terms = torch.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b])  # (4,K,B)
        y = (coef.T.unsqueeze(-1) * terms).sum(0)                                  # (K,B)
        w = torch.softmax(lg / tau, 0)
        out = (w.unsqueeze(-1) * y).sum(0)
        vals.append(out); outs.append(out)
    return outs

def entropy(logits, taus):
    tot = 0.
    for lg, tau in zip(logits, taus):
        q = torch.softmax(lg / tau, 0)
        tot = tot + -(q * q.clamp_min(1e-12).log()).sum()
    return tot

def per_node_entropy(logits, taus):
    return [float(-(q * q.clamp_min(1e-12).log()).sum())
            for q in (torch.softmax(lg / tau, 0).detach() for lg, tau in zip(logits, taus))]
