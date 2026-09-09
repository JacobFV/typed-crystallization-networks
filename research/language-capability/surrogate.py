"""Measure the shipped relaxations at this task's operating distances.

`eq`'s surrogate is exp(-(a-b)^2/tau) and `index`'s is a Gaussian kernel over
positions, both at the node temperature (1.0 as shipped). Section 16 records the
`eq` underflow; the `index` kernel is section 16's M3. Both are measured here at
the byte distances this task actually produces: '(' = 40, ')' = 41.
"""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def eq_surrogate(d, tau=1.0):
    a = torch.tensor([[float(d)]], dtype=torch.float32); b = torch.zeros_like(a)
    return float(torch.exp(-((a - b) ** 2).sum(-1) / tau))

def index_kernel(count, addr, tau=1.0):
    b = torch.tensor([float(addr)])
    w = torch.softmax(-(b - torch.arange(count, dtype=torch.float32)) ** 2 / tau, dim=-1)
    return w

if __name__ == '__main__':
    out = {'eq_surrogate_tau1': {str(d): eq_surrogate(d) for d in
                                 [0, 1, 2, 5, 8, 10, 11, 12, 25, 40, 44, 46, 84]},
           'eq_surrogate_tau256': {str(d): eq_surrogate(d, 256.) for d in [1, 11, 40, 44, 84]}}
    w = index_kernel(128, 14)
    out['index_kernel_tau1'] = {'mass_on_addressed': float(w[14]),
                                'mass_on_neighbours_pm1': float(w[13] + w[15]),
                                'mass_on_pm2': float(w[12] + w[16]),
                                'effective_byte_if_field_alternates_40_41':
                                    float((w * torch.tensor([40. if i % 2 == 0 else 41. for i in range(128)])).sum())}
    # the decisive number: can the relaxed pipeline separate '(' from ')' at all?
    v = out['index_kernel_tau1']['mass_on_addressed']
    blur_open = 40 * v + 40.5 * (1 - v)   # neighbours are the other bracket half the time
    blur_close = 41 * v + 40.5 * (1 - v)
    out['separation_after_index_blur'] = {
        'relaxed_byte_when_true_symbol_is_open': blur_open,
        'relaxed_byte_when_true_symbol_is_close': blur_close,
        'eq_vs_40_when_open': eq_surrogate(blur_open - 40),
        'eq_vs_40_when_close': eq_surrogate(blur_close - 40),
        'gap': abs(eq_surrogate(blur_open - 40) - eq_surrogate(blur_close - 40))}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
