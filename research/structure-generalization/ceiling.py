"""Analytic ceiling for any input-independent gate choice.

The record scaffold's `relation` node picks ONE of 16 truth tables with a global
parameter; the choice cannot depend on the episode.  So for a set of gate
families T the best attainable accuracy is max_g mean_{t in T} mean_i [g_i = t_i].
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
AFFINE = (0, 3, 5, 6, 9, 10, 12, 15)
NONAFFINE = (1, 2, 4, 7, 8, 11, 13, 14)

def bits(t): return [(t >> i) & 1 for i in range(4)]

def ceiling(tables):
    best = max(range(16), key=lambda g: sum(sum(a == b for a, b in zip(bits(g), bits(t))) for t in tables))
    acc = max(sum(sum(a == b for a, b in zip(bits(g), bits(t))) for t in tables) for g in range(16)) / (4 * len(tables))
    per_index = [sum(bits(t)[i] for t in tables) / len(tables) for i in range(4)]
    return {'best_fixed_table': best, 'best_accuracy': acc, 'max_return_horizon4': 4 * acc,
            'fraction_true_per_input': per_index}

out = {'affine_set': list(AFFINE), 'nonaffine_set': list(NONAFFINE),
       'affine': ceiling(AFFINE), 'nonaffine': ceiling(NONAFFINE),
       'all16': ceiling(range(16)), 'single_table_6_xor': ceiling([6])}
print(json.dumps(out, indent=2))
Path(__file__).resolve().parent.joinpath('out', 'ceiling.json').write_text(json.dumps(out, indent=2))
