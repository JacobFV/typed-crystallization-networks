"""How stable is each lesson's surface realization, byte by byte?

A typed program reaching text through `index` needs an address. Addresses are
either constants, or computed from the one numeric handle the observation
carries (the length field). Either way the program can only follow content that
sits at a predictable offset. This measures, per lesson, how much of the prompt
is at a predictable offset:

  `positional_agreement`  mean over byte positions of the modal byte's frequency,
                          aligned from the left (constant-address reach)
  `right_aligned`         the same aligned from the right (length-relative reach)
  `distinct_lengths`      how many prompt lengths occur
"""
from __future__ import annotations
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generators.language.engine.registry import lesson_ids, get

def agreement(prompts, right=False):
    m = min(len(p) for p in prompts)
    tot = 0.
    for i in range(m):
        col = collections.Counter(p[-1 - i] if right else p[i] for p in prompts)
        tot += col.most_common(1)[0][1] / len(prompts)
    return tot / max(1, m), m

def main(n=200):
    out = {}
    for lid in lesson_ids(implemented_only=True):
        try:
            ps = [get(lid).example(seed=s).prompt.encode('utf8') for s in range(n)]
        except Exception as e:
            out[lid] = {'error': str(e)}; continue
        left, m = agreement(ps); right, _ = agreement(ps, True)
        pre = 0
        while pre < m and len({p[pre] for p in ps}) == 1: pre += 1
        suf = 0
        while suf < m and len({p[-1 - suf] for p in ps}) == 1: suf += 1
        alpha = len({b for p in ps for b in p[pre:len(p) - suf]})
        out[lid] = {'constant_prefix': pre, 'constant_suffix': suf,
                    'variable_alphabet': alpha,
                    'max_variable_span': max(len(p) - pre - suf for p in ps),
                    'positional_agreement': left, 'right_aligned': right,
                    'common_prefix_len': m, 'distinct_lengths': len({len(p) for p in ps}),
                    'max_bytes': max(len(p) for p in ps)}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realization.json'), 'w'), indent=1)
    rows = sorted(out.items(), key=lambda kv: -max(kv[1].get('positional_agreement', 0), kv[1].get('right_aligned', 0)))
    print(f"{'lesson':34s}{'left':>7}{'right':>7}{'pre':>5}{'suf':>5}{'span':>6}{'alpha':>6}{'maxB':>6}")
    for k, v in rows[:22]:
        print(f"{k:34s}{v['positional_agreement']:7.3f}{v['right_aligned']:7.3f}"
              f"{v['constant_prefix']:5d}{v['constant_suffix']:5d}{v['max_variable_span']:6d}{v['variable_alphabet']:6d}{v['max_bytes']:6d}")
    small = [(k, v) for k, v in out.items() if 'error' not in v and v['variable_alphabet'] <= 4]
    print('lessons whose whole variable region is over <= 4 distinct bytes:', [k for k, _ in small])
    both = [max(v['positional_agreement'], v['right_aligned']) for v in out.values() if 'error' not in v]
    print('lessons with either alignment >= 0.9:', sum(x >= .9 for x in both), 'of', len(both))
    print('>= 0.75:', sum(x >= .75 for x in both))

if __name__ == '__main__':
    main()
