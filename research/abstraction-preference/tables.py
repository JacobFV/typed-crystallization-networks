"""Summary tables over the arm runs, plus a two-sided Fisher exact test."""
import json, math, statistics, sys


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    def logfact(n): return math.lgamma(n + 1)
    n = a + b + c + d
    def p(a_, b_, c_, d_):
        return math.exp(logfact(a_ + b_) + logfact(c_ + d_) + logfact(a_ + c_) + logfact(b_ + d_)
                        - logfact(n) - logfact(a_) - logfact(b_) - logfact(c_) - logfact(d_))
    obs = p(a, b, c, d); total = 0.
    for i in range(0, a + b + 1):
        j = a + b - i; k = a + c - i; l = c + d - k
        if k < 0 or l < 0 or j < 0: continue
        q = p(i, j, k, l)
        if q <= obs * (1 + 1e-9): total += q
    return min(1., total)


def summarize(path):
    d = json.load(open(path))
    rows = []
    for w in d['weights']:
        got = [x for x in d['runs'] if x['mdl_weight'] == w]
        ok = [x for x in got if x['found']]
        end = [x for x in got if x.get('final_conformant')]
        row = {'weight': w, 'n': len(got), 'conformant': len(ok)}
        if ok:
            row |= {'median_step': statistics.median(x['first_conformant_step'] for x in ok),
                    'median_live': statistics.median(x['found']['live_nodes'] for x in ok),
                    'median_calls': statistics.median(x['found']['module_calls'] for x in ok),
                    'median_bits': statistics.median(x['found']['description_bits'] for x in ok),
                    'median_cost': statistics.median(x['found']['execution_cost'] for x in ok),
                    'minimal_programs': sum(1 for x in ok if x['found']['live_nodes'] == 3),
                    'redundant_programs': sum(1 for x in ok if x['found']['live_nodes'] > 3)}
        if end:
            row |= {'exact_at_end': len(end),
                    'median_live_at_end': statistics.median(x['final']['live_nodes'] for x in end),
                    'median_bits_at_end': statistics.median(x['final']['description_bits'] for x in end)}
        rows.append(row)
    base = rows[0]
    for row in rows[1:]:
        row['fisher_vs_w0'] = round(fisher(row['conformant'], row['n'] - row['conformant'],
                                           base['conformant'], base['n'] - base['conformant']), 4)
    return {'scaffold': d['scaffold'], 'arm': d['arm'], 'candidates': d['candidates'], 'rows': rows}


out = {}
for path in sys.argv[1:]:
    out[path] = summarize(path)
    print('==', path, out[path]['scaffold'], 'arm', out[path]['arm'])
    for row in out[path]['rows']:
        print('  ', json.dumps(row))
json.dump(out, open('tables.json', 'w'), indent=1)
