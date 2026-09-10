import argparse, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--max-nodes', type=int, default=3)
ap.add_argument('--rel', default='Sstar')
a = ap.parse_args()
o = json.loads((HERE / 'out' / f'shared_{a.max_nodes}.json').read_text())
for r in o['cross_domain_' + a.rel]:
    print('==', '+'.join(r['domains']), 'all_trivial', r['all_trivial'], 'key', r['key'][:12])
    for m in r['members']:
        print(f"   {m['digest'][:12]} nodes={m['nodes']} arity={m['arity']} "
              f"trivial={m['trivial']} ({m['trivial_reason']})")
        print(f"      {m['spell']}")
        print(f"      {m['in_types']} -> {m['out_type']}  occ={m['occurrences']}")
