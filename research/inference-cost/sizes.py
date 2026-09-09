"""Where the bytes in an exported artifact actually are.

`save_program` writes `json.dumps(..., indent=2)` and `export_executable` calls
`zipapp.create_archive` with its default `compressed=False`, so a `.pyz` STORES
that JSON rather than deflating it. This script decomposes the JSON by role so
the report can say, per artifact, how many bytes are type declarations, how many
are node/operator declarations, how many are constant payloads, and how many are
JSON indentation -- and what the same content costs deflated.

The decomposition is exact, not sampled: every sub-object is re-serialized and
its own `len(json.dumps(...))` charged to its category, with the categories
disjoint and summing to within the punctuation that joins them.
"""
import gzip, json, sys, zipfile, pathlib
ROOT = '/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/research/inference-cost')
from harness import dump, OUT

J = lambda o: len(json.dumps(o, sort_keys=True, separators=(',', ':')).encode())


def split_program(p):
    """Charge every byte of one `Program.to_dict()` to a role."""
    types = sum(J(t) for _, t in p['inputs'])
    nodes = constants = 0
    for n in p['nodes']:
        types += J(n['output'])
        for c in n['candidates']:
            op = c['operator']
            types += sum(J(t) for t in op['inputs']) + J(op['output'])
            nodes += J({k: v for k, v in op.items() if k not in ('inputs', 'output')}) + J(c['sources'])
        nodes += J([n['name'], n['region'], n['depth'], n['selected']])
    for _, v in p['constants']:
        types += J(v['type']); constants += J(v['raw'])
    for _, v, u in p.get('state', ()):
        types += J(v['type']); constants += J(v['raw']); nodes += J(u)
    other = J({k: v for k, v in p.items() if k not in ('inputs', 'nodes', 'constants', 'state')})
    return {'type_declaration_bytes': types, 'node_and_operator_bytes': nodes,
            'constant_payload_bytes': constants, 'other_bytes': other}


def report(name, path):
    path = pathlib.Path(path)
    if not path.exists():
        return {'name': name, 'missing': str(path)}
    with zipfile.ZipFile(path) as z:
        artifact = json.loads(z.read('program.json'))
        members = {i.filename: (i.file_size, i.compress_size) for i in z.infolist()}
    parts = split_program(artifact['program'])
    for m in artifact['modules']:
        for k, v in split_program(m).items():
            parts[k] += v
    minified = json.dumps(artifact, sort_keys=True, separators=(',', ':')).encode()
    pretty = json.dumps(artifact, sort_keys=True, indent=2).encode()
    deflated = gzip.compress(minified, 9)
    charged = sum(parts.values())
    return {'name': name,
            'pyz_bytes': path.stat().st_size,
            'pyz_stored_uncompressed': all(c == u for u, c in members.values()) or
                                       members.get('program.json', (0, 0))[0] == members.get('program.json', (0, 1))[1],
            'program_json_in_pyz_bytes': members.get('program.json', (0, 0))[0],
            'code_bytes_in_pyz': sum(u for k, (u, c) in members.items() if k != 'program.json'),
            'json_pretty_bytes': len(pretty),
            'json_minified_bytes': len(minified),
            'json_gzip_bytes': len(deflated),
            'indent_overhead_bytes': len(pretty) - len(minified),
            'roles_minified': parts,
            'roles_charged_bytes': charged,
            'type_declaration_share_of_minified': parts['type_declaration_bytes'] / len(minified),
            'gzip_ratio_vs_pyz': path.stat().st_size / len(deflated),
            'nodes': len(artifact['program']['nodes']),
            'module_nodes': sum(len(m['nodes']) for m in artifact['modules']),
            'distinct_operators': len({c['operator']['name'] for p in [artifact['program'], *artifact['modules']]
                                       for n in p['nodes'] for c in n['candidates']})}


if __name__ == '__main__':
    rows = [report(n, OUT / f'{n}.pyz') for n in ('mixed', 'language', 'computer', 'visual')]
    for r in rows:
        print(json.dumps(r, indent=1))
    dump('sizes', rows)
