"""Why the exported `.pyz` is slower than the same program in process.

The `.pyz` rebuilds every `Type` with `Type.from_dict`, so two structurally equal
types are no longer the same object. `Registry.exact` opens with
`tuple(v.type for v in args) != op.inputs`, and `Program.execute` looks types up
in dicts, so on a reconstructed artifact those become *deep* structural compares
of a 3,072-element tuple type instead of identity hits.

This isolates that effect from the interpreter version: both arms run under the
repository's own Python 3.13, one on the live objects and one on the artifact
reloaded through `load_program`, on the identical input, with the outputs
checked identical.
"""
import sys, json, time
ROOT = '/home/brandonin/Documents/typed-crystallization-networks'
for p in (ROOT, ROOT + '/research/visual-ladder', ROOT + '/research/inference-cost'):
    sys.path.insert(0, p)
from harness import timed, dump, OUT
from common import FLAT, Registry, bytes_type, episode, load
from tcn.types import Value
from tcn.runtime import save_program, load_program
import rung3_widgets as R

found = load('rung3'); registry = Registry()
probe = episode(0, 'train', **FLAT); W, H = probe['width'], probe['height']
offsets = R.offset_pool(W)
p0 = R.same_scaffold(registry, W, H); m0 = registry.register_module(p0.harden(found['s0']['chosen']))
p1 = R.corner_scaffold(registry, W, H, m0, offsets); m1 = registry.register_module(p1.harden(found['s1']['chosen']))
p2 = R.rect_scaffold(registry, W, H, m0, offsets); m2 = registry.register_module(p2.harden(found['s2']['chosen']))
ep = episode(200, 'test', **FLAT); BT = bytes_type(ep['width'], ep['height'])
live = R.assembly(registry, BT, R.interior_positions(ep), m1, m2)
value = Value.of(BT, ep['pixels'])

live_ms, live_min, a = timed(lambda: set(live.run({'observation': value}, registry=registry)[0]['mapped'].decoded), repeats=1)
path = OUT / 'visual_reload.json'
save_ms, _, _ = timed(lambda: save_program(live, path, registry), repeats=1)
load_ms, load_min, pair = timed(lambda: load_program(path), repeats=3)
reloaded, reg2 = pair
value2 = Value.of(dict(reloaded.inputs)['observation'], ep['pixels'])
re_ms, re_min, b = timed(lambda: set(reloaded.run({'observation': value2}, registry=reg2)[0]['mapped'].decoded), repeats=1)

out = {'note': 'both arms under the repository Python 3.13; only object identity of Types differs',
       'live_objects_ms': live_ms, 'reloaded_from_json_ms': re_ms,
       'slowdown': re_ms / live_ms, 'load_program_ms': load_ms,
       'artifact_bytes': path.stat().st_size,
       'outputs_identical': a == b, 'rects': len(a)}
print(json.dumps(out, indent=1))
dump('reload_cost', out)
