"""Is the interpreter cost a fundamental limit or a caching bug? Monkeypatched, not modified."""
import sys,time
R='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,R); sys.path.insert(0,R+'/research/visual-ladder')
from common import FLAT,Registry,bytes_type,episode,load
import tcn.types as T
from tcn.types import Value
import rung3_widgets as Rw

def build():
    found=load('rung3'); registry=Registry()
    probe=episode(0,'train',**FLAT); W,H=probe['width'],probe['height']; offsets=Rw.offset_pool(W)
    p0=Rw.same_scaffold(registry,W,H); m0=registry.register_module(p0.harden(found['s0']['chosen']))
    p1=Rw.corner_scaffold(registry,W,H,m0,offsets); m1=registry.register_module(p1.harden(found['s1']['chosen']))
    p2=Rw.rect_scaffold(registry,W,H,m0,offsets); m2=registry.register_module(p2.harden(found['s2']['chosen']))
    return registry,m1,m2
registry,m1,m2=build()
ep=episode(200,'test',**FLAT); BT=bytes_type(ep['width'],ep['height'])
parser=Rw.assembly(registry,BT,Rw.interior_positions(ep),m1,m2)
v=Value.of(BT,ep['pixels'])
t0=time.perf_counter(); base=sorted(parser.run({'observation':v},registry=registry)[0]['mapped'].decoded); t_base=time.perf_counter()-t0
print('baseline run',round(t_base,3),'rects',len(base))

# --- Patch 1: memoize Value.decoded (the value is frozen; decode is pure) ---
def decoded(self):
    try: return object.__getattribute__(self,'_dec')
    except AttributeError: pass
    d=T.decode(self.type,self.raw); object.__setattr__(self,'_dec',d); return d
Value.decoded=property(decoded)
t0=time.perf_counter(); a=sorted(parser.run({'observation':v},registry=registry)[0]['mapped'].decoded); t_dec=time.perf_counter()-t0
print('memoized decode',round(t_dec,3),'identical',a==base)

# --- Patch 2: also skip re-validating carriers produced by the interpreter itself ---
T.Value.__post_init__=lambda self: None
t0=time.perf_counter(); b=sorted(parser.run({'observation':v},registry=registry)[0]['mapped'].decoded); t_both=time.perf_counter()-t0
print('memoized + no revalidate',round(t_both,3),'identical',b==base)
import json;json.dump({'baseline_s':t_base,'memoized_decode_s':t_dec,'memoized_plus_novalidate_s':t_both,
  'identical_outputs':bool(a==base and b==base),'rects':len(base)},open(R+'/research/inference-cost/out/fixable.json','w'),indent=1)
