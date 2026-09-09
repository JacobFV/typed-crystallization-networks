import sys,time,pathlib
R='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,R); sys.path.insert(0,R+'/research/visual-ladder')
from common import FLAT,Registry,bytes_type,episode,load
from tcn.types import Value
import rung3_widgets as Rw
t0=time.perf_counter()
found=load('rung3')
registry=Registry()
probe=episode(0,'train',**FLAT)
W,H=probe['width'],probe['height']
offsets=Rw.offset_pool(W)
p0=Rw.same_scaffold(registry,W,H); m0=registry.register_module(p0.harden(found['s0']['chosen']))
p1=Rw.corner_scaffold(registry,W,H,m0,offsets); m1=registry.register_module(p1.harden(found['s1']['chosen']))
p2=Rw.rect_scaffold(registry,W,H,m0,offsets); m2=registry.register_module(p2.harden(found['s2']['chosen']))
print('build',time.perf_counter()-t0)
t0=time.perf_counter(); ep=episode(200,'test',**FLAT); print('episode',time.perf_counter()-t0)
BT=bytes_type(ep['width'],ep['height'])
parser=Rw.assembly(registry,BT,Rw.interior_positions(ep),m1,m2)
print('nodes',len(parser.nodes),'pruned',len(parser.pruned().nodes),'cost',parser.execution_cost(registry))
t0=time.perf_counter(); v=Value.of(BT,ep['pixels']); print('encode',time.perf_counter()-t0)
t0=time.perf_counter(); got=parser.run({'observation':v},registry=registry)[0]; print('run',time.perf_counter()-t0)
t0=time.perf_counter(); d=sorted(got['mapped'].decoded); print('decode',time.perf_counter()-t0,len(d))
