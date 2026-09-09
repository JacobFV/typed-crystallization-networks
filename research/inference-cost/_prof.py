import sys,time,cProfile,pstats,io
R='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,R); sys.path.insert(0,R+'/research/visual-ladder')
from common import FLAT,Registry,bytes_type,episode,load
from tcn.types import Value
import rung3_widgets as Rw
found=load('rung3'); registry=Registry()
probe=episode(0,'train',**FLAT); W,H=probe['width'],probe['height']; offsets=Rw.offset_pool(W)
p0=Rw.same_scaffold(registry,W,H); m0=registry.register_module(p0.harden(found['s0']['chosen']))
p1=Rw.corner_scaffold(registry,W,H,m0,offsets); m1=registry.register_module(p1.harden(found['s1']['chosen']))
p2=Rw.rect_scaffold(registry,W,H,m0,offsets); m2=registry.register_module(p2.harden(found['s2']['chosen']))
print('module nodes: same',len(registry.modules[m0].nodes),'corner',len(registry.modules[m1].nodes),'rect',len(registry.modules[m2].nodes))
ep=episode(200,'test',**FLAT); BT=bytes_type(ep['width'],ep['height'])
parser=Rw.assembly(registry,BT,Rw.interior_positions(ep),m1,m2)
v=Value.of(BT,ep['pixels'])
pr=cProfile.Profile(); pr.enable(); parser.run({'observation':v},registry=registry); pr.disable()
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('tottime').print_stats(18); print(s.getvalue())
