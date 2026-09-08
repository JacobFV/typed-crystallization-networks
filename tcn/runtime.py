"""Portable, immutable program artifacts and independent exact execution bundles."""
from pathlib import Path
import json
import shutil
import tempfile
import time
import statistics
import zipapp
from .graph import Program
from .operators import Registry

def save_program(program,path,registry=None):
    if not program.is_frozen:raise ValueError('export requires a fully crystallized program')
    r=registry or Registry();program.validate(r)
    # Dependency order is insertion order: a module can only reference previously
    # registered modules. The loader checks every digest and type signature.
    d={'format':'tcn.artifact/1','digest':program.digest,'program':program.to_dict(),'modules':[m.to_dict() for m in r.modules.values()]}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,sort_keys=True,indent=2))
    return path

def load_program(path):
    d=json.loads(Path(path).read_text());r=Registry()
    if d['format']!='tcn.artifact/1':raise ValueError('artifact version mismatch')
    for spec in d['modules']:r.register_module(Program.from_dict(spec,r))
    p=Program.from_dict(d['program'],r)
    if not p.is_frozen or p.digest!=d['digest']:raise ValueError('artifact integrity failure')
    return p,r

def export_executable(program,path,registry=None):
    """Standard-library-only zip application; no torch or project install needed."""
    with tempfile.TemporaryDirectory(prefix='tcn-export-') as tmp:
        root=Path(tmp);pkg=root/'tcn';pkg.mkdir();source=Path(__file__).parent
        for name in ['types.py','operators.py','graph.py']:shutil.copyfile(source/name,pkg/name)
        (pkg/'__init__.py').write_text('')
        save_program(program,root/'program.json',registry)
        (root/'__main__.py').write_text('''import json,sys,zipfile
from tcn.graph import Program
from tcn.operators import Registry
from tcn.types import Value
with zipfile.ZipFile(sys.argv[0]) as z: artifact=json.loads(z.read('program.json'))
r=Registry()
for spec in artifact['modules']:r.register_module(Program.from_dict(spec,r))
p=Program.from_dict(artifact['program'],r)
if p.digest!=artifact['digest']:raise ValueError('artifact integrity failure')
state=None
for line in sys.stdin:
    row=json.loads(line)
    if row.get('reset'):state=None
    if 'state' in row:state={k:Value.from_dict(v) for k,v in row['state'].items()}
    inputs={k:Value.from_dict(v) for k,v in row['inputs'].items()}
    out,state=p.run(inputs,state,r)
    print(json.dumps({'outputs':{k:v.to_dict() for k,v in out.items()},'state':{k:v.to_dict() for k,v in state.items()}},sort_keys=True),flush=True)
''')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);zipapp.create_archive(root,path,interpreter='/usr/bin/env python3')
    return path

def benchmark(program,examples,registry=None,repetitions=100):
    samples=[]
    for i in range(repetitions):
        inputs=examples[i%len(examples)];start=time.perf_counter_ns();program.run(inputs,registry=registry);samples.append((time.perf_counter_ns()-start)/1e6)
    samples.sort()
    return {'scope':'exact model, typed input through typed output, batch one','repetitions':repetitions,'p50_ms':statistics.median(samples),'p95_ms':samples[min(len(samples)-1,int(.95*len(samples)))],'description_bits':program.description_bits(registry),'estimated_operator_cost':program.execution_cost(registry)}
