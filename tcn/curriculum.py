"""Declarative DAG orchestration with resumable, evidence-gated stages."""
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import multiprocessing
from .generation import source_fingerprint

@dataclass(frozen=True)
class Artifacts:
    """What a stage inherits, and what it is expected to leave behind.

    Prerequisites already form a DAG; this is what flows along its edges. A
    stage sees exactly the library modules published by the prior stages it
    named in `inherits` -- never the whole library, never its prerequisites'
    prerequisites unless it names them too. `modules` are library references a
    runner passes straight to `Library.load` to obtain candidate operators.
    """
    library: str
    modules: tuple[str,...] = ()
    publishes: tuple[str,...] = ()
    inherited_from: tuple[str,...] = ()

def execute_stage(runner,stage,out,artifacts):
    result=runner(stage,out/stage.name,artifacts)
    failures=[]
    for metric,limits in stage.gates.items():
        if metric not in result:failures.append(metric+' missing');continue
        if 'min' in limits and result[metric]<limits['min']:failures.append(metric+' below minimum')
        if 'max' in limits and result[metric]>limits['max']:failures.append(metric+' above maximum')
    # Publication is an evidence gate like any other: a stage that declares it
    # supplies a module to later stages has not passed until the module is in
    # the library and loadable.
    published=[]
    if stage.publishes:
        from .library import Library
        library=Library(artifacts.library)
        for name in stage.publishes:
            try:published.append(library.head(name).reference)
            except KeyError:failures.append('module '+name+' not published')
    return {'status':'failed' if failures else 'passed','metrics':result,'failures':failures,
            'published':published,'inherited':list(artifacts.modules)}

@dataclass(frozen=True)
class Stage:
    name: str
    requires: tuple[str,...]
    operation: str
    configuration: dict
    gates: dict
    inherits: tuple[str,...] = ()
    publishes: tuple[str,...] = ()

class Curriculum:
    def __init__(self,stages):
        self.stages={s.name:s for s in stages}
        if len(self.stages)!=len(stages):raise ValueError('duplicate stage')
        for s in stages:
            if set(s.requires)-self.stages.keys():raise ValueError('unknown prerequisite')
            # Inheritance is explicit configuration and is never implied: a stage
            # may only consume artifacts from a prerequisite it has named twice,
            # once as an ordering constraint and once as an artifact source.
            if set(s.inherits)-set(s.requires):raise ValueError('inherits names a stage that is not a prerequisite')
        for s in stages:
            for source in s.inherits:
                if not self.stages[source].publishes:raise ValueError('stage '+source+' publishes nothing to inherit')
        self.order=self._order()
    def inherited(self,name):
        """Library references stage `name` may bind as candidate operators."""
        out=[]
        for source in self.stages[name].inherits:
            for module in self.stages[source].publishes:
                if module not in out:out.append(module)
        return tuple(out)
    def _order(self):
        done=[]
        while len(done)<len(self.stages):
            ready=sorted(k for k,s in self.stages.items() if k not in done and set(s.requires)<=set(done))
            if not ready:raise ValueError('curriculum dependency cycle')
            done.extend(ready)
        return done
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if d['format']!='tcn.curriculum/1':raise ValueError('curriculum format mismatch')
        return cls([Stage(s['name'],tuple(s.get('requires',())),s['operation'],s.get('configuration',{}),s.get('gates',{}),tuple(s.get('inherits',())),tuple(s.get('publishes',()))) for s in d['stages']])
    def run(self,runner,outdir,workers=1,resume=True,library=None):
        if workers<1:raise ValueError('positive worker count required')
        out=Path(outdir);out.mkdir(parents=True,exist_ok=True);journal=out/'curriculum.json'
        # One library per run directory unless the caller names a shared one, so
        # a curriculum leaves behind a growing artifact rather than nothing.
        library=Path(library) if library else out/'library'
        if any(s.publishes for s in self.stages.values()) and workers>1:
            raise ValueError('publishing stages need workers=1: one manifest, one writer')
        fingerprint=hashlib.sha256(json.dumps({'source':source_fingerprint(),'stages':[asdict(self.stages[k]) for k in self.order]},sort_keys=True).encode()).hexdigest()
        records={}
        if resume and journal.exists():
            previous=json.loads(journal.read_text())
            if previous['fingerprint']!=fingerprint:raise ValueError('curriculum changed: choose a new run directory')
            records=previous['stages']
        def save():
            tmp=journal.with_suffix('.tmp');tmp.write_text(json.dumps({'fingerprint':fingerprint,'stages':records},indent=2,sort_keys=True));tmp.replace(journal)
        pending=set(self.order)-{k for k,v in records.items() if v['status']=='passed'}
        running={}
        pool_class=ThreadPoolExecutor if workers==1 else ProcessPoolExecutor
        kwargs={} if workers==1 else {'mp_context':multiprocessing.get_context('spawn')}
        with pool_class(max_workers=workers,**kwargs) as pool:
            while pending or running:
                passed={k for k,v in records.items() if v['status']=='passed'}
                ready=sorted(k for k in pending if set(self.stages[k].requires)<=passed)
                for name in ready[:max(0,workers-len(running))]:
                    stage=self.stages[name]
                    artifacts=Artifacts(str(library),self.inherited(name),stage.publishes,stage.inherits)
                    running[pool.submit(execute_stage,runner,stage,out,artifacts)]=name;pending.remove(name)
                if not running:
                    for name in pending:records[name]={'status':'blocked','failures':['prerequisite failed']}
                    break
                finished,_=wait(running,return_when=FIRST_COMPLETED)
                for future in finished:
                    name=running.pop(future)
                    try:records[name]=future.result()
                    except Exception as e:records[name]={'status':'failed','failures':[str(e)]}
                    save()
        save();return records
