"""Declarative DAG orchestration with resumable, evidence-gated stages."""
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import multiprocessing
from .generation import source_fingerprint

def execute_stage(runner,stage,out):
    result=runner(stage,out/stage.name)
    failures=[]
    for metric,limits in stage.gates.items():
        if metric not in result:failures.append(metric+' missing');continue
        if 'min' in limits and result[metric]<limits['min']:failures.append(metric+' below minimum')
        if 'max' in limits and result[metric]>limits['max']:failures.append(metric+' above maximum')
    return {'status':'failed' if failures else 'passed','metrics':result,'failures':failures}

@dataclass(frozen=True)
class Stage:
    name: str
    requires: tuple[str,...]
    operation: str
    configuration: dict
    gates: dict

class Curriculum:
    def __init__(self,stages):
        self.stages={s.name:s for s in stages}
        if len(self.stages)!=len(stages):raise ValueError('duplicate stage')
        for s in stages:
            if set(s.requires)-self.stages.keys():raise ValueError('unknown prerequisite')
        self.order=self._order()
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
        return cls([Stage(s['name'],tuple(s.get('requires',())),s['operation'],s.get('configuration',{}),s.get('gates',{})) for s in d['stages']])
    def run(self,runner,outdir,workers=1,resume=True):
        if workers<1:raise ValueError('positive worker count required')
        out=Path(outdir);out.mkdir(parents=True,exist_ok=True);journal=out/'curriculum.json'
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
                    running[pool.submit(execute_stage,runner,self.stages[name],out)]=name;pending.remove(name)
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
