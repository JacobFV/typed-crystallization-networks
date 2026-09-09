"""The dense supervision comes from the gated probe channel, not from the harness.

`search_run.py` labels its examples from the configuration this script chose. That is
ordinary, but it does not show that the *generator* can supply the signal, which is
what ARCHITECTURE section 7 asks of a supervision interface and what `JointTrainer`
requires (its `Target` can only read `probes`, `latent_states` or `observations`).

Here every target is read out of `StepRecord.probes` with the gated channel on, the
transform search is re-run against those targets, and the result is compared with the
harness-labelled run. The same comparison is attempted from the shipped
`state_counts` probe, which cannot supply it.
"""
import json,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability')
from tcn.generation import read_text
from tcn.operators import Registry
from tcn.search import enumerate_fit,space_size
from tcn.types import Value
import program as P
import task as T
import search_run as S

BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'
PROBE={'root':'/home/agent','depth':1,'filesystem_capacity':24,'process_capacity':8,
       'content_capacity':64,'content_paths':[T.TASK_PATH]}

def collect(documents):
    """Run the reference trajectory with the probe on and label from the probe alone."""
    examples=[];audit=[]
    for name,digit in documents:
        host=T.host(name,digit,3,probe=PROBE)
        for action in (T.read_action(),T.write_action(T.target(digit)),T.wait_action()):
            host.step((action,))
        # tick 1 is the decision point for the write; tick 2 is the state it produced.
        observation=host.records[1].observations['terminal'].value
        after=read_text(host.records[2].probes['content_0'])
        before=read_text(host.records[1].probes['content_0'])
        counts_before=host.records[1].probes['state_counts'].decoded
        counts_after=host.records[2].probes['state_counts'].decoded
        examples.append({'inputs':{'terminal':observation},
                         'targets':{'reference_byte':Value.of(P.U8,ord(after[-1]))},
                         'label':before})
        audit.append({'document':T.document(name,digit),'probe_content_before':before,
                      'probe_content_after':after,'derived_target_byte':ord(after[-1]),
                      'state_counts_before':counts_before,'state_counts_after':counts_after,
                      'goal_reached_after':host.records[2].probes['goal_reached'].decoded})
    return examples,audit

if __name__=='__main__':
    started=time.perf_counter()
    train,audit=collect(T.TRAIN_DOCUMENTS)
    test,audit_test=collect(T.TEST_DOCUMENTS[:5])
    registry=Registry()
    program=P.transform_program(registry,encode=False)
    found=enumerate_fit(program,train,P.signals(),registry,tolerance=.001,rank='description')
    harness=json.loads(open(BASE+'search.json').read())['transform']['enumeration']
    report={'probe_labelled_examples':len(train),'audit':audit,'test_audit':audit_test,
            'space_size':space_size(program),
            'enumeration':found.to_dict(),
            'agrees_with_harness_labelled_run':found.selections==harness['selections'],
            'harness_selections':harness['selections']}
    if found.solved:
        report['test_accuracy']=S.accuracy(program,found.selections,test,P.signals(),registry)
    # What the shipped probe could have supplied instead.
    report['state_counts_as_a_target']={
        'distinct_series_over_%d_documents'%len(audit):len({json.dumps([a['state_counts_before'],a['state_counts_after']]) for a in audit}),
        'distinct_probe_content_series':len({json.dumps([a['probe_content_before'],a['probe_content_after']]) for a in audit}),
        'conclusion':'state_counts takes one value across every document, so no target derived '
                     'from it can depend on the file the agent must read'}
    report['seconds']=time.perf_counter()-started
    open(BASE+'probe_supervision.json','w').write(json.dumps(report,indent=2,sort_keys=True,default=str))
    print(json.dumps({k:v for k,v in report.items() if k not in {'audit','test_audit'}},indent=2,sort_keys=True,default=str))
