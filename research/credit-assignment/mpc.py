"""Model-based arm: enumerate action sequences against a frozen exact model, no policy.

The cheapest approach `research/policy-learning/RESULTS.md` measured (27 environment
episodes, against 400 for reward-only REINFORCE) was this one, so it is the one most
likely to work here. It has two parts and both are named:

* **Learned.** The perception, from `perception.py`: two typed programs found by
  exhaustive enumeration against the `answer` and `showing_task` probes on N
  recorded episodes, then hardened. They turn the raw terminal into `answer` and
  `showing_task`. Nothing else is learned; there is no policy and no reward-driven
  parameter anywhere in this arm.
* **Declared.** The transition and reward model: `dial` sets the register, `commit`
  writes it and ends the episode, `look` at an unseen slot shows the task record
  with probability 1/(unseen), and the answer is uniform on 1..9 until seen. This is
  the same kind of declaration `e6_mpc.py` made when it wrote
  `r(a, z) = [a == (z xor sign)]` by hand. It is a model *of the mechanism*, not of
  the episode: it contains no episode-specific content.

Planning enumerates every sequence over the four abstract actions
`wait | look at the lowest-index unseen slot | dial the current best answer | commit`
to the end of the episode -- 4^H sequences, all of them, scored by expectation under
the model -- and executes the first action of the best. The reduction from "look at
slot k" to "look at an unseen slot" is exact: the model is symmetric in unseen slots.

Run: .venv/bin/python research/credit-assignment/mpc.py [perception_episodes] [eval_episodes]
"""
import itertools,json,statistics,sys,time
from fractions import Fraction as Q
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+'/research/credit-assignment')
from tcn.types import Value
from panel import Counter,look,dial,COMMIT,WAIT,HORIZON
from generators.computer.generator import PANEL_SLOTS,PANEL_DIGITS,DIAL
import perception as P

REGISTERS=1<<DIAL.bits
BLIND_REGISTER=Q(1,REGISTERS);BLIND_DIAL=Q(1,PANEL_DIGITS)
ABSTRACT=('wait','look','dial','commit')


def frozen_perception(counter,episodes):
    """Enumerate, then harden. Returns callables terminal -> answer, terminal -> brand."""
    train_t,train_p=P.collect(counter,range(episodes))
    tp,tr=P.transform_program();pp,pr=P.predicate_program()
    from tcn.graph import Signal
    from tcn.search import enumerate_fit
    from program import F
    from tcn.types import BOOL
    a=enumerate_fit(tp,train_t,(Signal('shift','reference_answer',('perception','transform'),F,'mse'),),tr,tolerance=1e-6,rank='description')
    b=enumerate_fit(pp,train_p,(Signal('brand','reference_brand',('perception',),BOOL,'mse'),),pr,tolerance=1e-6,rank='description')
    if not (a.solved and b.solved):raise RuntimeError('perception enumeration found no conforming program')
    ht=tp.harden(a.selections);hp=pp.harden(b.selections)
    def answer_of(terminal):
        out,_=ht.run({'terminal':terminal},registry=tr);return float(out['shift'].decoded)
    def brand_of(terminal):
        out,_=hp.run({'terminal':terminal},registry=pr);return bool(out['brand'].decoded)
    return answer_of,brand_of,a,b


def sequence_value(sequence,seen,known,dialled):
    """Expected return of one abstract action sequence under the declared model."""
    if not sequence:return Q(0)
    action,rest=sequence[0],sequence[1:]
    if action=='commit':
        return Q(1) if dialled=='answer' else (BLIND_DIAL if dialled=='blind' else BLIND_REGISTER)
    if action=='wait':return sequence_value(rest,seen,known,dialled)
    if action=='dial':return sequence_value(rest,seen,known,'answer' if known else 'blind')
    if seen>=PANEL_SLOTS or known:return sequence_value(rest,seen,known,dialled)
    hit=Q(1,PANEL_SLOTS-seen)
    return hit*sequence_value(rest,seen,True,dialled)+(1-hit)*sequence_value(rest,seen+1,known,dialled)


def plan(remaining,seen,known,dialled):
    best=None;evaluated=0
    for sequence in itertools.product(ABSTRACT,repeat=remaining):
        evaluated+=1
        v=sequence_value(sequence,seen,known,dialled)
        if best is None or v>best[0]:best=(v,sequence)
    return best[1][0],float(best[0]),evaluated


def rollout(counter,answer_of,brand_of,index,horizon=HORIZON,split='test'):
    host=counter.create(index,split=split,horizon=horizon)
    seen=0;known=False;answer=1;dialled=None;total=0.;trace=[];evaluated=0;unseen=list(range(PANEL_SLOTS))
    for t in range(horizon):
        if host.records[-1].done:break
        view=host.view();terminal=view.observations['terminal']
        if brand_of(terminal):known=True;answer=int(round(answer_of(terminal)))
        choice,value,n=plan(horizon-t,seen,known,dialled);evaluated+=n
        if choice=='wait':action=WAIT
        elif choice=='look':
            slot=unseen[0] if unseen else 0;action=look(slot)
        elif choice=='dial':
            action=dial(max(1,min(REGISTERS-1,answer)));dialled='answer' if known else 'blind'
        else:action=COMMIT
        record=counter.step(host,action)
        if choice=='look':
            terminal=record.observations['terminal'].value
            if brand_of(terminal):known=True;answer=int(round(answer_of(terminal)))
            else:
                if unseen:unseen.pop(0)
                seen+=1
        total+=sum(v.decoded for v in record.reward_components.values());trace.append(choice)
        if record.done:break
    return total,trace,evaluated


def main(perception_episodes=25,eval_episodes=64):
    counter=Counter(session='mpc');t0=time.perf_counter()
    answer_of,brand_of,a,b=frozen_perception(counter,perception_episodes)
    after_perception=counter.episodes
    totals=[];traces=[];evaluated=0
    for k in range(eval_episodes):
        total,trace,n=rollout(counter,answer_of,brand_of,10000+k)
        totals.append(total);traces.append(tuple(trace));evaluated+=n
    report={'perception_episodes':after_perception,'eval_episodes':eval_episodes,
            'transform_program':{n.name:f'{n.candidates[a.selections[n.name]].operator.name}({",".join(n.candidates[a.selections[n.name]].sources)})'
                                 for n in P.transform_program()[0].nodes if len(n.candidates)>1},
            'predicate_program':{n.name:f'{n.candidates[b.selections[n.name]].operator.name}({",".join(n.candidates[b.selections[n.name]].sources)})'
                                 for n in P.predicate_program()[0].nodes if len(n.candidates)>1},
            'transform_space':a.space_size,'transform_conforming':a.conforming,
            'predicate_space':b.space_size,'predicate_conforming':b.conforming,
            'mean':statistics.fmean(totals),'solved':sum(1 for x in totals if x>0),
            'modal_trace':statistics.mode(traces),'sequences_evaluated':evaluated,
            'environment_episodes':counter.episodes,'environment_steps':counter.steps,
            'seconds':time.perf_counter()-t0}
    print(json.dumps(report,indent=2,default=str))
    open(ROOT+'/research/credit-assignment/out/mpc.json','w').write(json.dumps(report,indent=2,default=str))

if __name__=='__main__':
    main(int(sys.argv[1]) if len(sys.argv)>1 else 25,int(sys.argv[2]) if len(sys.argv)>2 else 64)
