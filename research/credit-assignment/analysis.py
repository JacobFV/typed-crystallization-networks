"""Does the panel task pose credit assignment? Exact answers, before anything is trained.

Three questions, each answered by exhaustive computation over the task's own
generative distribution (task slot uniform over 4, digit uniform over 0..8,
initial register uniform over 0..15), not by sampling:

1. `values`      -- the optimal undiscounted return V*(H) and the myopic (gamma = 0)
                    return at each horizon, and the optimal first action. A task that
                    decomposes into independent bandits has V*(H) = H * V*(1) and a
                    myopic policy that is optimal; this one does not.
2. `discount`    -- the discount at which the optimal policy flips from "commit now"
                    to "look first". Credit assignment bites exactly there.
3. `exploration` -- the probability that a neutral policy's episode is rewarded, in
                    the same sense `research/computer-capability/RESULTS.md` computed
                    5.65e-06 for `write.text`.

Run: .venv/bin/python research/credit-assignment/analysis.py
"""
from __future__ import annotations
import json,sys
from fractions import Fraction as Q
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from panel import SLOT_NEUTRAL,DIAL_NEUTRAL,HORIZON
from generators.computer.generator import PANEL_SLOTS,PANEL_DIGITS,DIAL

REGISTERS=1<<DIAL.bits
ANSWERS=PANEL_DIGITS                     # the answer is uniform on 1..PANEL_DIGITS
P_BLIND_REGISTER=Q(1,REGISTERS)          # P(initial register == answer)
P_BLIND_DIAL=Q(1,ANSWERS)                # P(a uniform dial in 1..9 == answer)


# ---------------------------------------------------------------------------
# 1 & 2. exact values by belief-state dynamic programming
# ---------------------------------------------------------------------------
# Belief state, from the agent's side. The agent never observes the register, and
# learns the digit only by looking at the task slot.
#   ('search', k, dialled)  k slots eliminated, none was the task
#   ('found',     dialled)  the task record is on the panel, the digit is known
# `dialled` is None (register unknown, uniform), 'blind' (dialled a guess), or
# 'answer' (dialled the digit the agent read).

def solve(horizon,discount=1.):
    """Optimal value and greedy action at every belief state, for one discount."""
    g=Q(discount).limit_denominator(10**9) if isinstance(discount,(int,float)) else discount
    memo={}
    def value(state,m):
        """Expected *discounted* return with m steps remaining."""
        if m<=0:return Q(0),'-'
        key=(state,m)
        if key in memo:return memo[key]
        kind=state[0];options={}
        # commit: immediate reward, then the episode ends.
        if kind=='found':
            dialled=state[1]
            options['commit']=P_BLIND_REGISTER if dialled is None else (Q(1) if dialled=='answer' else P_BLIND_DIAL)
            options['dial']=g*value(('found','answer'),m-1)[0]
            options['wait']=g*value(state,m-1)[0]
            options['look']=g*value(state,m-1)[0]       # re-looking learns nothing new
        else:
            k=state[1];dialled=state[2]
            options['commit']=P_BLIND_REGISTER if dialled is None else (Q(1) if dialled=='answer' else P_BLIND_DIAL)
            # a blind dial cannot be the read answer: the digit is still unknown
            if dialled!='blind':options['dial']=g*value(('search',k,'blind'),m-1)[0]
            options['wait']=g*value(state,m-1)[0]
            if k<PANEL_SLOTS:
                hit=Q(1,PANEL_SLOTS-k)
                options['look']=g*(hit*value(('found',dialled),m-1)[0]+(1-hit)*value(('search',k+1,dialled),m-1)[0])
        best=max(options.values());action=sorted(a for a,v in options.items() if v==best)[0]
        memo[key]=(best,action);return memo[key]
    return value(('search',0,None),horizon),value


def values_table(max_horizon=8):
    rows=[]
    for h in range(1,max_horizon+1):
        (opt,first),_=solve(h,1.)
        (myo,myo_first),_=solve(h,0.)
        rows.append({'horizon':h,'optimal':float(opt),'optimal_first_action':first,
                     'myopic':float(myo),'myopic_first_action':myo_first,
                     'gap':float(opt-myo),'ratio':float(opt/myo) if myo else None,
                     'bandit_prediction':float(h*solve(1,1.)[0][0])})
    return rows


def discount_threshold(horizon=HORIZON,steps=2001):
    """The smallest discount at which the optimal first action stops being `commit`."""
    lo,hi=0.,1.
    first=lambda g:solve(horizon,g)[0][1]
    if first(1.)=='commit':return None
    for _ in range(60):
        mid=(lo+hi)/2
        if first(mid)=='commit':lo=mid
        else:hi=mid
    return {'threshold':hi,'first_below':first(lo),'first_above':first(hi),
            'closed_form':(1/REGISTERS)**(1/(horizon-1)),
            'note':'commit now pays 1/16; the plan pays gamma^(H-1); they cross at (1/16)^(1/(H-1))'}


# ---------------------------------------------------------------------------
# 3. exploration probability under a neutral policy
# ---------------------------------------------------------------------------

def exploration(horizon=HORIZON,templates=4):
    """Exact P(reward) for a policy that has learned nothing.

    The neutral typed policy is observation-independent: uniform over the four
    templates, `slot` and `value` drawn by `tcn/policy.py:sample_typed` from
    zero parameters. Success therefore depends only on the register at the moment
    of the first commit, so the whole chain is a three-state Markov computation.
    """
    p_commit=1/templates;p_dial=1/templates
    p_dial_right=sum(P for v,P in DIAL_NEUTRAL.items() if 1<=v<=ANSWERS)/ANSWERS
    p_blind=float(P_BLIND_REGISTER)
    # state: (has dialled) -> probability mass still running
    mass_undialled,mass_dialled,hit=1.,0.,0.
    per_step=[]
    for t in range(horizon):
        step_hit=mass_undialled*p_commit*p_blind+mass_dialled*p_commit*p_dial_right
        hit+=step_hit;per_step.append(step_hit)
        surviving_u=mass_undialled*(1-p_commit);surviving_d=mass_dialled*(1-p_commit)
        mass_dialled=surviving_d*(1-p_dial)+surviving_d*p_dial+surviving_u*p_dial
        mass_undialled=surviving_u*(1-p_dial)
    # the intended route: look at the task slot, then dial what was read, then commit.
    p_look_task=sum(P for s,P in SLOT_NEUTRAL.items() if s<PANEL_SLOTS)/(templates*PANEL_SLOTS)
    p_seq=p_look_task*(p_dial*p_dial_right)*p_commit
    orderings=sum(1 for a in range(horizon) for b in range(a+1,horizon) for c in range(b+1,horizon))
    return {'horizon':horizon,'p_neutral_commit_rewarded':p_dial_right,
            'p_blind_register_rewarded':p_blind,
            'p_episode_rewarded':hit,'episodes_per_reward':1/hit if hit else None,
            'per_step':per_step,
            'p_intended_triple_in_order':min(1.,p_seq*orderings),
            'shell_write_text_reference':5.65e-06,
            'improvement_over_write_text':p_dial_right/5.65e-06}


def bandit_check(max_horizon=8):
    """The `logic` failure mode, tested for directly.

    On `generators/logic` the state is constant, every action is rewarded on the
    step it is taken, and a horizon-H episode is H independent draws of one
    bandit: V*(H) = H*V*(1) exactly, and the myopic policy is optimal at every
    horizon. Both identities are checked here and both fail.
    """
    rows=values_table(max_horizon)
    v1=rows[0]['optimal']
    return {'linear_in_horizon':all(abs(r['optimal']-r['horizon']*v1)<1e-12 for r in rows),
            'myopic_is_optimal':all(abs(r['optimal']-r['myopic'])<1e-12 for r in rows),
            'per_horizon':[{'horizon':r['horizon'],'optimal':r['optimal'],
                            'bandit_prediction':r['horizon']*v1,'myopic':r['myopic']} for r in rows],
            'verdict':'not a bandit' if not all(abs(r['optimal']-r['myopic'])<1e-12 for r in rows) else 'bandit'}


def main():
    report={'values':values_table(8),'discount':discount_threshold(),
            'exploration':{h:exploration(h) for h in (3,4,6,8)},
            'bandit_check':bandit_check(8),
            'neutral_distributions':{'slot':SLOT_NEUTRAL,'dial':DIAL_NEUTRAL}}
    print(json.dumps({k:v for k,v in report.items() if k!='neutral_distributions'},indent=2,default=str))
    open(ROOT+'/research/credit-assignment/out/analysis.json','w').write(json.dumps(report,indent=2,default=str))

if __name__=='__main__':main()
