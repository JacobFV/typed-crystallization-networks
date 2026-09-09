"""Typed scaffold for the panel task, built only from registered operators.

WHAT IS DECLARED AND WHAT IS SEARCHED (ARCHITECTURE section 5, AGENTS.md).

Declared, and named here so a reader can check that none of it is the answer:

* The *shape* of the graph, as every scaffold in this repository is authored.
* The perception chain `terminal -> length -> byte at length-1 -> unpack -> code`
  and `terminal -> byte 0`. `research/computer-capability/RESULTS.md` established
  that the address choice on this observation receives `grad is None`, because
  `unpack` -- the only declared exit from `role="byte"` -- is `gradient="none"`.
  Leaving the address searched in a reward-only arm therefore measures a known
  gradient boundary, not credit assignment. It is searched anyway in
  `perception=True`, so the boundary is re-measured rather than assumed, and
  declared in the arms whose subject is the policy.
* The policy-parameter encoding. `tcn/policy.py:sample_typed` decodes a numeric
  action argument as `lo + (tanh(u)+1)/2*(hi-lo)`, so a program that wants to emit
  `x` must output `mean = atanh(2*(x-lo)/(hi-lo) - 1)`, written out of `log`, `sub`
  and `mul` because there is no `atanh` operator. The `min`/`max` clamp in front
  of it is not a task hint: `log` is partial and the terminal is a byte string, so
  without it a stray byte is a numerical domain error rather than a wrong action.
* That the policy's situation is `(previously dialled?, is the panel showing a
  task record?)`. This is the same kind of declaration as `examples/joint.py`
  declaring that `z` feeds the policy. `policy_state=False` ablates it to a single
  state-independent logit vector, and both numbers are reported.

Searched, every candidate legal by type and the whole space enumerable:

* `next_slot` -- 12 candidates over the executed-slot input and two constants.
  Two of them sweep the panel; the rest stare at one slot or repeat the last.
* the three logit vectors -- what to do when nothing is showing, when the task
  record is showing, and after dialling. Zero-initialised, so the policy starts
  uniform over the four templates and every bit of "look before you commit" has
  to come from the delayed reward.
* `pos`, `shift`, `qpos`, `brand` when `perception=True`.
"""
from __future__ import annotations
import math,sys
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from tcn.generation import text_value
from tcn.types import BOOL,Value,integer,floating,product
from tcn.operators import Registry
from tcn.graph import Program,Node,Candidate,Signal,legal_candidates
from generators.computer.generator import SLOT,DIAL

F=floating()
TERMINAL=text_value('',4096).type
LEN=TERMINAL.items[0];DATA=TERMINAL.items[1];BYTE=DATA.items[0]
U8=integer(8,signed=False)
ACTION=product(F,F,F,F)                    # one-hot over (wait, look, dial, commit)
PAIR=product(F,F)
DIGIT_LO,DIGIT_HI=1.,9.                    # the answer is always in 1..9
DIAL_HI=float((1<<DIAL.bits)-1)
SLOT_HI=float((1<<SLOT.bits)-1)
LOG_STD=-5.
ZERO4=(0.,0.,0.,0.)

def _node(name,output,candidates,region,depth):
    """Unselected even when there is one candidate.

    `SoftProgram` treats `selected is not None` as frozen and computes a frozen
    node with `.detach()`, which severs every choice upstream of it (FINDINGS
    section 18, re-confirmed twice since). A one-candidate node left unselected is
    a one-way softmax, which is the identity, and the relaxation is unchanged.
    """
    return Node(name,output,tuple(candidates),region,depth,None)

def constant(name,type_,value):return (name,Value.of(type_,value))


def encoder(r,nodes,source,name,depth):
    """`mean = atanh(2*(x-lo)/(hi-lo) - 1)`, as log((x-lo)) - log((hi-x)), halved.

    Exactly inverts `tcn/policy.py:sample_typed`'s bounded decode, so the sampled
    integer argument is `round(x)` whenever `x` is in range.
    """
    nodes+=[
      _node(f'{name}_lo',F,[Candidate(r.resolve('sub',(F,F)),(source,f'{name}_c_lo'))],'encode',depth),
      _node(f'{name}_hi',F,[Candidate(r.resolve('sub',(F,F)),(f'{name}_c_hi',source))],'encode',depth),
      _node(f'{name}_loglo',F,[Candidate(r.resolve('log',(F,)),(f'{name}_lo',))],'encode',depth+1),
      _node(f'{name}_loghi',F,[Candidate(r.resolve('log',(F,)),(f'{name}_hi',))],'encode',depth+1),
      _node(f'{name}_diff',F,[Candidate(r.resolve('sub',(F,F)),(f'{name}_loglo',f'{name}_loghi'))],'encode',depth+2),
      _node(f'{name}_mean',F,[Candidate(r.resolve('mul',(F,F)),(f'{name}_diff','half'))],'encode',depth+3),
      _node(f'{name}_params',PAIR,[Candidate(r.resolve('tuple',(F,F)),(f'{name}_mean','logstd'))],'encode',depth+4),
    ]
    return nodes


def panel_program(perception=False,policy_state=True,slot_pool=True,seed_logits=None):
    """The scaffold. `perception=True` puts the four perception choices in the search."""
    r=Registry()
    inputs=(('terminal',TERMINAL),('action',ACTION),('action.1.slot',SLOT),('action.2.value',DIAL))
    constants=[constant('one_len',LEN,1),constant('zero_len',LEN,0),
               constant('t_byte',BYTE,ord('t')),
               constant('c47',F,47.),constant('c15',F,DIAL_HI),
               constant('dlo',F,DIGIT_LO),constant('dhi',F,DIGIT_HI),
               constant('half',F,.5),constant('logstd',F,LOG_STD),
               constant('slot_scale',F,.96),constant('slot_bias',F,.06),
               constant('zero_slot',SLOT,0),constant('one_slot',SLOT,1),constant('three_slot',SLOT,3)]
    if perception:
        constants+=[constant(f'k{i}',LEN,v) for i,v in enumerate((1,2,3))]
        constants+=[constant(f'a{i}',LEN,v) for i,v in enumerate((0,4,8,42))]
        constants+=[constant(f'u{i}',F,float(v)) for i,v in enumerate((45.,46.,47.,48.))]
        constants+=[constant(f'b{i}',BYTE,ord(c)) for i,c in enumerate(('{','t','n','0'))]

    nodes=[
      _node('length',LEN,[Candidate(r.resolve('project',(TERMINAL,),LEN,{'index':0}),('terminal',))],'perception',1),
      _node('data',DATA,[Candidate(r.resolve('project',(TERMINAL,),DATA,{'index':1}),('terminal',))],'perception',1),
    ]
    if perception:
        ports={'length':LEN,'one_len':LEN}|{f'k{i}':LEN for i in range(3)}|{f'a{i}':LEN for i in range(4)}
        nodes.append(_node('pos',LEN,legal_candidates(r,('identity','sub'),ports,LEN,arities=(1,2)),'perception',2))
    else:
        nodes.append(_node('pos',LEN,[Candidate(r.resolve('sub',(LEN,LEN)),('length','one_len'))],'perception',2))
    nodes+=[
      _node('byte',BYTE,[Candidate(r.resolve('index',(DATA,LEN),BYTE),('data','pos'))],'perception',3),
      _node('unpacked',product(U8),[Candidate(r.resolve('unpack',(BYTE,),product(U8)),('byte',))],'perception',4),
      _node('code',U8,[Candidate(r.resolve('project',(product(U8),),U8,{'index':0}),('unpacked',))],'perception',5),
      _node('codef',F,[Candidate(r.resolve('decode',(U8,),F),('code',))],'perception',6),
    ]
    if perception:
        vports={'codef':F}|{f'u{i}':F for i in range(4)}
        nodes.append(_node('shift',F,legal_candidates(r,('identity','sub'),vports,F,arities=(1,2)),'transform',7))
        pports={'length':LEN,'zero_len':LEN}|{f'a{i}':LEN for i in range(4)}
        nodes.append(_node('qpos',LEN,legal_candidates(r,('identity','sub'),pports,LEN,arities=(1,2)),'perception',2))
        brand_sources=[Candidate(r.resolve('eq',(BYTE,BYTE)),('brandbyte',f'b{i}')) for i in range(4)]
    else:
        nodes.append(_node('shift',F,[Candidate(r.resolve('sub',(F,F)),('codef','c47'))],'transform',7))
        nodes.append(_node('qpos',LEN,[Candidate(r.resolve('identity',(LEN,)),('zero_len',))],'perception',2))
        brand_sources=[Candidate(r.resolve('eq',(BYTE,BYTE)),('brandbyte','t_byte'))]
    nodes+=[
      _node('brandbyte',BYTE,[Candidate(r.resolve('index',(DATA,LEN),BYTE),('data','qpos'))],'perception',3),
      _node('brand',BOOL,brand_sources,'perception',4),
      # clamp into the range the encoder is total on; `log` is partial and the
      # terminal is a byte string, so a stray byte must be a wrong action, not a crash
      _node('answer_lo',F,[Candidate(r.resolve('max',(F,F)),('shift','dlo'))],'transform',8),
      _node('answer',F,[Candidate(r.resolve('min',(F,F)),('answer_lo','dhi'))],'transform',9),
    ]
    nodes=encoder(r,nodes,'answer','dial',10)

    # --- slot: which panel slot to look at next -----------------------------
    sports={'action.1.slot':SLOT,'zero_slot':SLOT,'one_slot':SLOT}
    slot_candidates=legal_candidates(r,('identity','add'),sports,SLOT,arities=(1,2)) if slot_pool else \
                    [Candidate(r.resolve('add',(SLOT,SLOT)),('action.1.slot','one_slot'))]
    nodes+=[
      _node('next_slot',SLOT,slot_candidates,'plan',2),
      # `SLOT` is a wrapping 2-bit ring and the relaxed forward is a mixture over the
      # candidate pool, which does not wrap by itself, so the wrap is written out as
      # `x - 4 if x > 3.5 else x`. `mod` was the obvious operator and is wrong here:
      # `torch.remainder` is discontinuous at 4, so a mixture at 4 - 1e-8 comes back
      # as 3.99999 rather than 0 and the sweep silently sticks. The `min`/`max` after
      # it only bite outside [0, 3], which `sample_typed` would clamp anyway; they
      # are there because the encoder's `log` is partial. The candidate pool is
      # chosen so the uniform mixture stays inside the clamp -- a pool whose mixture
      # saturates it gives the choice logits exactly zero gradient, the same failure
      # shape as the uniform `truth_*` cancellation in FINDINGS section 22.
      _node('next_slot_f',F,[Candidate(r.resolve('decode',(SLOT,),F),('next_slot',))],'encode',3),
      _node('slot_over',BOOL,[Candidate(r.resolve('gt',(F,F)),('next_slot_f','slot_edge'))],'encode',4),
      _node('slot_shifted',F,[Candidate(r.resolve('sub',(F,F)),('next_slot_f','slot_mod'))],'encode',4),
      _node('slot_wrapped',F,[Candidate(r.resolve('mux',(BOOL,F,F)),('slot_over','slot_shifted','next_slot_f'))],'encode',5),
      _node('slot_clamped_lo',F,[Candidate(r.resolve('max',(F,F)),('slot_wrapped','zerof'))],'encode',6),
      _node('slot_clamped',F,[Candidate(r.resolve('min',(F,F)),('slot_clamped_lo','slot_hif'))],'encode',7),
      # y = 0.06 + 0.96*slot keeps the atanh finite at both ends and still rounds
      # to the slot it came from
      _node('slot_scaled',F,[Candidate(r.resolve('mul',(F,F)),('slot_clamped','slot_scale'))],'encode',8),
      _node('slot_y',F,[Candidate(r.resolve('add',(F,F)),('slot_scaled','slot_bias'))],'encode',9),
    ]
    constants+=[constant('zerof',F,0.),constant('slot_hif',F,SLOT_HI),constant('slot_mod',F,SLOT_HI+1),constant('slot_edge',F,SLOT_HI+.5),
                constant('slot_c_lo',F,0.),constant('slot_c_hi',F,SLOT_HI),
                constant('dial_c_lo',F,0.),constant('dial_c_hi',F,DIAL_HI)]
    nodes=encoder(r,nodes,'slot_y','slot',10)

    # --- policy -------------------------------------------------------------
    # `Program.validate` requires a trainable constant to be numeric, so each logit
    # vector is four trainable scalars assembled by a `tuple` node rather than one
    # tuple-typed constant. Same parameters, same gradient path.
    init=seed_logits or {}
    situations=('idle','found','dialled')
    for name in situations:
        values=tuple(init.get(name,ZERO4))
        constants+=[constant(f'{name}{i}',F,float(values[i])) for i in range(4)]
    constants+=[constant('baseline',F,0.)]
    for name in situations:
        nodes.append(_node(f'logits_{name}',ACTION,[Candidate(r.resolve('tuple',(F,F,F,F)),tuple(f'{name}{i}' for i in range(4)))],'plan',4))
    if policy_state:
        nodes+=[
          _node('prev_dial_f',F,[Candidate(r.resolve('project',(ACTION,),F,{'index':2}),('action',))],'plan',1),
          _node('prev_dial',BOOL,[Candidate(r.resolve('gt',(F,F)),('prev_dial_f','half'))],'plan',2),
          _node('inner',ACTION,[Candidate(r.resolve('mux',(BOOL,ACTION,ACTION)),('brand','logits_found','logits_idle'))],'plan',5),
          _node('policy',ACTION,[Candidate(r.resolve('mux',(BOOL,ACTION,ACTION)),('prev_dial','logits_dialled','inner'))],'plan',6),
        ]
    else:
        nodes.append(_node('policy',ACTION,[Candidate(r.resolve('identity',(ACTION,)),('logits_idle',))],'plan',6))
    nodes+=[
      _node('value',product(F),[Candidate(r.resolve('tuple',(F,)),('baseline',))],'value',1),
      _node('brandf',F,[Candidate(r.resolve('encode',(BOOL,),F),('brand',))],'encode',5),
      _node('probe',PAIR,[Candidate(r.resolve('tuple',(F,F)),('answer','brandf'))],'probe',10),
    ]
    outputs=(('policy','policy'),('slot_params','slot_params'),('dial_params','dial_params'),
             ('value','value'),('probe','probe'))
    trainable=tuple(f'{name}{i}' for name in situations for i in range(4))+('baseline',)
    program=Program(inputs,tuple(nodes),outputs,tuple(constants),trainable_constants=trainable)
    return program.validate(r),r


def perception_signals():
    """Probe supervision for the perception/transform region only."""
    return (Signal('probe','reference_probe',('perception','transform','encode'),PAIR,'mse'),)
