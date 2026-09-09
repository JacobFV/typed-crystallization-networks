"""Typed scaffolds for the successor task, built only from registered operators.

WHAT IS DECLARED AND WHAT IS SEARCHED (ARCHITECTURE section 5 / AGENTS.md).

Declared, and stated here because a reader must be able to check that the
initialisation is a prior and not the answer:

* The *shape* of the graph -- which node feeds which -- is authored, exactly as
  every scaffold in this repository is authored.
* The byte/numeric representation boundary. `role="byte"` puts the terminal
  outside `Type.numeric`, so no arithmetic is legal on it; `unpack` is the
  declared conversion out and it is the only one. This is a representation
  contract, not a task hint.
* The policy-parameter encoding. `tcn/policy.py:sample_typed` decodes a numeric
  action argument as `lo + (tanh(mean)+1)/2*(hi-lo)`, so a program that wants to
  emit byte `x` must output `mean = atanh(2x/255 - 1) = (log x - log(255-x))/2`.
  That algebra is the inverse of the action encoding, and it is fixed. It is
  written out of `log`, `sub` and `mul` because no `atanh` operator exists.
* The three action-logit constants, and that `wait` is template 0.

Searched, with every candidate legal by type and the whole space enumerable:

* `pos` -- which byte of the terminal holds the digit. Constants for absolute
  addressing are in the pool alongside `sub(length, k)`, so the search has to
  *choose* a computed address over a constant one; the training documents have
  five different lengths, so no constant address conforms.
* `shift` -- what arithmetic turns the observed byte into the byte to write.
  `identity` is in the pool, so "write what you read" is a legal answer that the
  supervision has to rule out.
* `ppos` and `brand` -- which byte of the terminal, compared against which
  constant, decides whether the terminal is showing file content or a command
  result. Both are free.

Every choice logit starts at zero (`SoftProgram` zero-initialises), so the
gradient arm gets a uniform prior over every candidate, and the enumerative arm
gets no prior at all.
"""
import math
from tcn.generation import text_value
from tcn.types import BOOL,Value,integer,floating,product
from tcn.operators import Registry
from tcn.graph import Program,Node,Candidate,Signal,legal_candidates

F=floating()
TERMINAL=text_value('',4096).type
LEN=TERMINAL.items[0]
DATA=TERMINAL.items[1]
BYTE=DATA.items[0]
U8=integer(8,signed=False)
ACTION=product(F,F,F)
TEXT=text_value('',512).type

READ_LOGITS=(0.,1.,0.);WRITE_LOGITS=(0.,0.,1.);WAIT_LOGITS=(1.,0.,0.)
# atanh(2*1/512 - 1) puts `sample_typed`'s bounded decode of the length field on 1.
MEAN_LENGTH=.5*math.log((1/512.)/(1-1/512.))
LOG_STD=-5.

def constant(name,type_,value): return (name,Value.of(type_,value))

def _node(name,output,candidates,region,depth):
    return Node(name,output,tuple(candidates),region,depth,0 if len(candidates)==1 else None)

def address_candidates(registry,ports,names=('identity','sub','add')):
    return legal_candidates(registry,names,ports,LEN,arities=(1,2))

def transform_program(registry,offsets=(1,2,3),digits=(0,1,2,3),wide=False,encode=True):
    """terminal -> the byte to write, and the policy parameters that emit it.

    `wide=True` widens both pools; the narrow default keeps the joint space small
    enough to exhaust in seconds. `encode=False` drops the declared action-parameter
    encoder, which is what both search arms run on: the encoder is downstream of the
    supervised node and cannot change it, but `log` is partial, so leaving it in
    would let a numeric domain error silently delete conforming candidates from the
    enumeration and abort the relaxed forward pass at the initial uniform mixture.
    """
    absolutes=(0,1,2,4,6,8,10,12,20,41,42) if wide else (0,4,8,42)
    unit=(0,1,2,3,4,5,6,7) if wide else digits
    constants=[constant('one',LEN,1)]
    constants+= [constant(f'k{i}',LEN,v) for i,v in enumerate(offsets)]
    constants+= [constant(f'a{i}',LEN,v) for i,v in enumerate(absolutes)]
    constants+= [constant(f'u{i}',U8,v) for i,v in enumerate(unit)]
    constants+= [constant('c255',F,255.),constant('chalf',F,.5),
                 constant('cmeanlen',F,MEAN_LENGTH),constant('clogstd',F,LOG_STD),
                 constant('ctail',product(*(F for _ in range(1022))),tuple(0. for _ in range(1022)))]
    address_ports={'length':LEN}|{f'k{i}':LEN for i in range(len(offsets))}|{f'a{i}':LEN for i in range(len(absolutes))}
    value_ports={'value':U8}|{f'u{i}':U8 for i in range(len(unit))}
    nodes=[
      _node('length',LEN,[Candidate(registry.resolve('project',(TERMINAL,),LEN,{'index':0}),('terminal',))],'perception',1),
      _node('data',DATA,[Candidate(registry.resolve('project',(TERMINAL,),DATA,{'index':1}),('terminal',))],'perception',1),
      _node('pos',LEN,address_candidates(registry,address_ports),'perception',2),
      _node('byte',BYTE,[Candidate(registry.resolve('index',(DATA,LEN),BYTE),('data','pos'))],'perception',3),
      _node('unpacked',product(U8),[Candidate(registry.resolve('unpack',(BYTE,),product(U8)),('byte',))],'perception',4),
      _node('value',U8,[Candidate(registry.resolve('project',(product(U8),),U8,{'index':0}),('unpacked',))],'perception',5),
      _node('shift',U8,legal_candidates(registry,('identity','add','sub'),value_ports,U8,arities=(1,2)),'transform',6),
    ]
    outputs=(('shift','shift'),)
    if encode: nodes+=[
      _node('shiftf',F,[Candidate(registry.resolve('decode',(U8,),F),('shift',))],'encode',7),
      _node('lower',F,[Candidate(registry.resolve('log',(F,)),('shiftf',))],'encode',8),
      _node('complement',F,[Candidate(registry.resolve('sub',(F,F)),('c255','shiftf'))],'encode',8),
      _node('upper',F,[Candidate(registry.resolve('log',(F,)),('complement',))],'encode',9),
      _node('difference',F,[Candidate(registry.resolve('sub',(F,F)),('lower','upper'))],'encode',10),
      _node('mean',F,[Candidate(registry.resolve('mul',(F,F)),('difference','chalf'))],'encode',11),
      _node('head',product(F,F,F,F),[Candidate(registry.resolve('tuple',(F,F,F,F)),('cmeanlen','clogstd','mean','clogstd'))],'encode',12),
      _node('text',product(product(F,F,F,F),product(*(F for _ in range(1022)))),
            [Candidate(registry.resolve('tuple',(product(F,F,F,F),product(*(F for _ in range(1022))))),('head','ctail'))],'encode',13),
    ];outputs=outputs+(('text','text'),)
    return Program((('terminal',TERMINAL),),tuple(nodes),outputs,tuple(constants)).validate(registry)

def policy_program(registry,probes=(0,1,2,8,41,42),brands=('{','"','c','0',' ','}','[','t'),wide=False):
    if wide: probes=(0,1,2,3,4,6,8,10,20,41,42,43);brands=('{','"','c','0',' ','}','[','t','=','n','a','9')
    constants=[constant(f'q{i}',LEN,v) for i,v in enumerate(probes)]
    constants+=[constant(f'b{i}',BYTE,ord(c)) for i,c in enumerate(brands)]
    constants+=[constant('half',F,.5),constant('read_logits',ACTION,READ_LOGITS),
                constant('write_logits',ACTION,WRITE_LOGITS),constant('wait_logits',ACTION,WAIT_LOGITS)]
    probe_ports={'length':LEN}|{f'q{i}':LEN for i in range(len(probes))}
    nodes=[
      _node('length',LEN,[Candidate(registry.resolve('project',(TERMINAL,),LEN,{'index':0}),('terminal',))],'perception',1),
      _node('data',DATA,[Candidate(registry.resolve('project',(TERMINAL,),DATA,{'index':1}),('terminal',))],'perception',1),
      _node('ppos',LEN,address_candidates(registry,probe_ports,('identity','sub')),'perception',2),
      _node('pbyte',BYTE,[Candidate(registry.resolve('index',(DATA,LEN),BYTE),('data','ppos'))],'perception',3),
      _node('brand',BOOL,[Candidate(registry.resolve('eq',(BYTE,BYTE)),('pbyte',f'b{i}')) for i in range(len(brands))],'perception',4),
      _node('visible',BOOL,[Candidate(registry.resolve('not',(BOOL,)),('brand',))],'plan',5),
      _node('previous_write',F,[Candidate(registry.resolve('project',(ACTION,),F,{'index':2}),('action',))],'plan',1),
      _node('wrote',BOOL,[Candidate(registry.resolve('gt',(F,F)),('previous_write','half'))],'plan',2),
      _node('inner',ACTION,[Candidate(registry.resolve('mux',(BOOL,ACTION,ACTION)),('visible','write_logits','read_logits'))],'plan',6),
      _node('policy',ACTION,[Candidate(registry.resolve('mux',(BOOL,ACTION,ACTION)),('wrote','wait_logits','inner'))],'plan',7),
    ]
    return Program((('terminal',TERMINAL),('action',ACTION)),tuple(nodes),(('policy','policy'),),tuple(constants)).validate(registry)

def agent_program(registry,transform,policy,transform_selection,policy_selection):
    """One frozen program with both regions, hardened at the searched selections.

    The `action.2.text` port is the executed-action input `tcn/policy.py:action_inputs`
    requires for every binding. Nothing reads it; it is part of the declared interface.
    """
    hardened_t=transform.harden(transform_selection)
    hardened_p=policy.harden(policy_selection)
    shared={'length','data'}
    nodes=list(hardened_t.nodes)+[n for n in hardened_p.nodes if n.name not in shared]
    constants=list(hardened_t.constants)+[c for c in hardened_p.constants if c[0] not in dict(hardened_t.constants)]
    return Program((('terminal',TERMINAL),('action',ACTION),('action.2.text',TEXT)),tuple(nodes),
                   (('policy','policy'),('text','text')),tuple(constants)).validate(registry)

def signals():
    return (Signal('shift','reference_byte',('transform',),U8,'mse'),)

def policy_signals():
    return (Signal('policy','reference_action',('plan',),ACTION,'mse'),)
