"""Joint goal-conditioned control and latent prediction in a typed logic scaffold."""
from tcn.types import BOOL,Value,product
from tcn.generation import Host,Action
from tcn.scaffold import F
from tcn.graph import Program,Node,Candidate
from tcn.operators import Registry
from tcn.learning import SoftProgram
from tcn.training import TrainConfig,Target,JointTrainer

def trainer(episodes=160,seed=0):
    settings={'depth':1,'table':6,'fixed_inputs':True};host=Host.create('logic',configuration=settings);r=Registry()
    names=('bits','goal');inputs=tuple((k,host.view().observations[k].type) for k in names)+(('action',product(F,F)),('dt',F))
    nodes=[]
    for i in range(2):
        op=r.resolve('project',(dict(inputs)['bits'],),parameters={'index':i})
        nodes.append(Node(f'bit_{i}',BOOL,(Candidate(op,('bits',)),),'observation',1))
    for name,sources,depth in [('relation',('bit_0','bit_1'),2),('goal_relation',('relation','goal'),3)]:
        nodes.append(Node(name,BOOL,tuple(Candidate(r.resolve(f'truth_{i}',(BOOL,BOOL)),sources) for i in range(16)),'latent',depth))
    for name,source in [('z','goal_relation'),('world','relation')]:
        nodes.append(Node(name,F,(Candidate(r.resolve('encode',(BOOL,),F),(source,)),),'encoding',4))
    constants=(('w0',Value.of(F,-2.)),('w1',Value.of(F,2.)),('bias0',Value.of(F,1.)),('bias1',Value.of(F,-1.)),('baseline',Value.of(F,1.)))
    for i in range(2):
        nodes.append(Node(f'mul{i}',F,(Candidate(r.resolve('mul',(F,F)),('z',f'w{i}')),),'policy',5))
        nodes.append(Node(f'logit{i}',F,(Candidate(r.resolve('add',(F,F)),(f'mul{i}',f'bias{i}')),),'policy',6))
    nodes.append(Node('policy',product(F,F),(Candidate(r.resolve('tuple',(F,F)),('logit0','logit1')),),'policy',7))
    nodes.append(Node('prediction',product(F,F),(Candidate(r.resolve('tuple',(F,F)),('z','world')),),'prediction',5))
    nodes.append(Node('value',product(F),(Candidate(r.resolve('tuple',(F,)),('baseline',)),),'value',1))
    program=Program(inputs,tuple(nodes),(('policy','policy'),('prediction','prediction'),('probe','prediction'),('value','value')),constants,trainable_constants=tuple(k for k,_ in constants))
    model=SoftProgram(program)
    # Residual initializations break the uniform truth-table cancellation; no
    # correct operator is supplied. Both latent nodes initially copy input one.
    import torch
    with torch.no_grad():
        for n,p in zip(program.nodes,model.choices):
            if len(n.candidates)==16:p[12]=2.
    actions=(Action('answer',arguments=(('value',Value.of(BOOL,False)),)),Action('answer',arguments=(('value',Value.of(BOOL,True)),)))
    config=TrainConfig('logic',names,(Target('probes','target',1),Target('probes','gate',1)),actions,settings,objectives=({'invert':False},{'invert':True}),episodes=episodes,horizon=4,lr=.04,probe_weight=1.,seed=seed)
    return JointTrainer(model,config)
