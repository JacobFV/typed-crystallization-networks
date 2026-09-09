"""Command-line entry points for synthesis, curricula, episodes, and exact runtime."""
import argparse
import json
from pathlib import Path
import sys

def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False))

def mixed(out,steps=300,baseline_limit=1<<16,mode='relax'):
    """`mode` selects the search backend; it defaults to the shipped gradient path.

    `'auto'` lets `tcn.select` decide from the scaffold and the data, which on
    this fixture is enumeration -- 96 programs, exhaustible in milliseconds,
    with a uniqueness certificate a gradient run cannot produce. The discrete
    reference below is reported either way, so the two remain comparable.
    """
    from examples.mixed import problem
    from .synthesis import fit
    from .runtime import save_program,export_executable,benchmark
    from .search import enumerate_fit,space_size
    p,signals,examples=problem();model,report=fit(p,examples,signals,steps=steps,tolerance=.005,mode=mode)
    # A synthesis number means little without the discrete reference beside it:
    # enumeration searches the identical candidate space, and when it exhausts
    # that space it also reports whether the solution is unique.
    report['space_size']=space_size(p)
    if report['space_size']<=baseline_limit:
        found=enumerate_fit(p,examples,signals,tolerance=.005)
        report['discrete_baseline']=found.to_dict()
        report['agrees_with_enumeration']=(model.selections()==found.selections) if found.solved else None
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if report['fully_frozen'] and report['exact_conformance']:
        exported=model.export();save_program(exported,out/'program.json');export_executable(exported,out/'program.pyz')
        report['benchmark']=benchmark(exported,[x['inputs'] for x in examples])
    write_json(out/'report.json',report);return report

def joint(out,episodes=160):
    from examples.joint import trainer
    t=trainer(episodes);history=t.run(out)
    evaluations=[t.episode(10000+i,train=False,split='test')[0] for i in range(16)]
    report={'episodes':len(history),'initial_prediction_loss':sum(x['prediction_loss'] for x in history[:8])/len(history[:8]),'final_prediction_loss':sum(x['prediction_loss'] for x in history[-8:])/len(history[-8:]),'evaluation_mean_return':sum(x['return'] for x in evaluations)/len(evaluations),'maximum_return':t.config.horizon,'evaluations':evaluations}
    from .crystallize import Crystallizer,Objective
    from .runtime import save_program,export_executable
    from .agent import Agent
    from .generation import Host
    scheduler=Crystallizer(t.model,t.optimizer,tolerance=.05)
    # Two objectives, not one: the regularized total is what residual retraining
    # descends, and the unregularized task loss is what the connectivity guard
    # probes. Without the split the discreteness term keeps every unfrozen choice
    # logit attached to the loss graph and the guard cannot see a severed interior.
    def validation(regularized=True):return sum(t.episode(20000+i,False,'validation',loss_only=True,regularized=regularized) for i in range(2))/2
    scheduler.run(Objective(validation,lambda:validation(False)),rounds=24,retrain_steps=2)
    report['fully_frozen']=len(t.model.frozen)==len(t.model.program.nodes) and all(not p.requires_grad for p in t.model.constants.values())
    report['freeze_events']=[vars(e) for e in scheduler.events]
    if report['fully_frozen']:
        p=t.model.export();save_program(p,Path(out)/'program.json',t.model.registry);export_executable(p,Path(out)/'program.pyz',t.model.registry)
        agent=Agent(p,t.model.registry,t.config);returns=[]
        for i in range(16):
            host=Host.create(t.config.generator,seed=t.config.seed,index=30000+i,split='test',configuration=t.config.generator_config|{'horizon':t.config.horizon},objective=t.config.objectives[i%len(t.config.objectives)])
            agent.rollout(host,deterministic=True);returns.append(sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records))
        report['frozen_evaluation_mean_return']=sum(returns)/len(returns)
        write_json(Path(out)/'agent.json',t.config.to_dict())
    write_json(Path(out)/'evaluation.json',report);return report

# ---------------------------------------------------------------------------
# Demonstrations
#
# `tcn demo` re-runs the capabilities this repository can actually show, on the
# current tree, and prints every number next to the reference it has to beat.
# The standing rule from the 2026-09-08 measurement pass is that a number
# without its baseline is uninterpretable, so no row here reports a bare score:
# a return is printed against the best constant policy measured on the same
# episodes, a synthesis result against exhaustive enumeration over the identical
# candidate space, and a search result against the size of the space it settled
# and whether the answer is unique in it.
#
# The measurement scaffolds live under `research/<track>/` and are re-run rather
# than quoted. Several tracks each define a top-level `common` module, so the
# module cache and `sys.path` are reset between demos instead of shared.
# ---------------------------------------------------------------------------
DEMOS=('synthesis','joint','structure','depth','abstraction','positional','segmentation','edge',
       'control','language')

def _root(): return Path(__file__).resolve().parents[1]

def _track(name):
    """Put one research track on the path, evicting any previously loaded one."""
    import importlib
    root=_root();base=str(root/'research')
    for key,module in list(sys.modules.items()):
        if str(getattr(module,'__file__',None) or '').startswith(base):sys.modules.pop(key,None)
    sys.path[:]=[p for p in sys.path if not p.startswith(base)]
    sys.path.insert(0,str(root/'research'/name))
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    importlib.invalidate_caches()

def _finite(value):
    """JSON has no infinity and a failing measurement often is one.

    `write_json` refuses non-finite floats, and a demonstration that fails is
    exactly when its detail is most worth keeping, so name the value rather than
    losing the whole record to a serialization error.
    """
    import math
    if isinstance(value,float) and not math.isfinite(value):return repr(value)
    if isinstance(value,dict):return {k:_finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_finite(v) for v in value]
    return value

def _row(name,claim,measured,baseline,ok,seconds,evidence,command,detail=None):
    return {'demo':name,'claim':claim,'measured':measured,'baseline':baseline,'ok':bool(ok),
            'verdict':'PASS' if ok else 'FAIL','seconds':round(seconds,2),'evidence':evidence,
            'command':command,'detail':_finite(detail or {})}

# --- individual demonstrations ---------------------------------------------

def _demo_synthesis(out,quick):
    """Typed synthesis with exact export, cross-checked against enumeration."""
    import math,time
    from .runtime import load_program
    from .types import BOOL,Value,floating
    start=time.perf_counter();report=mixed(out,steps=300);wall=time.perf_counter()-start
    reference=report['discrete_baseline'];F=floating();worst=0.;points=0
    program,registry=load_program(out/'program.json')
    # The fitting set holds four values of x. Score the exported program on 146
    # unseen x over the same interval `research/baselines` scored its MLPs on,
    # because "exact" is a claim about points that were never fitted.
    for i in range(146):
        x=-.7+1.5*i/145
        for a in (False,True):
            for b in (False,True):
                outputs,_=program.run({'a':Value.of(BOOL,a),'b':Value.of(BOOL,b),'x':Value.of(F,x)},registry=registry)
                worst=max(worst,abs(outputs['answer'].decoded-math.sin(float(a!=b)+x)));points+=1
    ok=(report['exact_conformance'] and report['fully_frozen'] and reference['solved']
        and reference['unique'] and report['agrees_with_enumeration'] and worst<=1e-6)
    detail={'exact_max_error_on_fitted_set':report['exact_max_error'],'relaxed_loss':report['relaxed_loss'],
            'held_out_points':points,'held_out_max_error':worst,'gradient_seconds':wall,
            'enumeration_seconds':reference['seconds'],'space_size':reference['space_size'],
            'conforming':reference['conforming'],'unique':reference['unique'],
            'agrees_with_enumeration':report['agrees_with_enumeration'],
            'recorded_matched_mlp_interpolation_max':1.8e-2,'recorded_matched_mlp_extrapolation_rmse':.232,
            'description_bits':report.get('benchmark',{}).get('description_bits'),
            'batch_one_p50_ms':report.get('benchmark',{}).get('p50_ms')}
    return _row('synthesis','exact typed program from 16 examples, exported and checked',
                f"held-out max |err| {worst:.1e} on {points} unfitted points",
                f"enumeration: unique of {reference['space_size']} in {reference['seconds']*1e3:.1f} ms "
                f"(gradient {wall:.0f} s, same program); matched MLP 1.8e-02",
                ok,wall,'research/baselines/RESULTS.md §3; research/enumerative-baseline/RESULTS.md §0',
                'tcn demo --only synthesis',detail)

def _joint_reference(config,indices,answer=None,rng=None):
    """Return of a constant or uniform-random policy on the same episodes."""
    from .generation import Host,Action
    from .types import BOOL,Value
    returns=[]
    for i,index in enumerate(indices):
        host=Host.create(config.generator,seed=config.seed,index=index,split='test',
                         configuration=config.generator_config|{'horizon':config.horizon},
                         objective=config.objectives[i%len(config.objectives)])
        for _ in range(config.horizon):
            if host.records[-1].done:break
            value=(rng.random()<.5) if rng is not None else answer
            host.step((Action('answer',arguments=(('value',Value.of(BOOL,value)),)),),config.dt)
        returns.append(sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records))
    return sum(returns)/len(returns)

def _demo_joint(out,quick):
    """The recorded joint result, reproduced with the references it lacked."""
    import random,time
    from examples.joint import trainer
    start=time.perf_counter();report=joint(out,160);wall=time.perf_counter()-start
    config=trainer(1).config
    # The recorded protocol scores the frozen agent on sixteen episodes. Those
    # sixteen are not balanced: a constant answer already reaches 3.00 there, so
    # the reference is also measured on a sixty-four episode extension of the
    # same family, which is the number the verdict is taken against.
    scored=range(30000,30016);extended=range(30000,30064)
    references={'recorded_16_episodes':{'always_true':_joint_reference(config,scored,True),
                                        'always_false':_joint_reference(config,scored,False),
                                        'uniform_random':_joint_reference(config,scored,rng=random.Random(0))},
                'extended_64_episodes':{'always_true':_joint_reference(config,extended,True),
                                        'always_false':_joint_reference(config,extended,False),
                                        'uniform_random':_joint_reference(config,extended,rng=random.Random(0))}}
    for block in references.values():block['best_constant']=max(block['always_true'],block['always_false'])
    near=references['recorded_16_episodes']['best_constant']
    far=references['extended_64_episodes']['best_constant']
    measured=report['frozen_evaluation_mean_return']
    ok=report['fully_frozen'] and measured>=3.8 and measured>far+1.
    detail={'initial_prediction_loss':report['initial_prediction_loss'],
            'final_prediction_loss':report['final_prediction_loss'],
            'soft_model_return':report['evaluation_mean_return'],
            'frozen_program_return':measured,'maximum_return':report['maximum_return'],
            'references':references,'declared_episode_budget':160,'actual_environment_episodes':331,
            'caveats':['The policy decoder constants in examples/joint.py already implement the '
                       'correct decision rule before training (logit0=-2z+1, logit1=2z-1); the '
                       'learned content is two 16-way truth-table choices, 8 bits.',
                       'Enumeration over the same 256-program space returns the identical program '
                       'in 0.081 ms against 10-36 s of gradient descent, with a uniqueness '
                       'certificate (research/enumerative-baseline/RESULTS.md).',
                       'Crystallization is not what produced this: an argmax of the trained soft '
                       'graph with no crystallizer and zero extra objective evaluations reaches the '
                       'same 4/4 and the same frozen program (research/crystallization-ablation, arm B0).',
                       'This is one fixed Boolean function with held-out episode addresses; it is '
                       'not structural generalization. See the structure demo for that.',
                       '--episodes 160 consumes 331 environment episodes: 170 undisclosed '
                       "split='validation' rollouts inside the crystallizer loss closure.",
                       'A budget-matched 153-parameter MLP with replay also reaches 4.00, and so '
                       'does a 32-bit lookup table (research/baselines/RESULTS.md section 4).',
                       'On the sixteen episodes the recorded protocol scores, a constant answer '
                       f'already reaches {near:.2f}/4.']}
    return _row('joint','goal-conditioned control and latent prediction, exact frozen agent',
                f"frozen program {measured:.2f}/4, prediction loss "
                f"{report['initial_prediction_loss']:.5f} -> {report['final_prediction_loss']:.5f}",
                f"best constant {far:.2f}/4 over 64 episodes ({near:.2f}/4 over the 16 the record "
                f"scores), uniform random "
                f"{references['extended_64_episodes']['uniform_random']:.2f}/4, oracle 4.00/4",
                ok,wall,'research/baselines/RESULTS.md section 4; research/crystallization-ablation/RESULTS.md',
                'tcn demo --only joint',detail)

def _demo_structure(out,quick):
    """Generalization to Boolean gate families never trained on."""
    import time,torch
    _track('structure-generalization');import common as C
    from .generation import Host
    start=time.perf_counter();torch.set_num_threads(1)
    episodes=96 if quick else 320
    base={'depth':1,'inputs':2,'fixed_inputs':True,'nondegenerate':True}
    pool_a,pool_b=(1,2,13,14),(4,6,9,11)
    seen_schedule=C.table_schedule(pool_a,base=base);unseen_schedule=C.table_schedule(pool_b,base=base)
    torch.manual_seed(0);settings=seen_schedule(0,'train')
    names,model=C.program_scaffold(Host.create('logic',configuration=settings))
    C.residual_init(model)
    trainer=C.ScheduledTrainer(model,C.make_config(names,settings,episodes,0),seen_schedule)
    trainer.run()
    indices=range(10000,10032) if quick else range(10000,10064)
    seen=C.summarize(C.deterministic_returns(trainer,indices,'test',seen_schedule))
    unseen=C.summarize(C.deterministic_returns(trainer,indices,'test',unseen_schedule))
    reference_a=C.baselines(indices,'test',seen_schedule);reference_b=C.baselines(indices,'test',unseen_schedule)
    node=next(n for n in model.program.nodes if n.name=='relation')
    chosen=model.selections()['relation'];interpreter=chosen==len(node.candidates)-1
    best=max(reference_a['majority_constant'],reference_b['majority_constant'])
    ok=interpreter and unseen['mean']>best+.5
    detail={'training_pool':pool_a,'held_out_pool':pool_b,'episodes':episodes,
            'evaluation_episodes':len(list(indices)),'seen_return':seen,'unseen_return':unseen,
            'relation_candidates':len(node.candidates),'relation_selected':chosen,
            'interpreter_candidate_selected':interpreter,
            'baselines_seen_pool':reference_a,'baselines_unseen_pool':reference_b,
            'recorded_8_seed_unseen':{'pool_a_to_b':3.74,'pool_b_to_a':4.00,
                                      'record_scaffold_unseen':1.95,
                                      'interpreter_selected_seeds':'8/8 in every condition'},
            'recorded_best_constant':{'pool_a':2.13,'pool_b':2.56},
            'recorded_enumeration':{'space':272,'held_out_return':4.00,'seconds':130}}
    return _row('structure','held-out Boolean gate families, interpreter candidate not supplied',
                f"unseen pool {unseen['mean']:.2f}/4 (seen {seen['mean']:.2f}/4), "
                f"interpreter candidate chosen from {len(node.candidates)}: {interpreter}",
                f"best constant {best:.2f}/4; the recorded scaffold measures 1.95-2.03 and cannot exceed 2.00",
                ok,time.perf_counter()-start,'research/nondegenerate-generalization/RESULTS.md',
                'tcn demo --only structure',detail)

def _demo_depth(out,quick):
    """One fixed graph trained at depths 1-2, evaluated at 3, 4, 6 and 8."""
    import time,torch
    _track('depth-generalization');import interpreter as I
    from .agent import Agent
    from .generation import Host
    from .learning import SoftProgram
    from .search import space_size
    start=time.perf_counter();torch.set_num_threads(1)
    width,capacity,horizon,scored=4,8,12,4
    base={'inputs':width,'gate_capacity':capacity,'nondegenerate':True,'min_relevant_inputs':2}
    episodes=64 if quick else 320
    host=Host.create('logic',seed=0,index=0,split='train',configuration=base|{'depth':1,'horizon':horizon})
    names,program,registry=I.interpreter_scaffold(host,width,capacity,horizon,wire_choice=False,settle_mux=True)
    model=SoftProgram(program,registry)
    # SoftProgram zero-initializes every choice logit, so torch.manual_seed does
    # not vary synthesis at all (FINDINGS section 11, fault P2). Perturbing is
    # what makes a seed mean something here.
    I.perturb(model,.05,0)
    config=I.make_config(names,base|{'depth':1},episodes,0,horizon)
    trainer=I.ScheduledTrainer(model,config,I.depth_schedule((1,2),base),scored)
    trainer.run()
    exported=model.export();indices=tuple(range(10000,10016) if quick else range(10000,10064))
    returns={};references={}
    for depth in (1,2,3,4,6,8):
        schedule=I.fixed_depth(depth,base);agent=Agent(exported,registry,config,seed=0);scores=[]
        for i in indices:
            episode=Host.create('logic',seed=config.seed,index=i,split='test',
                                configuration=dict(schedule(i,'test'))|{'horizon':horizon},
                                objective=I.OBJECTIVES[i%len(I.OBJECTIVES)])
            agent.rollout(episode,deterministic=True);scores.append(I.settled_return(episode,scored))
        returns[depth]=sum(scores)/len(scores)
        references[depth]=I.baselines(indices,'test',schedule,0,horizon,scored)['best_constant']
    unseen=[d for d in (3,4,6,8)]
    ok=all(returns[d]>=3.9 for d in (1,2)+tuple(unseen))
    detail={'trained_depths':[1,2],'episodes':episodes,'space_size':space_size(program),
            'evaluation_episodes':len(indices),'exact_frozen_return':returns,
            'best_constant':references,'selections':model.selections(),
            'observation_width':{'program':'3 x depth, so a fixed-width program type-errors at unseen depth',
                                 'gates':'40 at every depth, the set-shaped view that makes this expressible'},
            'recorded':{'pinned_interpreter':'4.00 at every depth, sd 0.00 over 8 seeds',
                        'record_scaffold':{'1':2.12,'2':2.12,'3':1.62,'4':1.50,'6':2.06,'8':2.44},
                        'enumeration':{'space':272,'seconds':25,'unique':True}}}
    worst=min(returns[d] for d in unseen);worst_reference=max(references[d] for d in unseen)
    return _row('depth','trained at depths 1-2 only, scored at depths never trained on',
                'exact frozen program '+', '.join(f'd{d} {returns[d]:.2f}' for d in (1,2,3,4,6,8)),
                'best constant '+', '.join(f'd{d} {references[d]:.2f}' for d in (1,2,3,4,6,8)),
                ok,time.perf_counter()-start,'research/depth-generalization/RESULTS.md',
                'tcn demo --only depth',detail)

def _demo_segmentation(out,quick):
    """Foreground/background from raw pixels, with a uniqueness certificate."""
    import time
    _track('perception-ladder');import rung3_geometry as G
    from .operators import Registry
    from .search import enumerate_fit,evaluate,space_size
    start=time.perf_counter();registry=Registry()
    # The full byte alphabet, not a curated pool: the program has to recover the
    # renderer's background colour from 0-255 with nothing pre-digested.
    program=G.centre_program(registry,2,free_address=False,pool=tuple(range(256)))
    signals=G.centre_signal()
    train=G.examples(range(0,24),2);held=G.examples(range(9000,9048),2)
    result=enumerate_fit(program,train,signals,registry=registry,tolerance=1e-3,max_programs=1<<22)
    error=evaluate(program,result.selections,held,signals,registry) if result.solved else None
    labels=[bool(example['targets']['centre'].decoded) for example in held]
    majority=max(sum(labels),len(labels)-sum(labels))/len(labels)
    ok=bool(result.solved and result.exhausted and result.unique and error==0.)
    detail={'space_size':space_size(program),'evaluated':result.evaluated,'exhausted':result.exhausted,
            'unique':result.unique,'conforming':result.conforming,'search_seconds':result.seconds,
            'selections':result.selections,'training_episodes':24,'held_out_episodes':48,
            'held_out_max_error':error,'majority_class_accuracy':majority,
            'recorded':{'gradient_seeds_exact':'12/12 at R=2 and R=4, held-out error 0.0',
                        'median_gradient_seconds':43.4,'enumeration_seconds':7.87},
            'note':'supervision is an equivalence class of the generator depth probe; the input '
                   'is the raw byte tuple with role="byte", so only eq, index and pack are legal '
                   'on it and no arithmetic on a pixel is expressible.'}
    return _row('segmentation','background colour recovered from raw pixels over the full byte alphabet',
                f"held-out max |err| {error} on 48 unseen episodes",
                f"unique among {space_size(program):,} programs in {result.seconds:.1f} s; "
                f"constant predictor {majority:.3f} accuracy",
                ok,time.perf_counter()-start,'research/perception-ladder/RESULTS.md §4',
                'tcn demo --only segmentation',detail)

# The rung-3 module this staging starts from, as searched exhaustively in
# research/discrete-perception. Re-deriving it is a 32,000-program sweep costing
# 111 s; the demo verifies the frozen module is exact at every position instead,
# which is the property the staged search actually depends on.
_FOREGROUND_MODULE={'pos':0,'obs':0,'g_at':0,'b_at':0,'red':0,'green':0,'blue':0,
                    'cmp_r':0,'cmp_g':0,'cmp_b':2,'rg':0,'foreground':1,'record':0}

def _demo_edge(out,quick):
    """A two-position spatial operator, reachable only by staging."""
    import time
    _track('discrete-perception')
    import rung3_mask as M,rung35_window as W
    from common import accuracy,enumerate_reference,exact_error
    from .operators import Registry
    from .search import space_size
    start=time.perf_counter();registry=Registry();resolution=8;tolerance=1e-6
    train_seeds=tuple(range(4 if quick else 8));held_seeds=tuple(range(100,100+(4 if quick else 8)))
    scaffold=M.module_scaffold(registry,resolution,M.POOL);signals=M.signals()
    every_train=M.pixel_examples(train_seeds,resolution,'train',None,seed=11,objects=6)
    every_held=M.pixel_examples(held_seeds,resolution,'test',None,seed=12,objects=6)
    module_train=accuracy(scaffold,every_train,signals,registry,selections=_FOREGROUND_MODULE)
    module_held=accuracy(scaffold,every_held,signals,registry,selections=_FOREGROUND_MODULE)
    name=registry.register_module(scaffold.harden(_FOREGROUND_MODULE))
    staged=W.staged_scaffold(registry,resolution,name);edge=W.edge_signals()
    train=W.window_examples(train_seeds,resolution,'train',None,seed=3,objects=6)
    held=W.window_examples(held_seeds,resolution,'test',None,seed=4,objects=6)
    result=enumerate_reference(staged,train,edge,registry,tolerance=tolerance)
    error=exact_error(staged,held,edge,registry,selections=result['selections']) if result['solved'] else None
    score=accuracy(staged,held,edge,registry,selections=result['selections']) if result['solved'] else 0.
    labels=[bool(example['targets']['edge'].decoded) for example in held]
    majority=max(sum(labels),len(labels)-sum(labels))/len(labels)
    ok=bool(result['solved'] and result['exhausted'] and result['unique'] and error==0.
            and module_train==1. and module_held==1.)
    detail={'staged_space':space_size(staged),'evaluated':result['evaluated'],'unique':result['unique'],
            'search_seconds':result['seconds'],'selections':result['selections'],
            'offsets':list(W.offsets(resolution)),'positions':len(W.window_positions(resolution)),
            'frozen_module_accuracy_train':module_train,'frozen_module_accuracy_held_out':module_held,
            'held_out_max_error':error,'held_out_accuracy':score,'majority_class_accuracy':majority,
            'undecomposed_space':49_152_000_000,'undecomposed_projected_seconds':239_581_823,
            'recorded':{'staged_enumeration_seconds':4.608,'gradient_seeds':'1/4, median 750.8 s',
                        'flat_gradient_seeds':'0/3'}}
    return _row('edge','a learned two-position operator over raw pixels, offset searched',
                f"held-out max |err| {error}, accuracy {score:.3f}",
                f"unique among {space_size(staged)} staged programs in {result['seconds']:.1f} s; "
                f"undecomposed 4.9e10 programs (7.6 years projected); constant predictor {majority:.3f}",
                ok,time.perf_counter()-start,'research/discrete-perception/RESULTS.md §5',
                'tcn demo --only edge',detail)

def _demo_control(out,quick):
    """An external simulator behind the replay contract."""
    import time
    from .generation import Action,Host
    from .types import Value
    from generators.control.generator import torque_type
    start=time.perf_counter()
    limit=2.;kind=torque_type(limit)
    def torque(value):return Action('torque',arguments=(('value',Value.of(kind,value)),))
    def pump(host,steps):
        for _ in range(steps):
            if host.records[-1].done:break
            rate=host.records[-1].latent_states['qvel'].decoded
            host.step((torque(limit if rate>=0 else -limit),),dt=.05)
        return host
    config={'task':'pendulum','horizon':40,'disturbance':.3}
    host=pump(Host.create('control',seed=11,configuration=config),12)
    replayed=host.replay().digest==host.digest
    restored=Host.restore(host.snapshot());same=restored.digest==host.digest
    for _ in range(8):
        host.step((torque(.4),),dt=.05);restored.step((torque(.4),),dt=.05)
    difference=max(abs(a-b) for a,b in zip(host.state['integration'],restored.state['integration']))
    saved=Path(out)/'control-episode.json.gz';Path(out).mkdir(parents=True,exist_ok=True)
    source=pump(Host.create('control',seed=11,configuration=config),12);source.save(saved)
    reloaded=Host.load(saved)
    for _ in range(8):
        source.step((torque(.4),),dt=.05);reloaded.step((torque(.4),),dt=.05)
    round_trip=max(abs(a-b) for a,b in zip(source.state['integration'],reloaded.state['integration']))
    # The task has to be a real control problem or replay proves nothing about
    # it: a zero-torque arm hangs down forever, an energy-pumping one gets up.
    swing={'task':'pendulum','horizon':200,'start':{'qpos':[0.],'qvel':[0.]}}
    driven=pump(Host.create('control',seed=2,configuration=swing),120)
    passive=Host.create('control',seed=2,configuration=swing)
    for _ in range(120):
        if passive.records[-1].done:break
        passive.step(dt=.05)
    def upright(h):return max(r.reward_components['upright'].decoded for r in h.records[1:])
    driven_peak,passive_peak=upright(driven),upright(passive)
    ok=(replayed and same and difference==0. and round_trip==0. and driven_peak>.99 and passive_peak<-.99)
    detail={'replay_digest_matches':replayed,'restore_digest_matches':same,
            'max_state_difference_after_8_steps':difference,'save_reload_state_difference':round_trip,
            'scripted_controller_peak_upright':driven_peak,'zero_torque_peak_upright':passive_peak,
            'episode':str(saved),
            'recorded':{'cross_process_state_bit_identical':True,'contract_overhead_fraction':.7543,
                        'seconds_per_host_step':1.7228e-3,'peak_abs_qvel_during_swing_up':40.49},
            'note':'77% of a typed step is contract rather than physics, and the typing is authored '
                   'rather than derived; MuJoCo qualifies only because mjSTATE_INTEGRATION exists.'}
    return _row('control','MuJoCo behind the generator contract, replay verified not assumed',
                f"replay and restore identical, max |state delta| {difference:g} after 8 further steps",
                f"scripted controller reaches upright {driven_peak:.4f} where zero torque never "
                f"exceeds {passive_peak:.4f}",
                ok,time.perf_counter()-start,'research/external-environments/RESULTS.md §3',
                'tcn demo --only control',detail)

def _demo_abstraction(out,quick):
    """A crystallized module on the output path, against a scaffold too small to
    solve the target any other way."""
    import time
    _track('recursive-abstraction-retest')
    import experiment as E,enumerate_check as K
    start=time.perf_counter()
    # Arm A has no module candidate, arm B is offered MAJ3, arm C the same-size
    # distractor. One scaffold, identical nodes, depths and pools.
    arms={arm:E.run_one((arm,4,'tight',300 if not quick or arm!='C' else 60)) for arm in ('A','B','C')}
    counts=K.solution_count()
    flat=K.run('A',None,4_000_000,False)
    ok=(arms['B']['final_conformant'] and arms['B']['module_on_output_path']
        and not arms['A']['final_conformant'] and not arms['C']['final_conformant']
        and counts['total_solutions']==counts['solutions_using_a_module']
        and not flat['solved'] and flat['exhausted'])
    detail={'arms':{a:{'conformant':r['final_conformant'],'first_conformant_step':r['first_conformant_step'],
                       'module_on_output_path':r['module_on_output_path'],'live_nodes':r['live_nodes'],
                       'description_bits_pruned':r['description_bits_pruned'],
                       'execution_cost_pruned':r['execution_cost_pruned'],'steps_run':r['steps_run'],
                       'seconds':round(r['total_wall_seconds'],1)} for a,r in arms.items()},
            'flat_space_exhausted_no_solution':{'space_size':flat['space_size'],'evaluated':flat['evaluated'],
                                                'exhausted':flat['exhausted'],'solved':flat['solved'],
                                                'seconds':round(flat['seconds'],1)},
            'module_space':counts,
            'chance_rate_module_on_output_path':.816,
            'recorded':{'wide_8_seeds':{'A':'0/8','B':'8/8','C':'0/8'},
                        'tight_24_seeds':{'A':'0/24','B':'19/24','C':'0/24'},
                        'module_on_output_path_of_successes':'27/27',
                        'fisher_p':{'wide':1.6e-4,'tight':7.4e-9}},
            'caveat':'Track 5 recorded the opposite verdict (module on the output path in 0 of 20 runs). '
                     'That was an accounting artifact: Program.execute re-validated per batch row and a '
                     'one-output module resolved to a one-field product, so every call site paid a project '
                     'node. Both are fixed; the search, not the conclusion, was what changed.'}
    return _row('abstraction','a learned sub-program reused, on a scaffold below the flat minimum',
                f"arm B conformant at step {arms['B']['first_conformant_step']}, module on output path "
                f"{arms['B']['module_on_output_path']}, {arms['B']['live_nodes']} live nodes",
                f"arm A (no module) and arm C (distractor) both fail; the flat space of "
                f"{flat['space_size']:,} programs is exhausted with no solution, and "
                f"{counts['solutions_using_a_module']} of the {counts['total_solutions']} solutions in "
                f"{counts['space_size']:,} use the module",
                ok,time.perf_counter()-start,'research/recursive-abstraction-retest/RESULTS.md',
                'tcn demo --only abstraction',detail)

def _demo_positional(out,quick):
    """One frozen module applied at every position, with three caller nodes."""
    import time
    _track('positional-reuse')
    import measure_cost,measure_scaling
    from .operators import Registry
    from .scaffold import positional_scaffold
    from .types import Value,integer
    start=time.perf_counter()
    # The description invariant: the caller is a fixed size while both
    # alternatives grow linearly in the number of positions.
    cost=measure_cost.run(34,(0,1,2))
    # The searched case at one resolution: the shared sub-program is found by
    # enumeration over the same space, then applied at every position.
    small=measure_scaling.run(8)
    # The width claim, through the public helper rather than the track's own.
    resolution=32 if quick else 48
    positions=tuple(3*i for i in range(resolution*resolution));width=3*resolution*resolution
    registry=Registry();carrier=measure_scaling.bytes_type(width)
    pixels=measure_scaling.synthetic(width,positions,None)
    module=registry.register_module(measure_scaling.scaffold(registry,width,searched=False))
    caller=positional_scaffold(registry,carrier,positions,[module],index=integer(16,signed=False))
    applied=time.perf_counter()
    outputs,_=caller.run({'observation':Value.of(carrier,pixels)},registry=registry)
    seconds=time.perf_counter()-applied
    labels=[(pixels[p],pixels[p+1],pixels[p+2])!=measure_scaling.BACKGROUND for p in positions]
    exact=outputs['mapped'].decoded==frozenset((positions[i],labels[i]) for i in range(len(positions)))
    ok=(exact and len(caller.nodes)==3 and small['exact_apply_correct'] and small['caller_nodes']==3
        and small['enumerate_solved'] and cost['shared']['caller_nodes']==3
        and cost['shared']['structural_symbols']==17)
    detail={'wide':{'resolution':resolution,'observation_values':width,'positions':len(positions),
                    'caller_nodes':len(caller.nodes),'exact_apply_correct':exact,
                    'exact_apply_seconds':seconds,'execution_cost':caller.execution_cost(registry)},
            'searched_at_resolution_8':{k:small[k] for k in ('resolution','width','positions','space_size',
                                                             'enumerate_solved','enumerate_seconds',
                                                             'caller_nodes','exact_apply_correct',
                                                             'exact_apply_seconds')},
            'cost_at_32_positions':{'shared':cost['shared'],'per_position':cost['per_position'],
                                    'inlined':cost['inlined'],'module_nodes':cost['module_nodes']},
            'closed_forms':{'shared':'17 structural symbols at every N','per_position':'3N + 13',
                            'inlined':'8N + 1','description_crossover_positions':2},
            'recorded':{'widest_measured':'27,648 values over 9,216 positions',
                        'geometry_end_to_end':'shared module learned from the dense probe, max error 0.0 '
                                              'over 640 held-out pixels, 3 caller nodes for 64 positions'},
            'caveat':'On the geometry demonstration the supervision does not identify the program: 1,952 '
                     'of 32,000 candidates fit all 512 training pixels and 1,584 of those fit all 640 '
                     "held-out pixels, the renderer's own test among them. The claim is that a shared "
                     'sub-program was learned, crystallized and applied everywhere, not that the right '
                     'one was. The whole pattern also sits behind a declared gradient boundary: insert, '
                     'pair and map have no relaxation.'}
    return _row('positional','one frozen module at every position of a wide observation',
                f"{len(positions):,} positions over a {width:,}-value observation with "
                f"{len(caller.nodes)} caller nodes, exact apply correct ({seconds:.0f} s)",
                f"{cost['shared']['structural_symbols']} structural symbols shared against "
                f"{cost['per_position']['structural_symbols']} per-position and "
                f"{cost['inlined']['structural_symbols']} inlined at 32 positions, at "
                f"{cost['shared']['execution_cost']/cost['per_position']['execution_cost']:.2f}x the "
                f"execution cost",
                ok,time.perf_counter()-start,'research/positional-reuse/RESULTS.md',
                'tcn demo --only positional',detail)

# The stage-B program a validation split selects from the ten that conform on
# training episodes. Enumerating that 45,375-program space costs 363 s, and
# declaration order returns a different member that fails at the longest unseen
# length; the demo re-derives stage A, and applies this recorded selection.
_LANGUAGE_RULE={'symbols':101,'plus':0,'minus':4,'answer':6}

def _demo_language(out,quick):
    """A language task learned from raw prompt bytes, lexical unit and all."""
    import time
    _track('language-capability')
    import common as L,scaffolds,baselines as B
    from run_stage_a import examples as position_examples
    from run_stage_b import accuracy,baselines as label_baselines,build_module
    from .search import enumerate_fit,evaluate,space_size
    start=time.perf_counter()
    seeds=300 if quick else 900
    pool=L.dataset(seeds,seed0=0,split='train')
    train=[e for e in pool if e['length'] in (2,4,6)][:12 if quick else 24]
    tests=L.dataset(500 if quick else 1500,seed0=100000,split='test')
    unseen=[e for e in tests if e['length'] not in (2,4,6)]
    # Stage A: the agent is given no tokenizer, so it searches for its own
    # lexical unit -- which byte denotes an opening bracket, over the whole
    # 0-255 alphabet, and where the symbol field starts.
    lexical,registry,signals=scaffolds.stage_a()
    found=enumerate_fit(lexical,position_examples(train),signals,registry,tolerance=1e-6,max_programs=1<<20)
    positions=evaluate(lexical,found.selections,position_examples(unseen[:40]),signals,registry)
    # Stage B: that module is frozen and called at each of 16 positions; the
    # grammaticality rule over the recovered symbol sequence is what is scored.
    module,rule_registry,_=build_module()
    scaffold,_=scaffolds.stage_b(module,rule_registry)
    program=scaffold.harden(_LANGUAGE_RULE).pruned()
    score,per_length=accuracy(program,{n.name:0 for n in program.nodes},rule_registry,unseen)
    constant=label_baselines(unseen)['majority_constant']
    fitted={name:B.fit_predict(train,unseen,feature) for name,feature in B.FEATURES.items()}
    best=max(fitted.values())
    ok=bool(found.solved and found.exhausted and found.unique and positions==0.
            and found.selections['base']==14 and found.selections['open']==40
            and score>=.99 and score>best+.3)
    detail={'stage_a':{'space_size':space_size(lexical),'evaluated':found.evaluated,
                       'exhausted':found.exhausted,'unique':found.unique,
                       'seconds':found.seconds,'selections':found.selections,
                       'held_out_position_max_error':positions,
                       'reading':'base 14, open byte 40 = ASCII "(" -- searched, not supplied'},
            'stage_b':{'space_size':space_size(scaffold),'selection':_LANGUAGE_RULE,
                       'held_out_unseen_length_accuracy':score,'per_length':per_length,
                       'episodes':len(unseen),'majority_constant':constant,
                       'best_fitted_feature':best,'fitted_features':fitted},
            'recorded':{'stage_b_enumeration':'45,375 programs in 363 s, 10 conforming, not unique',
                        'gradient_on_the_same_spaces':'0 of 44 runs conform',
                        'undecomposed_space':41*256*121*5*5*15},
            'caveats':['The lesson named context_free_language does not exercise a stack as '
                       'sampled: its negatives always break the bracket count, so #( == #) and '
                       'balanced agree on 20,000 of 20,000 seeds. What was learned is counting '
                       'over a recovered symbol sequence, not recursion.',
                       'Only 6 of 179 lessons have prompts byte-predictable at a fixed offset, '
                       'which bounds this method to a small corner of the catalogue.',
                       'construction fails to determine answer in 39 of 179 lessons, so dense '
                       'staging is not uniformly available.',
                       'Ten of 45,375 programs conform on training episodes; declaration order '
                       'returns one that fails at length 16. Requiring exactness on a validation '
                       'split containing an unseen length is what separates them.',
                       'Accuracy is 1.000 up to the scaffold\'s declared 16-position capacity and '
                       'chance beyond it. Depth is generalized up to a declared capacity.']}
    return _row('language','a lexical unit and a grammaticality rule learned from raw prompt bytes',
                f"stage A unique among {space_size(lexical):,} (base 14, open byte 40), held-out "
                f"position error {positions}; stage B {score:.3f} on {len(unseen)} episodes at "
                f"lengths never trained on",
                f"majority constant {constant:.3f}, best fitted feature {best:.3f}, random 0.500; "
                f"gradient descent on the same spaces conforms 0 of 44 runs",
                ok,time.perf_counter()-start,'research/language-capability/RESULTS.md',
                'tcn demo --only language',detail)

_DEMO_RUNNERS={'synthesis':_demo_synthesis,'joint':_demo_joint,'structure':_demo_structure,
               'depth':_demo_depth,'abstraction':_demo_abstraction,'positional':_demo_positional,
               'segmentation':_demo_segmentation,'edge':_demo_edge,'control':_demo_control,
               'language':_demo_language}

def _demo_table(rows):
    header=('demo','measured','baseline','verdict')
    body=[(r['demo'],r['measured'],r['baseline'],r['verdict']) for r in rows]
    widths=[max(len(str(x[i])) for x in (header,)+tuple(body)) for i in range(4)]
    widths=[min(w,c) for w,c in zip(widths,(14,66,92,7))]
    def line(cells,pad=' '):
        out=[]
        for cell,width in zip(cells,widths):
            text=str(cell)
            out.append(text if len(text)<=width else text[:width-1]+'…')
        return pad.join(text.ljust(width) for text,width in zip(out,widths)).rstrip()
    return '\n'.join([line(header),line(tuple('-'*w for w in widths),'-')]+[line(x) for x in body])

def _demo_wrap(text,width,indent):
    import textwrap
    return textwrap.fill(text,width,initial_indent=indent,subsequent_indent=indent)

def demo(out='artifacts/demo',only=(),quick=True):
    """Run every demonstrated capability and print it beside its baseline."""
    import time,traceback
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    names=tuple(only) if only else DEMOS
    unknown=[n for n in names if n not in _DEMO_RUNNERS]
    if unknown:raise SystemExit('unknown demo: '+', '.join(unknown))
    rows=[];started=time.perf_counter()
    for name in names:
        print(f'== {name} ...',flush=True)
        target=out/name;target.mkdir(parents=True,exist_ok=True)
        try:
            row=_DEMO_RUNNERS[name](target,quick)
        except Exception as error:
            row=_row(name,'','raised '+type(error).__name__+': '+str(error)[:120],'',False,0.,
                     'see traceback','tcn demo --only '+name,{'traceback':traceback.format_exc()[-2000:]})
        rows.append(row);write_json(target/'result.json',row)
        print(f"   {row['verdict']}  {row['measured']}",flush=True)
        print(_demo_wrap('baseline: '+row['baseline'],96,'   '),flush=True)
    summary={'mode':'quick' if quick else 'full','demos':rows,'seconds':round(time.perf_counter()-started,1),
             'passed':sum(r['ok'] for r in rows),'total':len(rows)}
    write_json(out/'summary.json',summary)
    table=_demo_table(rows)
    (out/'summary.md').write_text('# tcn demo\n\n```\n'+table+'\n```\n\n'+
        '\n'.join(f"- **{r['demo']}** — {r['claim']}\n  - measured: {r['measured']}\n"
                  f"  - baseline: {r['baseline']}\n  - evidence: {r['evidence']}\n"
                  f"  - reproduce: `{r['command']}`\n" for r in rows)+'\n')
    print('\n'+table)
    print(f"\n{summary['passed']}/{summary['total']} demonstrations reproduced "
          f"in {summary['seconds']:.0f} s; artifacts under {out}")
    return summary

def stage_runner(stage,out):
    from .generation import Host
    if stage.operation=='sample':
        cfg=dict(stage.configuration);name=cfg.pop('generator');steps=cfg.pop('steps',2);seed=cfg.pop('seed',0)
        h=Host.create(name,seed=seed,configuration=cfg)
        for _ in range(steps):
            if h.records[-1].done:break
            h.step(dt=.05)
        h.replay();h.save(out/'episode.json.gz');return {'replay':1,'steps':len(h.inputs),'observations':len(h.view().observations)}
    if stage.operation=='synthesize':
        result=mixed(out,stage.configuration.get('steps',300),mode=stage.configuration.get('mode','relax'));return {'exact_conformance':int(result['exact_conformance']),'fully_frozen':int(result['fully_frozen']),'loss':result['loss']}
    if stage.operation=='train':
        from .training import JointTrainer,TrainConfig
        from .learning import SoftProgram
        from .graph import Program
        from .operators import Registry
        r=Registry()
        for spec in stage.configuration.get('modules',[]):r.register_module(Program.from_dict(spec,r))
        t=JointTrainer(SoftProgram(Program.from_dict(stage.configuration['program'],r),r),TrainConfig.from_dict(stage.configuration['training']))
        history=t.run(out)
        evaluations=[t.episode(10000+i,False,'test')[0] for i in range(stage.configuration.get('evaluations',16))]
        result={'episodes':len(history),'evaluation_mean_return':sum(x['return'] for x in evaluations)/len(evaluations),'prediction_loss':sum(x['prediction_loss'] for x in evaluations)/len(evaluations)}
        write_json(out/'evaluation.json',result);return result
    if stage.operation=='joint':return joint(out,stage.configuration.get('episodes',80))
    raise ValueError('unknown stage operation '+stage.operation)

def main(argv=None):
    parser=argparse.ArgumentParser(prog='tcn');sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('generators')
    sample=sub.add_parser('sample');sample.add_argument('generator');sample.add_argument('--seed',type=int,default=0);sample.add_argument('--steps',type=int,default=2);sample.add_argument('--dt',type=float,default=.05);sample.add_argument('--config');sample.add_argument('--actions');sample.add_argument('--out',default='artifacts/episode.json.gz')
    replay=sub.add_parser('replay');replay.add_argument('episode')
    synth=sub.add_parser('synthesize');synth.add_argument('--out',default='artifacts/mixed');synth.add_argument('--steps',type=int,default=300);synth.add_argument('--mode',default='relax',choices=('relax','auto','enumerate','hybrid'))
    train=sub.add_parser('train');train.add_argument('--out',default='artifacts/joint');train.add_argument('--episodes',type=int,default=160);train.add_argument('--config');train.add_argument('--program');train.add_argument('--resume')
    agent=sub.add_parser('agent');agent.add_argument('program');agent.add_argument('--config',required=True);agent.add_argument('--seed',type=int,default=0);agent.add_argument('--steps',type=int);agent.add_argument('--deterministic',action='store_true');agent.add_argument('--out',default='artifacts/agent-episode.json.gz')
    curr=sub.add_parser('curriculum');curr.add_argument('spec');curr.add_argument('--out',default='artifacts/curriculum');curr.add_argument('--workers',type=int,default=1)
    run=sub.add_parser('run');run.add_argument('program');run.add_argument('--inputs')
    export=sub.add_parser('export');export.add_argument('program');export.add_argument('out')
    render=sub.add_parser('render');render.add_argument('episode');render.add_argument('--out',default='artifacts/frames')
    show=sub.add_parser('demo');show.add_argument('--out',default='artifacts/demo')
    show.add_argument('--only',action='append',default=[],choices=DEMOS)
    show.add_argument('--full',action='store_true')
    show.add_argument('--list',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='generators':
        from .generation import manifests
        print(json.dumps(manifests(),indent=2))
    elif args.command=='sample':
        from .generation import Host,Action
        config=json.loads(Path(args.config).read_text()) if args.config else {}
        h=Host.create(args.generator,seed=args.seed,configuration=config)
        rows=json.loads(Path(args.actions).read_text()) if args.actions else []
        for i in range(args.steps):
            if h.records[-1].done:break
            h.step(tuple(Action.from_dict(a) for a in rows[i]) if i<len(rows) else (),args.dt)
        h.replay();h.save(args.out);print(json.dumps({'episode':args.out,'digest':h.digest,'steps':len(h.inputs)}))
    elif args.command=='replay':
        from .generation import Host
        h=Host.load(args.episode);h.replay();print(json.dumps({'replay':'identical','digest':h.digest}))
    elif args.command=='synthesize':
        result=mixed(args.out,args.steps,mode=args.mode);print(json.dumps({k:v for k,v in result.items() if k!='training'},indent=2))
        if not(result['fully_frozen'] and result['exact_conformance']):return 1
    elif args.command=='train':
        if args.config or args.resume:
            from .training import JointTrainer,TrainConfig
            from .graph import Program
            from .learning import SoftProgram
            if args.resume:t=JointTrainer.load(args.resume);t.config.episodes=args.episodes
            else:
                if not args.program:parser.error('--config requires --program')
                t=JointTrainer(SoftProgram(Program.from_dict(json.loads(Path(args.program).read_text()))),TrainConfig.from_dict(json.loads(Path(args.config).read_text())))
            t.run(args.out);print(json.dumps(t.history[-1],indent=2))
        else:print(json.dumps(joint(args.out,args.episodes),indent=2))
    elif args.command=='agent':
        from .runtime import load_program
        from .agent import Agent
        from .training import TrainConfig
        from .generation import Host
        p,r=load_program(args.program);c=TrainConfig.from_dict(json.loads(Path(args.config).read_text()))
        h=Host.create(c.generator,seed=args.seed,split='test',configuration=c.generator_config|{'horizon':args.steps or c.horizon},objective=c.objectives[args.seed%len(c.objectives)])
        Agent(p,r,c,args.seed).rollout(h,args.steps,args.deterministic);h.replay();h.save(args.out)
        print(json.dumps({'steps':len(h.inputs),'return':sum(sum(v.decoded for v in row.reward_components.values()) for row in h.records),'episode':args.out}))
    elif args.command=='curriculum':
        from .curriculum import Curriculum
        results=Curriculum.load(args.spec).run(stage_runner,args.out,args.workers);print(json.dumps(results,indent=2))
        if any(v['status']!='passed' for v in results.values()):return 1
    elif args.command in {'run','export'}:
        from .runtime import load_program,export_executable
        from .types import Value
        p,r=load_program(args.program)
        if args.command=='export':export_executable(p,args.out,r);print(args.out)
        else:
            source=open(args.inputs) if args.inputs else sys.stdin;state=None
            try:
                for line in source:
                    row=json.loads(line);out,state=p.run({k:Value.from_dict(v) for k,v in row['inputs'].items()},state,r)
                    print(json.dumps({'outputs':{k:v.to_dict() for k,v in out.items()},'state':{k:v.to_dict() for k,v in state.items()}}))
            finally:
                if args.inputs:source.close()
    elif args.command=='render':
        from .generation import Host
        from .visuals import render_episode
        paths=render_episode(Host.load(args.episode),args.out);print(json.dumps(paths))
    elif args.command=='demo':
        if args.list:
            print('\n'.join(DEMOS));return 0
        summary=demo(args.out,args.only,not args.full)
        if summary['passed']!=summary['total']:return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
