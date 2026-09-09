"""Command-line entry points for synthesis, curricula, episodes, and exact runtime."""
import argparse
import json
from pathlib import Path
import sys

def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False))

def mixed(out,steps=300,baseline_limit=1<<16):
    from examples.mixed import problem
    from .synthesis import fit
    from .runtime import save_program,export_executable,benchmark
    from .search import enumerate_fit,space_size
    p,signals,examples=problem();model,report=fit(p,examples,signals,steps=steps,tolerance=.005)
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
        result=mixed(out,stage.configuration.get('steps',300));return {'exact_conformance':int(result['exact_conformance']),'fully_frozen':int(result['fully_frozen']),'loss':result['loss']}
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
    synth=sub.add_parser('synthesize');synth.add_argument('--out',default='artifacts/mixed');synth.add_argument('--steps',type=int,default=300)
    train=sub.add_parser('train');train.add_argument('--out',default='artifacts/joint');train.add_argument('--episodes',type=int,default=160);train.add_argument('--config');train.add_argument('--program');train.add_argument('--resume')
    agent=sub.add_parser('agent');agent.add_argument('program');agent.add_argument('--config',required=True);agent.add_argument('--seed',type=int,default=0);agent.add_argument('--steps',type=int);agent.add_argument('--deterministic',action='store_true');agent.add_argument('--out',default='artifacts/agent-episode.json.gz')
    curr=sub.add_parser('curriculum');curr.add_argument('spec');curr.add_argument('--out',default='artifacts/curriculum');curr.add_argument('--workers',type=int,default=1)
    run=sub.add_parser('run');run.add_argument('program');run.add_argument('--inputs')
    export=sub.add_parser('export');export.add_argument('program');export.add_argument('out')
    render=sub.add_parser('render');render.add_argument('episode');render.add_argument('--out',default='artifacts/frames')
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
        result=mixed(args.out,args.steps);print(json.dumps({k:v for k,v in result.items() if k!='training'},indent=2))
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
    return 0

if __name__=='__main__':raise SystemExit(main())
