"""Structural limits of the shipped generator, each one measured rather than argued."""
import json,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability')
from tcn.generation import Host,read_text
from tcn.training import TrainConfig,JointTrainer
import task as T

BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'
report={}
started=time.perf_counter()

# L1. The episode address does not reach the task content. `initialize` uses
# `address.rng()` only to seed the kernel; the document is a fixed configuration
# string, and `JointTrainer.episode` passes one `generator_config` for every episode.
terminals=[];digests=[]
for index in (0,1,2,3):
    host=Host.create('computer',seed=0,index=index,split='train',
                     configuration={'document':'count = 7','horizon':2},objective={})
    terminals.append(read_text(host.view().observations['terminal']))
    digests.append(host.records[0].observations['terminal'].value.raw[0])
after_read=[]
for index in (0,1):
    host=Host.create('computer',seed=0,index=index,split='train',
                     configuration={'document':'count = 7','horizon':2},objective={})
    host.step((T.read_action(),))
    after_read.append(read_text(host.records[-1].observations['terminal'].value))
report['L1_address_does_not_vary_the_task']={
 'indices':[0,1,2,3],'distinct_initial_terminals':len(set(terminals)),
 'distinct_terminals_after_a_read':len(set(after_read)),'observed':after_read,
 'consequence':'held-out episode addresses hold out nothing about the task, so a '
   'generalization claim on this generator has to vary `configuration`, which '
   '`TrainConfig` fixes for a whole run'}

# L2. The objective reaches the actor view but no program can read it: `Agent.act`
# and `JointTrainer.inputs` build program inputs from `view.observations` alone.
host=Host.create('computer',seed=0,index=0,configuration={'document':'count = 7','horizon':2},
                 objective={'path':T.TASK_PATH,'content':'8'})
view=host.view()
report['L2_objective_not_consumable']={
 'actor_view_objective':view.objective,
 'observation_keys':sorted(view.observations),
 'objective_in_observations':any('goal' in k or 'objective' in k for k in view.observations),
 'consequence':'goal-conditioned generalization over held-out objectives is not '
   'expressible: the actor cannot see which objective it is being scored against'}

# L3. `TrainConfig` carries one generator configuration for every episode of a run.
config=TrainConfig(generator='computer',observations=('terminal',),targets=(),
                   action_templates=(T.wait_action(),),generator_config={'document':'count = 7'})
report['L3_trainconfig_generator_configuration_is_per_run']={
 'field':'generator_config','type':'dict','used_as':"Host.create(..., configuration=c.generator_config|{'horizon':c.horizon})",
 'varies_with_episode_index':False,'objectives_vary_with_episode_index':True,
 'consequence':'the only thing `JointTrainer` can vary per episode is the objective, '
   'and the objective is invisible to the actor (L2), so the integrated trainer cannot '
   'present this generator with a distribution of tasks'}

report['seconds']=time.perf_counter()-started
open(BASE+'limits.json','w').write(json.dumps(report,indent=2,sort_keys=True,default=str))
print(json.dumps(report,indent=2,sort_keys=True,default=str))
