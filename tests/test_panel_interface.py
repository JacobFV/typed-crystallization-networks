"""The gated `panel` interface of `generators/computer`: contract, gating, and delay.

`research/credit-assignment/RESULTS.md` is the study these invariants support. The
point of the interface is a *narrow typed action argument* -- a bounded integer, two
policy parameters -- where the shell interface's `write.text` alone is 1026 parameters
scored by exact string equality, and a reward that arrives strictly after the
actions that earn it.
"""
import copy
import pytest
from tcn.generation import Host,Action,text_value
from tcn.types import Value
from tcn.policy import parameter_width,numeric_bounds
from generators.computer.generator import (SLOT,DIAL,TEXT,PATH,SHELL_MENU,PANEL_MENU,
                                           PANEL_OUT,Implementation)

# The panel tests run over the gated long-lived transport, which is ~4x cheaper than
# rebooting the kernel per transition; `test_session_transport_is_transparent` checks
# the two transports produce identical records, and every other assertion here holds
# under either.
PANEL={'interface':'panel','horizon':6,'session':'tests'}
BRIDGE={'interface':'panel','horizon':6}
def look(k):return Action('look',arguments=(('slot',Value.of(SLOT,k)),))
def dial(v):return Action('dial',arguments=(('value',Value.of(DIAL,v)),))
COMMIT=Action('commit',arguments=())


def solve(host):
    """The oracle plan: sweep to the task record, dial the successor, commit."""
    panel=host.state['panel'];total=0.
    for k in list(range(panel['task']+1)):
        record=host.step((look(k),));total+=record.reward_components['goal'].decoded
    for action in (dial(panel['answer']),COMMIT):
        record=host.step((action,));total+=record.reward_components['goal'].decoded
        if record.done:break
    return total,host


def test_panel_menu_and_default_shell_menu_are_disjoint_and_stable():
    shell=Host.create('computer',configuration={'horizon':2,'session':'tests'})
    assert shell.view().available_actions==SHELL_MENU==('wait','command','type','key','read','write')
    panel=Host.create('computer',configuration=PANEL)
    assert panel.view().available_actions==PANEL_MENU==('wait','look','dial','commit')
    # the panel verbs exist in the schema but are not offered, and are refused,
    # outside the panel interface -- and vice versa
    with pytest.raises(ValueError):shell.step((COMMIT,))
    with pytest.raises(ValueError):
        panel.step((Action('read',arguments=(('path',text_value('/home/agent/slot0.txt',128)),)),))


def test_action_argument_is_narrow():
    """Two policy parameters per argument, against 1026 for the shell's `write`."""
    assert parameter_width(SLOT)==2 and numeric_bounds(SLOT,(-1.,1.))==(0.,3.)
    assert parameter_width(DIAL)==2 and numeric_bounds(DIAL,(-1.,1.))==(0.,15.)
    assert parameter_width(TEXT)==1026                    # FINDINGS section 23's figure
    assert parameter_width(PATH)+parameter_width(TEXT)==1284


def test_reward_is_delayed_and_terminal():
    host=Host.create('computer',seed=3,index=1,configuration=PANEL)
    panel=host.state['panel']
    rewards=[]
    for action in (look(panel['task']),dial(panel['answer']),COMMIT):
        record=host.step((action,));rewards.append(record.reward_components['goal'].decoded)
    assert rewards[:-1]==[0.]*(len(rewards)-1)   # nothing before the commit pays
    assert rewards[-1]==1.                        # the commit pays, once
    assert host.records[-1].done                  # and ends the episode


def test_commit_reads_the_filesystem_not_the_register():
    """A wrong dial commits a wrong file, and the reward is read back off it."""
    host=Host.create('computer',seed=3,index=2,configuration=PANEL)
    panel=host.state['panel'];wrong=(panel['answer']%9)+1
    assert wrong!=panel['answer']
    host.step((dial(wrong),))
    record=host.step((COMMIT,))
    assert record.reward_components['goal'].decoded==0.
    assert record.probes['content_4_present'].decoded is True      # /home/agent/out.txt exists
    body=record.probes['content_4'].decoded
    assert bytes(body[1][:body[0]]).decode()==str(wrong)


def test_a_myopic_commit_scores_far_below_the_oracle():
    myopic=0.;oracle=0.
    for index in range(12):
        host=Host.create('computer',seed=0,index=index,split='test',configuration=PANEL)
        myopic+=host.step((COMMIT,)).reward_components['goal'].decoded
        total,_=solve(Host.create('computer',seed=0,index=index,split='test',configuration=PANEL))
        oracle+=total
    assert oracle==12.          # the plan always wins
    assert myopic<=2.           # committing at once is a 1/16 lottery


def test_panel_probes_are_not_visible_to_the_actor():
    host=Host.create('computer',configuration=PANEL)
    record=host.records[-1]
    assert {'answer','task_slot','showing_task'}<=set(record.probes)
    assert 'register' in record.latent_states
    view=record.actor_view('agent_0')
    assert set(view.observations)=={'pixels','terminal'}
    assert not hasattr(view,'probes') and not hasattr(view,'latent_states')


def test_panel_episode_replays_and_restores():
    host=Host.create('computer',seed=5,index=4,configuration=PANEL)
    host.step((look(1),));host.step((dial(7),))
    assert host.replay().digest==host.digest
    restored=Host.restore(host.snapshot())
    assert restored.digest==host.digest


def test_session_transport_is_transparent():
    """`engine/session.ts` and `engine/bridge.ts` produce identical episodes."""
    import json
    plan=(look(2),dial(6),COMMIT)
    records=[]
    for configuration in (PANEL,BRIDGE):
        host=Host.create('computer',seed=2,index=9,configuration=configuration)
        for action in plan:
            if host.records[-1].done:break
            host.step((action,))
        records.append([json.dumps(r.to_dict(),sort_keys=True) for r in host.records])
    assert records[0]==records[1]


def test_panel_draws_on_its_own_random_stream():
    """Enabling the panel must not move any draw a shell configuration makes."""
    a=Host.create('computer',seed=11,index=3,configuration={'horizon':1,'session':'tests'})
    b=Host.create('computer',seed=11,index=3,configuration=PANEL)
    assert a.state['seed']==b.state['seed']


def test_out_of_range_slot_is_an_observable_failed_attempt():
    host=Host.create('computer',configuration=PANEL|{'panel_slots':2})
    before=host.records[-1].observations['terminal'].value.decoded
    record=host.step((look(3),))
    assert record.transition['events'].decoded==0
    assert record.observations['terminal'].value.decoded==before
