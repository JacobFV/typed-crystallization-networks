"""Recurrence, live environment return, and the prefix walk with its beam."""
import pytest
from examples.mixed import problem
from tcn.generation import Action, Host
from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.search import (DiscreteProblem, EnvironmentTask, EpisodeLedger, certificate_of,
                        enumerate_environment, enumerate_fit, enumerate_prefix,
                        enumerate_recurrent, evaluate_recurrent, probe_examples, route,
                        scored_ticks, solve, space_size, viability)
from tcn.types import BOOL, Value, floating, integer, product

F = floating()
COUNT = integer(16, signed=False)


# ---------------------------------------------------------------------------
# the certificate is the contract every mode shares
# ---------------------------------------------------------------------------
def test_the_certificate_names_what_a_sweep_established():
    assert certificate_of(True, True, 1) == 'unique'
    assert certificate_of(True, True, 3) == 'complete'
    assert certificate_of(True, True, 0) == 'complete'
    assert certificate_of(False, True, 1) == 'none'
    assert certificate_of(True, False, 1) == 'none'


def test_every_mode_reports_the_same_certificate_on_the_same_problem():
    p, signals, examples = problem()
    flat = enumerate_fit(p, examples, signals, tolerance=.005)
    walk = enumerate_prefix(p, examples, signals, tolerance=.005)
    once = enumerate_recurrent(p, examples, signals, ticks=1, tolerance=.005)
    for found in (flat, walk, once):
        assert found.certificate == 'unique' and found.unique is True
        assert found.selections == flat.selections and found.conforming == flat.conforming
        assert found.evaluated == found.space_size and found.exhausted


def test_stopping_early_is_recorded_as_a_forfeited_certificate():
    p, signals, examples = problem()
    early = enumerate_fit(p, examples, signals, tolerance=.005, stop_at_first=True)
    budget = enumerate_fit(p, examples, signals, tolerance=.005, max_programs=4)
    assert early.certificate == 'none' and early.unique is None
    assert budget.certificate == 'none' and budget.unique is None


# ---------------------------------------------------------------------------
# recurrence
# ---------------------------------------------------------------------------
def counter_program(registry):
    """`total' = total + step`, where the step size is the discrete choice."""
    constants = tuple((f'k{i}', Value.of(COUNT, i)) for i in range(4))
    node = Node('next', COUNT,
                tuple(Candidate(registry.resolve('add', (COUNT, COUNT)), ('total', f'k{i}'))
                      for i in range(4)), 'core', 1)
    return Program((('x', BOOL),), (node,), (('out', 'next'),), constants,
                   state=(('total', Value.of(COUNT, 0), 'next'),)).validate(registry)


def counter_example(final, per_tick=None):
    row = {'inputs': {'x': Value.of(BOOL, True)}, 'targets': {'total': Value.of(COUNT, final)}}
    if per_tick is not None:
        row['tick_targets'] = [{'total': Value.of(COUNT, v)} for v in per_tick]
    return row


def test_a_recurrent_program_is_scored_where_the_feed_forward_mode_cannot_reach_it():
    r = Registry(); p = counter_program(r)
    signals = (Signal('next', 'total', ('core',), COUNT),)
    # After five ticks of +3 the running total is 15. The feed-forward mode sees
    # only tick 0, where every candidate reads the initial state and no program
    # can produce 15; the recurrent mode reaches it and certifies it unique.
    examples = [counter_example(15)]
    assert not enumerate_fit(p, examples, signals, r, tolerance=1e-9).solved
    found = enumerate_recurrent(p, examples, signals, r, ticks=5, tolerance=1e-9)
    assert found.solved and found.exhausted and found.unique is True
    assert found.certificate == 'unique' and found.selections == {'next': 3}
    assert found.mode == 'recurrent' and found.ticks == 5 and found.settle_window == 1
    assert found.evaluated == found.space_size == space_size(p)


def test_a_settle_window_requires_the_answer_to_have_arrived_and_stayed():
    r = Registry(); p = counter_program(r)
    signals = (Signal('next', 'total', ('core',), COUNT),)
    # +0 holds the total at 0 at every tick, so it survives a window of any width.
    # No other step size can, since the total strictly increases.
    held = [counter_example(0)]
    assert enumerate_recurrent(p, held, signals, r, ticks=5, settle_window=1).selections == {'next': 0}
    stable = enumerate_recurrent(p, held, signals, r, ticks=5, settle_window=5)
    assert stable.solved and stable.conforming == 1 and stable.settle_window == 5
    # A target that is only correct on the final tick fails a wider window.
    transient = [counter_example(15)]
    assert enumerate_recurrent(p, transient, signals, r, ticks=5, settle_window=1).solved
    assert not enumerate_recurrent(p, transient, signals, r, ticks=5, settle_window=2).solved


def test_per_tick_targets_score_the_whole_trajectory():
    r = Registry(); p = counter_program(r)
    signals = (Signal('next', 'total', ('core',), COUNT),)
    examples = [{'inputs': {'x': Value.of(BOOL, True)},
                 'tick_targets': [{'total': Value.of(COUNT, v)} for v in (2, 4, 6)]}]
    found = enumerate_recurrent(p, examples, signals, r, ticks=3, tolerance=1e-9)
    assert found.selections == {'next': 2} and found.unique is True
    schedule = scored_ticks(examples[0], 3, 1)
    assert sorted(schedule) == [0, 1, 2]


def test_a_recurrent_sweep_reports_its_configuration_and_rejects_an_impossible_window():
    r = Registry(); p = counter_program(r)
    signals = (Signal('next', 'total', ('core',), COUNT),)
    with pytest.raises(ValueError):
        evaluate_recurrent(p, {'next': 0}, [counter_example(0)], signals, r, ticks=2, settle_window=3)
    with pytest.raises(ValueError):
        enumerate_recurrent(p, [counter_example(0)], signals, r, ticks=0)


def test_an_input_sequence_may_vary_per_tick():
    r = Registry()
    node = Node('next', COUNT, (Candidate(r.resolve('add', (COUNT, COUNT)), ('total', 'x')),),
                'core', 1)
    p = Program((('x', COUNT),), (node,), (('out', 'next'),),
                state=(('total', Value.of(COUNT, 0), 'next'),)).validate(r)
    signals = (Signal('next', 'total', ('core',), COUNT),)
    rows = [{'inputs': [{'x': Value.of(COUNT, v)} for v in (1, 2, 3)],
             'targets': {'total': Value.of(COUNT, 6)}}]
    assert enumerate_recurrent(p, rows, signals, r, ticks=3, tolerance=1e-9).solved


# ---------------------------------------------------------------------------
# live environment return
# ---------------------------------------------------------------------------
LOGIC = {'depth': 1, 'table': 6, 'fixed_inputs': True}
ANSWERS = (Action('answer', arguments=(('value', Value.of(BOOL, False)),)),
           Action('answer', arguments=(('value', Value.of(BOOL, True)),)))


def logic_program():
    """Decide the answer from the two wired bits; the relation is the choice."""
    r = Registry()
    host = Host.create('logic', configuration=LOGIC)
    inputs = ((('bits', host.view().observations['bits'].type)), ('goal', BOOL),
              ('action', product(F, F)), ('dt', F))
    constants = (('half', Value.of(F, .5)),)
    nodes = [Node(f'bit_{i}', BOOL,
                  (Candidate(r.resolve('project', (inputs[0][1],), parameters={'index': i}), ('bits',)),),
                  'observation', 1) for i in range(2)]
    nodes.append(Node('decision', BOOL,
                      tuple(Candidate(r.resolve(f'truth_{t}', (BOOL, BOOL)), ('bit_0', 'bit_1'))
                            for t in range(16)), 'latent', 2))
    nodes.append(Node('z', F, (Candidate(r.resolve('encode', (BOOL,), F), ('decision',)),), 'encoding', 3))
    nodes.append(Node('lo', F, (Candidate(r.resolve('sub', (F, F)), ('half', 'z')),), 'policy', 4))
    nodes.append(Node('hi', F, (Candidate(r.resolve('sub', (F, F)), ('z', 'half')),), 'policy', 4))
    nodes.append(Node('policy', product(F, F), (Candidate(r.resolve('tuple', (F, F)), ('lo', 'hi')),), 'policy', 5))
    return Program(inputs, tuple(nodes), (('policy', 'policy'),), constants).validate(r), r


def logic_task(indices, split='train'):
    return EnvironmentTask(generator='logic', observations=('bits', 'goal'),
                           action_templates=ANSWERS, generator_config=LOGIC,
                           objectives=({'invert': False},), indices=tuple(indices),
                           horizon=4, split=split)


def test_a_program_is_scored_by_the_return_it_actually_earns():
    program, registry = logic_program()
    ledger = EpisodeLedger()
    found = enumerate_environment(program, logic_task(range(8)), registry, threshold=4., ledger=ledger)
    # xor is the generator's table and the objective is not inverted, so exactly
    # one of the sixteen relations answers correctly at every tick.
    assert found.solved and found.selections['decision'] == 6
    assert found.exhausted and found.unique is True and found.certificate == 'unique'
    assert found.best_return == 4. and found.threshold == 4. and found.exact_max_error == 0.
    assert found.mode == 'environment'
    # Environment episodes are a reported cost, not a hidden one.
    assert found.episodes == 16 * 8 == ledger.episodes and found.steps == ledger.steps > 0


def test_return_under_identifies_the_program_at_a_small_episode_budget():
    """Reward is coarse: two relations agree on four episodes and separate on eight.

    The sweep is still exhaustive, so it reports the ambiguity as `conforming=2`
    and a `'complete'` rather than a `'unique'` certificate instead of picking one
    and calling it the answer.
    """
    program, registry = logic_program()
    thin = enumerate_environment(program, logic_task(range(4)), registry, threshold=4.)
    assert thin.exhausted and thin.conforming == 2 and thin.certificate == 'complete'
    assert thin.unique is False


def test_an_episode_budget_stops_the_sweep_and_forfeits_the_certificate():
    program, registry = logic_program()
    found = enumerate_environment(program, logic_task(range(2)), registry, threshold=4.,
                                  max_episodes=6)
    assert found.episodes <= 6 and not found.exhausted
    assert found.unique is None and found.certificate == 'none'


def test_without_a_threshold_conformance_means_best_in_the_enumerated_set():
    program, registry = logic_program()
    found = enumerate_environment(program, logic_task(range(8)), registry)
    assert found.solved and found.best_return == 4. and found.threshold is None
    assert found.selections['decision'] == 6 and found.certificate == 'unique'


def test_an_early_exit_needs_a_threshold_to_be_meaningful():
    program, registry = logic_program()
    with pytest.raises(ValueError):
        enumerate_environment(program, logic_task(range(2)), registry, stop_at_first=True)


def test_probe_examples_charge_the_same_ledger_a_reward_sweep_does():
    program, registry = logic_program()
    ledger = EpisodeLedger()
    rows = probe_examples(logic_task(range(5)), (('target', 'probes', 'target'),), ledger,
                          extra_inputs={'action': Value.of(product(F, F), (1., 0.)),
                                        'dt': Value.of(F, 1.)})
    assert len(rows) == 5 and ledger.episodes == 5
    assert set(rows[0]['inputs']) == {'bits', 'goal', 'action', 'dt'}
    found = enumerate_environment(program, logic_task((90000,)), registry, threshold=4., ledger=ledger)
    assert ledger.episodes == 5 + found.space_size
    assert found.episodes == ledger.episodes            # one honest total across both stages


# ---------------------------------------------------------------------------
# the prefix walk and its beam
# ---------------------------------------------------------------------------
def chain(registry, length=3):
    sources = [('x0', 'x1'), ('h0', 'x0'), ('h1', 'x1')][:length]
    nodes = tuple(Node(f'h{i}', BOOL,
                       tuple(Candidate(registry.resolve(f'truth_{t}', (BOOL, BOOL)), src)
                             for t in range(16)), 'core', i + 1)
                  for i, src in enumerate(sources))
    return Program((('x0', BOOL), ('x1', BOOL)), nodes, (('y', f'h{length-1}'),)).validate(registry)


def chain_data(registry, program, reference, length=3):
    import itertools
    rows = []
    for x0, x1 in itertools.product((False, True), repeat=2):
        inputs = {'x0': Value.of(BOOL, x0), 'x1': Value.of(BOOL, x1)}
        _, _, trace = program.execute(inputs, registry=registry, selections=reference)
        rows.append({'inputs': inputs, 'targets': {f'h{i}': trace[f'h{i}'] for i in range(length)}})
    return rows


def test_the_prefix_walk_returns_the_same_conforming_set_for_far_less_work():
    r = Registry(); p = chain(r)
    reference = {'h0': 6, 'h1': 1, 'h2': 9}
    rows = chain_data(r, p, reference)
    signals = tuple(Signal(f'h{i}', f'h{i}', ('core',), BOOL, 'bce') for i in range(3))
    flat = enumerate_fit(p, rows, signals, r, tolerance=1e-9)
    walk = enumerate_prefix(p, rows, signals, r, tolerance=1e-9)
    assert flat.conforming == walk.conforming and flat.selections == walk.selections
    assert walk.evaluated == walk.space_size == 16 ** 3 and walk.exhausted
    assert walk.certificate == flat.certificate and walk.unique == flat.unique
    assert walk.node_evaluations < flat.node_evaluations / 100
    assert walk.mode == 'prefix' and walk.discarded == 0


def test_a_beam_that_discards_forfeits_the_certificate_and_says_so():
    r = Registry(); p = chain(r)
    reference = {'h0': 6, 'h1': 1, 'h2': 9}
    rows = chain_data(r, p, reference)
    signals = (Signal('h2', 'h2', ('core',), BOOL, 'bce'),)     # output supervision only
    exhaustive = enumerate_prefix(p, rows, signals, r, tolerance=1e-9)
    assert exhaustive.certificate in {'unique', 'complete'} and exhaustive.discarded == 0
    narrow = enumerate_prefix(p, rows, signals, r, tolerance=1e-9, beam=2)
    assert narrow.mode == 'beam' and narrow.beam == 2
    assert narrow.discarded > 0 and not narrow.exhausted
    assert narrow.unique is None and narrow.certificate == 'none'
    assert narrow.conforming < exhaustive.conforming


def test_a_beam_wide_enough_to_drop_nothing_keeps_the_certificate():
    r = Registry(); p = chain(r)
    reference = {'h0': 6, 'h1': 1, 'h2': 9}
    rows = chain_data(r, p, reference)
    signals = tuple(Signal(f'h{i}', f'h{i}', ('core',), BOOL, 'bce') for i in range(3))
    wide = enumerate_prefix(p, rows, signals, r, tolerance=1e-9, beam=4096)
    assert wide.discarded == 0 and wide.exhausted
    assert wide.certificate == enumerate_prefix(p, rows, signals, r, tolerance=1e-9).certificate


def test_prefix_reuse_is_refused_on_a_recurrence_rather_than_being_unsound():
    r = Registry(); p = counter_program(r)
    signals = (Signal('next', 'total', ('core',), COUNT),)
    with pytest.raises(ValueError, match='unsound'):
        enumerate_prefix(p, [counter_example(15)], signals, r)


# ---------------------------------------------------------------------------
# routing, for a selector that is not on main yet
# ---------------------------------------------------------------------------
def test_routing_sends_each_problem_shape_to_the_mode_that_can_score_it():
    r = Registry()
    p, signals, examples = problem()
    assert route(DiscreteProblem(p, tuple(examples), signals)) == 'fit'
    assert route(DiscreteProblem(p, tuple(examples), signals, prefix=True)) == 'prefix'
    assert route(DiscreteProblem(p, tuple(examples), signals, beam=8)) == 'beam'
    recurrent = counter_program(r)
    rows = (counter_example(15),)
    sig = (Signal('next', 'total', ('core',), COUNT),)
    assert route(DiscreteProblem(recurrent, rows, sig, ticks=5, registry=r)) == 'recurrent'
    program, registry = logic_program()
    assert route(DiscreteProblem(program, task=logic_task(range(1)), registry=registry)) == 'environment'
    assert route(DiscreteProblem(p)) is None
    with pytest.raises(ValueError):
        solve(DiscreteProblem(p))


def test_solve_runs_the_routed_mode():
    r = Registry()
    p, signals, examples = problem()
    found = solve(DiscreteProblem(p, tuple(examples), signals, tolerance=.005))
    assert found.mode == 'fit' and found.certificate == 'unique'
    recurrent = counter_program(r)
    rows = (counter_example(15),)
    sig = (Signal('next', 'total', ('core',), COUNT),)
    found = solve(DiscreteProblem(recurrent, rows, sig, ticks=5, tolerance=1e-9, registry=r))
    assert found.mode == 'recurrent' and found.selections == {'next': 3}


def test_viability_reports_where_a_mode_stops():
    r = Registry(); p = chain(r)
    note = viability(p, examples=4, rate=2e4)
    assert note['space_size'] == 16 ** 3 and note['executions'] == 16 ** 3 * 4
    assert note['exhaustible'] is True
    assert viability(p, examples=4, episodes=8)['environment_episodes'] == 16 ** 3 * 8
