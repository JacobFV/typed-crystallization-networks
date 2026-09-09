"""The persistent module library and the curriculum artifact flow."""
import json
import pathlib

import pytest

from tcn.curriculum import Artifacts, Curriculum, Stage
from tcn.graph import Candidate, Node, Program
from tcn.library import (FixtureMismatch, Library, LibraryError, MissingDependency,
                         SourceRevisionMismatch, module_dependencies)
from tcn.operators import Registry
from tcn.types import BOOL, Value

CHAIN_LIBRARY = pathlib.Path(__file__).resolve().parents[1] / 'research' / 'module-library' / 'library'


def gate(registry, truth=6):
    op = registry.resolve(f'truth_{truth}', (BOOL, BOOL))
    return Program((('a', BOOL), ('b', BOOL)),
                   (Node('y', BOOL, (Candidate(op, ('a', 'b')),), 'core', 1, 0),),
                   (('y', 'y'),)).validate(registry)


def caller(registry, name, sites=1):
    op = registry.resolve(name, (BOOL, BOOL))
    nodes = [Node(f'call{i}', BOOL, (Candidate(op, ('a', 'b')),), 'core', 1, 0)
             for i in range(sites)]
    current = 'call0'
    for i in range(1, sites):
        current = f'join{i}'
        nodes.append(Node(current, BOOL, (Candidate(registry.resolve('or', (BOOL, BOOL)),
                                                    (nodes[i].name, nodes[i - 1].name if i == 1
                                                     else f'join{i - 1}')),), 'core', 2 + i, 0))
    return Program((('a', BOOL), ('b', BOOL)), tuple(nodes), (('y', current),)).validate(registry)


def cases():
    return [{'a': Value.of(BOOL, a), 'b': Value.of(BOOL, b)}
            for a in (False, True) for b in (False, True)]


def test_publish_survives_the_process_that_learned_it(tmp_path):
    registry = Registry()
    program = gate(registry)
    entry = Library(tmp_path).publish('logic.xor', program, registry, fixture=cases(),
                                      provenance={'stage': 'test'})
    assert entry.reference == 'logic.xor@1'
    assert (tmp_path / 'manifest.json').exists()

    # A completely separate Registry, as a later unrelated run would have.
    fresh, aliases = Library(tmp_path).load(['logic.xor'])
    name = aliases['logic.xor']
    assert name in fresh.modules and name == entry.operator
    op = fresh.resolve(name, (BOOL, BOOL))
    for case in cases():
        got = fresh.exact(op, [case['a'], case['b']])
        assert got.decoded == (case['a'].decoded != case['b'].decoded)


def test_content_addressing_shares_one_definition(tmp_path):
    library = Library(tmp_path)
    registry = Registry()
    first = library.publish('logic.xor', gate(registry), registry, fixture=cases())
    same = library.publish('logic.xor', gate(Registry()), registry, fixture=cases())
    assert same.version == first.version == 1                     # republish is a no-op
    other = library.publish('logic.parity', gate(Registry()), registry, fixture=cases())
    assert other.digest == first.digest
    assert len(list((tmp_path / 'modules').glob('*.json'))) == 1  # counted once on disk


def test_relearning_versions_and_revalidates_dependents(tmp_path):
    library = Library(tmp_path)
    registry = Registry()
    library.publish('logic.gate', gate(registry, 6), registry, fixture=cases())
    name = registry.register_module(gate(registry, 6))
    library.publish('logic.caller', caller(registry, name), registry, fixture=cases())
    assert library.head('logic.caller').requires == (name,)

    second = library.publish('logic.gate', gate(Registry(), 9), registry, fixture=cases())
    assert second.version == 2
    assert library.head('logic.caller').stale == ('logic.gate@2',)
    with pytest.raises(LibraryError):
        library.load(['logic.caller'])
    library.revalidate('logic.caller')
    assert library.head('logic.caller').stale == ()
    library.load(['logic.caller'])                                 # loads once cleared


def test_source_revision_is_refused_loudly_and_revalidation_is_recorded(tmp_path):
    library = Library(tmp_path)
    registry = Registry()
    library.publish('logic.xor', gate(registry), registry, fixture=cases())
    manifest = tmp_path / 'manifest.json'
    payload = json.loads(manifest.read_text())
    payload['entries'][0]['source'] = 'a' * 64
    manifest.write_text(json.dumps(payload))

    with pytest.raises(SourceRevisionMismatch):
        Library(tmp_path).load(['logic.xor'])
    reopened = Library(tmp_path)
    reopened.load(['logic.xor'], policy='revalidate')
    stamped = json.loads(manifest.read_text())['entries'][0]
    assert stamped['revalidated'], 'a crossing must be recorded, not forgotten'


def test_revalidation_is_not_a_bypass(tmp_path):
    library = Library(tmp_path)
    registry = Registry()
    entry = library.publish('logic.xor', gate(registry), registry, fixture=cases())
    payload = json.loads(manifest_path(tmp_path).read_text())
    payload['entries'][0]['source'] = 'b' * 64
    manifest_path(tmp_path).write_text(json.dumps(payload))
    fixture = tmp_path / 'fixtures' / f'{entry.digest}.json'
    recorded = json.loads(fixture.read_text())
    recorded['cases'][1]['outputs']['y']['raw'] = False            # was True
    fixture.write_text(json.dumps(recorded))
    with pytest.raises(FixtureMismatch):
        Library(tmp_path).load(['logic.xor'], policy='revalidate')


def test_a_module_without_a_fixture_cannot_cross_a_revision(tmp_path):
    library = Library(tmp_path)
    registry = Registry()
    library.publish('logic.xor', gate(registry), registry)          # no fixture recorded
    payload = json.loads(manifest_path(tmp_path).read_text())
    payload['entries'][0]['source'] = 'c' * 64
    manifest_path(tmp_path).write_text(json.dumps(payload))
    with pytest.raises(FixtureMismatch):
        Library(tmp_path).load(['logic.xor'], policy='revalidate')


def manifest_path(root):
    return pathlib.Path(root) / 'manifest.json'


def test_missing_dependency_is_named_rather_than_stored(tmp_path):
    library = Library(tmp_path)
    registry = Registry()
    name = registry.register_module(gate(registry))
    with pytest.raises(MissingDependency):
        library.publish('logic.caller', caller(registry, name), registry, fixture=cases())


def test_description_cost_is_transitive_and_charges_a_definition_once(tmp_path):
    library = Library(tmp_path)
    registry = Registry()
    library.publish('logic.gate', gate(registry), registry, fixture=cases())
    fresh, aliases = Library(tmp_path).load(['logic.gate'])
    name = aliases['logic.gate']
    body = fresh.modules[name]
    one = caller(fresh, name, sites=1)
    two = caller(fresh, name, sites=2)
    assert module_dependencies(two) == (name,)
    # The definition is charged once however many call sites there are.
    assert one.description_bits(fresh) - one.description_bits(None) == body.description_bits()
    assert two.description_bits(fresh) - two.description_bits(None) == body.description_bits()
    # Execution is charged per use.
    assert two.execution_cost(fresh) > one.execution_cost(fresh)


def test_verify_reports_every_check(tmp_path):
    library = Library(tmp_path)
    registry = Registry()
    library.publish('logic.xor', gate(registry), registry, fixture=cases())
    row, = library.verify()
    assert row['ok'] and row['loads'] and row['fixture_reproduces'] and row['source_current']
    assert row['fixture_cases'] == 4


def test_inheritance_is_explicit_configuration():
    with pytest.raises(ValueError):                                # not a prerequisite
        Curriculum([Stage('a', (), 'x', {}, {}, publishes=('m',)),
                    Stage('b', (), 'x', {}, {}, inherits=('a',))])
    with pytest.raises(ValueError):                                # publishes nothing
        Curriculum([Stage('a', (), 'x', {}, {}),
                    Stage('b', ('a',), 'x', {}, {}, inherits=('a',))])
    c = Curriculum([Stage('a', (), 'x', {}, {}, publishes=('m',)),
                    Stage('b', ('a',), 'x', {}, {}),
                    Stage('c', ('a',), 'x', {}, {}, inherits=('a',))])
    assert c.inherited('b') == ()                                  # a prerequisite alone gives nothing
    assert c.inherited('c') == ('m',)


def test_artifacts_flow_along_declared_edges_and_publication_is_gated(tmp_path):
    seen = {}

    def runner(stage, path, artifacts):
        seen[stage.name] = tuple(artifacts.modules)
        if stage.publishes:
            registry = Registry()
            Library(artifacts.library).publish(stage.publishes[0], gate(registry), registry,
                                               fixture=cases(), provenance={'stage': stage.name})
        return {'success': 1}

    stages = [Stage('root', (), 'x', {}, {'success': {'min': 1}}, publishes=('logic.xor',)),
              Stage('blind', ('root',), 'x', {}, {'success': {'min': 1}}),
              Stage('heir', ('root',), 'x', {}, {'success': {'min': 1}}, inherits=('root',))]
    records = Curriculum(stages).run(runner, tmp_path)
    assert all(v['status'] == 'passed' for v in records.values())
    assert seen == {'root': (), 'blind': (), 'heir': ('logic.xor',)}
    assert records['root']['published'] == ['logic.xor@1']
    assert Library(tmp_path / 'library').head('logic.xor').version == 1

    # A stage that declares a module and does not publish it does not pass.
    lazy = [Stage('root', (), 'x', {}, {'success': {'min': 1}}, publishes=('logic.and',))]
    failed = Curriculum(lazy).run(lambda s, p, a: {'success': 1}, tmp_path / 'lazy')
    assert failed['root']['status'] == 'failed'
    assert 'module logic.and not published' in failed['root']['failures']


def _timing_runner(stage, out, artifacts):
    """Module-level so it survives the spawn pool; records through the filesystem."""
    import json, time
    start = time.perf_counter()
    time.sleep(0.05)
    out.mkdir(parents=True, exist_ok=True)
    (out / (stage.name + '.timing.json')).write_text(json.dumps([start, time.perf_counter()]))
    return {}


def test_a_publishing_stage_never_runs_concurrently(tmp_path):
    """The manifest has one writer, but only the publishing stage pays for it.

    The guard used to refuse `workers>1` outright, which broke the curriculum
    command documented in README.md. The invariant is narrower: a publishing
    stage runs alone, everything else keeps the caller's width.
    """
    import json
    stages = [Stage('a', (), 'x', {}, {}), Stage('b', (), 'x', {}, {}),
              Stage('pub', (), 'x', {}, {}, publishes=('m',))]
    # `pub` declares a module its runner never writes, so it fails its own
    # evidence gate. That is irrelevant here and deliberately tolerated: the
    # invariant under test is the SCHEDULE, and a stage is scheduled before it
    # is judged.
    Curriculum(stages).run(_timing_runner, tmp_path, workers=3)
    intervals = {q.name.split('.')[0]: json.loads(q.read_text())
                 for q in tmp_path.rglob('*.timing.json')}
    assert set(intervals) == {'a', 'b', 'pub'}, intervals

    def overlaps(x, y):
        return x[0] < y[1] and y[0] < x[1]

    for name in ('a', 'b'):
        assert not overlaps(intervals['pub'], intervals[name]), f'publishing stage overlapped {name}'
    # Deliberately NOT asserted: that `a` and `b` overlap each other. The
    # scheduler permits it, but a spawn pool under contention may serialize them
    # anyway, which made that assertion flaky (1 failure in 3 runs) without ever
    # indicating a regression. Non-overlap of the publishing stage is enforced by
    # the scheduler and is therefore the deterministic half of the invariant.


@pytest.mark.skipif(not CHAIN_LIBRARY.exists(), reason='chain library not built')
def test_shipped_chain_library_verifies():
    """Every fixture shipped with the three-stage chain must reproduce."""
    rows = Library(CHAIN_LIBRARY).verify()
    assert rows, 'the shipped library is empty'
    for row in rows:
        assert row['ok'], row
        assert row['fixture_cases'] > 0
    names = [r['reference'].split('@')[0] for r in rows]
    assert 'perception.foreground' in names
    assert 'perception.edge' in names
    assert 'perception.region' in names
    edge = Library(CHAIN_LIBRARY).head('perception.edge')
    foreground = Library(CHAIN_LIBRARY).head('perception.foreground')
    assert edge.requires == (foreground.operator,), 'stage 2 must depend on stage 1'


@pytest.mark.skipif(not CHAIN_LIBRARY.exists(), reason='chain library not built')
def test_library_cli(capsys):
    from tcn.cli import main
    assert main(['library', 'list', '--root', str(CHAIN_LIBRARY)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing['versions'] == len(listing['entries'])
    assert main(['library', 'show', 'perception.edge', '--root', str(CHAIN_LIBRARY)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown['program']['format'] == 'tcn.program/1'
    assert main(['library', 'verify', '--root', str(CHAIN_LIBRARY)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified['ok'] == verified['checked']
