"""Contract, probe typing and recoverability invariants for `generators/gui`."""
import copy
import json

import numpy as np
import pytest

from tcn.generation import Action, Host, Value
from tcn.types import setof
from generators.gui.generator import (ELEMENT, FIELD, NO_PARENT, hierarchy_type, hierarchy_value,
                                      widgets_from_set)
from generators.gui.render import KINDS, PALETTE

BASE = {"resolution": 32, "widgets": 7, "nesting": 3, "horizon": 4}


def screen(seed=0, **configuration):
    return Host.create("gui", seed=seed, configuration=BASE | configuration)


def test_common_contract_replay_restore_and_visibility(tmp_path):
    h = screen(7)
    h.step([Action("focus", arguments=(("id", Value.of(FIELD, 1)),))])
    assert h.replay().digest == h.digest
    restored = Host.restore(h.snapshot())
    assert restored.digest == h.digest
    h.step(dt=.5)
    restored.step(dt=.5)
    assert h.digest == restored.digest
    path = tmp_path / "gui.json.gz"
    h.save(path)
    assert Host.load(path).digest == h.digest
    view = h.view()
    assert set(view.observations) == {"pixels"}
    assert not hasattr(view, "probes") and not hasattr(view, "latent_states")
    before = copy.deepcopy(h.state)
    view.objective["modified"] = True
    assert h.state == before


def test_seeds_and_splits_are_distinct_and_deterministic():
    assert screen(1).digest == screen(1).digest
    assert screen(1).digest != screen(2).digest
    held = Host.create("gui", seed=1, split="test", configuration=BASE)
    assert held.digest != screen(1).digest


def test_hierarchy_type_is_independent_of_widget_count_and_nesting():
    """One probe type across every screen; cardinality moves, width does not."""
    types, widths, counts = set(), set(), set()
    for widgets in (2, 4, 8, 12):
        for nesting in (1, 2, 4):
            probe = screen(5, widgets=widgets, nesting=nesting).records[-1].probes["hierarchy"]
            types.add(json.dumps(probe.type.to_dict(), sort_keys=True))
            widths.add(len(probe.flat()))
            counts.add(len(probe.decoded))
    assert len(types) == 1 and len(widths) == 1 and len(counts) > 1


def test_hierarchy_round_trip_and_the_id_field_is_load_bearing():
    widgets = [{"id": 0, "parent": NO_PARENT, "kind": 0, "rect": [0, 0, 16, 16]},
               {"id": 1, "parent": 0, "kind": 2, "rect": [1, 1, 6, 6]},
               {"id": 2, "parent": 0, "kind": 2, "rect": [1, 1, 6, 6]}]
    value = hierarchy_value(widgets, 8)
    assert len(value.decoded) == 3                      # a set drops duplicates without the id
    assert widgets_from_set(value) == widgets


def test_capacity_and_field_widths_are_declared_and_enforced():
    with pytest.raises(ValueError):
        screen(0, widgets=9, hierarchy_capacity=8)
    with pytest.raises(ValueError):
        screen(0, resolution=300)
    with pytest.raises(ValueError):
        screen(0, palette=len(PALETTE) + 1)
    assert hierarchy_type(8) == setof(ELEMENT, 8)


def test_owner_probe_is_the_deepest_containing_widget():
    h = screen(11)
    record = h.records[-1]
    rows = {w["id"]: w for w in widgets_from_set(record.probes["hierarchy"])}
    depth = {}
    for i in sorted(rows):
        parent = rows[i]["parent"]
        depth[i] = 0 if parent == NO_PARENT else depth[parent] + 1
    owner = np.array(record.probes["owner"].decoded, dtype=int).reshape(32, 32)
    for y in range(32):
        for x in range(32):
            covering = [i for i, w in rows.items()
                        if w["rect"][0] <= x < w["rect"][0] + w["rect"][2]
                        and w["rect"][1] <= y < w["rect"][1] + w["rect"][3]]
            assert owner[y, x] == max(covering, key=lambda i: (depth[i], i))


def test_colour_determines_owner_when_the_palette_is_injective():
    """The invariant rung one depends on, checked rather than assumed."""
    h = screen(4, widgets=6, palette=32, borders=False, labels=False)
    record = h.records[-1]
    px = record.observations["pixels"].value.decoded
    data = px[3]
    owner = np.array(record.probes["owner"].decoded, dtype=int).flatten()
    seen = {}
    for i, o in enumerate(owner):
        colour = data[3 * i:3 * i + 3]
        assert seen.setdefault(colour, o) == o


def test_borders_and_labels_break_that_invariant_and_are_off_by_default():
    h = screen(4, widgets=6, borders=True, labels=True)
    record = h.records[-1]
    data = record.observations["pixels"].value.decoded[3]
    owner = np.array(record.probes["owner"].decoded, dtype=int).flatten()
    ink = {i for i in range(len(owner)) if data[3 * i:3 * i + 3] == (0, 0, 0)}
    assert ink and len({int(owner[i]) for i in ink}) > 1
    plain = screen(4, widgets=6).records[-1].observations["pixels"].value.decoded[3]
    assert plain != data


def test_actions_change_the_raster_and_reject_illegal_targets():
    h = Host.create("gui", seed=9, configuration=BASE, objective={"target": None})
    widgets = widgets_from_set(h.records[-1].probes["hierarchy"])
    pressable = [w["id"] for w in widgets if KINDS[w["kind"]] in ("button", "field")]
    other = [w["id"] for w in widgets if KINDS[w["kind"]] in ("window", "panel")]
    before = h.records[-1].observations["pixels"].value.raw
    h.step([Action("press", arguments=(("id", Value.of(FIELD, pressable[0])),))])
    assert h.records[-1].observations["pixels"].value.raw != before
    with pytest.raises(ValueError):
        h.step([Action("press", arguments=(("id", Value.of(FIELD, other[0])),))])
    with pytest.raises(ValueError):
        h.step([Action("focus", arguments=(("id", Value.of(FIELD, 200)),))])


def test_objective_rewards_the_named_widget_only():
    h = Host.create("gui", seed=9, configuration=BASE, objective={"target": 3})
    widgets = widgets_from_set(h.records[-1].probes["hierarchy"])
    if KINDS[widgets[3]["kind"]] not in ("button", "field"):
        pytest.skip("widget 3 is not pressable at this seed")
    record = h.step([Action("press", arguments=(("id", Value.of(FIELD, 3)),))])
    assert record.reward_components["goal"].decoded == 1.


def test_widget_and_nesting_dials_move_the_achieved_screen():
    counts = [len(screen(3, widgets=n, nesting=4).records[-1].probes["hierarchy"].decoded)
              for n in (2, 4, 8, 12)]
    assert counts == sorted(counts) and counts[0] < counts[-1]
    depths = []
    for nesting in (1, 2, 4):
        rows = {w["id"]: w for w in
                widgets_from_set(screen(3, widgets=12, nesting=nesting).records[-1].probes["hierarchy"])}
        d = {}
        for i in sorted(rows):
            d[i] = 0 if rows[i]["parent"] == NO_PARENT else d[rows[i]["parent"]] + 1
        depths.append(max(d.values()))
    assert depths == sorted(depths) and depths[0] < depths[-1]


def test_colour_mode_kind_fixes_appearance_across_episodes():
    def kind_colours(seed):
        h = screen(seed, colour_mode="kind")
        record = h.records[-1]
        data = record.observations["pixels"].value.decoded[3]
        owner = np.array(record.probes["owner"].decoded, dtype=int).flatten()
        rows = {w["id"]: w for w in widgets_from_set(record.probes["hierarchy"])}
        return {rows[int(o)]["kind"]: data[3 * i:3 * i + 3] for i, o in enumerate(owner)}
    a, b = kind_colours(1), kind_colours(2)
    shared = set(a) & set(b)
    assert shared and all(a[k] == b[k] for k in shared)


def test_glyph_channel_is_configured_typed_and_faithful():
    from generators.gui.generator import glyphs_from_set
    assert "glyphs" not in screen(2).records[-1].probes
    h = screen(2, resolution=48, labels=True)
    record = h.records[-1]
    probe = record.probes["glyphs"]
    assert probe.type == setof(ELEMENT, 64)
    rows = glyphs_from_set(probe)
    assert rows
    px = record.observations["pixels"].value.decoded
    width, data = px[1], px[3]
    widgets = {w["id"]: w for w in widgets_from_set(record.probes["hierarchy"])}
    for g in rows:
        x, y, w, h = g["rect"]
        wx, wy, ww, wh = widgets[g["id"]]["rect"]
        assert wx <= x and x + w <= wx + ww and wy <= y and y + h <= wy + wh
        ink = any(data[3 * ((y + dy) * width + (x + dx)):3 * ((y + dy) * width + (x + dx)) + 3]
                  == (0, 0, 0) for dy in range(h) for dx in range(w))
        assert ink                                     # the box really holds that glyph
    with pytest.raises(ValueError):
        screen(2, resolution=48, labels=True, glyph_capacity=1).records[-1]
