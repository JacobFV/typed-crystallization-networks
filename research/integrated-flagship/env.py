"""The integrated task: raw screenshot + raw instruction -> click -> kernel reward.

A research-side `tcn.generation.Generator` that composes three shipped
generators **by import, unmodified** (PREREGISTRATION §2):

* `generators/gui`      -- `Implementation.initialize/observe/advance` draw the
                           screen, render the raw pixels and apply `press`;
* `generators/language` -- the engine's English realization renders the
                           instruction from the term `press(<rel>(<colour>, box))`;
* `generators/computer` -- `execute()` is the actuator: the clicked widget id is
                           written into the kernel filesystem and the reward is
                           read back from it, exactly as the panel interface
                           reads its reward off the filesystem.

The agent sees `pixels` and `text` only. `hierarchy`, `owner`, `target`,
`anchor` and the kernel read-back are probes: supervision and diagnostics,
never program inputs.

`Host.create` finds generators only through `generators/*/manifest.json`, which
this track may not touch, so `make_host` builds a `Host` directly and `Ledger`
is an `EpisodeLedger` whose `create` does the same. Both leave `tcn.search`'s
episode accounting unchanged.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcn.generation import (Action, Address, Generator, Host, Value, image_value,  # noqa: E402
                            text_value)
from tcn.search import EpisodeLedger                                                # noqa: E402
from tcn.types import integer                                                       # noqa: E402
from generators.gui.generator import FIELD, Implementation as GUI                   # noqa: E402
from generators.gui.render import KINDS, PRESSABLE, palette_colours, render          # noqa: E402
from generators.language.engine._structure import Ident, Pred                       # noqa: E402
from generators.language.engine.languages import get_language                       # noqa: E402
from generators.computer import generator as computer                               # noqa: E402

RESOLUTION = 16
N_PIXELS = RESOLUTION * RESOLUTION
TEXT_CAPACITY = 64
GUI_CONFIGURATION = {"resolution": RESOLUTION, "widgets": 4, "nesting": 1, "min_size": 3,
                     "palette": 8, "palette_levels": 2, "gap": 0, "colour_mode": "random",
                     "horizon": 1}
# The 8 palette colours at palette_levels 2, named. Only the four anchor colours
# are ever spoken; the names of the other four are for reports.
PALETTE = palette_colours(2)
COLOUR_NAMES = dict(zip(PALETTE, ("black", "blue", "green", "cyan", "red", "purple",
                                  "yellow", "white")))
COLOURS = ("red", "green", "blue", "yellow")
RELATIONS = ("right_of", "left_of", "above")
COMBOS = tuple((c, r) for c in COLOURS for r in RELATIONS)      # index mod 12
MAX_DRAWS = 999
PRESSED = "/home/agent/pressed.txt"
PROBE_REQUEST = {"root": "/home/agent", "depth": 0, "contents": [PRESSED]}
# The typed click argument. Bounds (-1, N) put every pixel index strictly inside
# `tcn/policy.py`'s bounded decode, so the declared encoder can emit it exactly.
CLICK = integer(16, signed=False, bounds=(-1, N_PIXELS))
ID = integer(16, signed=False)


def instruction(colour, relation):
    """The language generator's own realization of the instruction term."""
    return get_language("english").render(Pred("press", Pred(relation, Ident(colour),
                                                              Ident("box"))))


def neighbour(anchor, children, relation, gap):
    """The child adjacent to `anchor` in `relation`, or None."""
    x, y, w, h = anchor["rect"]
    for b in children:
        if b is anchor:
            continue
        bx, by, bw, bh = b["rect"]
        overlap_x = bx < x + w and x < bx + bw
        overlap_y = by < y + h and y < by + bh
        if relation == "right_of" and bx == x + w + gap and overlap_y:
            return b
        if relation == "left_of" and bx + bw + gap == x and overlap_y:
            return b
        if relation == "above" and by + bh + gap == y and overlap_x:
            return b
    return None


def draw_screen(seed, index, split, colour, relation, gap):
    """Deterministic rejection over the GUI generator's own addresses."""
    gui = GUI()
    configuration = dict(GUI_CONFIGURATION, gap=gap)
    for k in range(MAX_DRAWS):
        state = gui.initialize(Address("gui", seed, index * 1000 + k, split), configuration)
        children = [w for w in state["widgets"] if w["parent"] == 0]
        for a in children:
            if COLOUR_NAMES[tuple(a["colour"])] != colour:
                continue
            b = neighbour(a, children, relation, gap)
            if b is not None:
                return state, a["id"], b["id"], k + 1
    raise RuntimeError(f"no screen realizes {(colour, relation)} in {MAX_DRAWS} draws")


def owner_map(gui_state):
    _, owner, _ = render(gui_state["widgets"], gui_state["width"], gui_state["height"])
    return [int(v) for v in owner.flatten()]


class Integrated(Generator):
    version = "integrated-flagship/1"
    action_schema = {"wait": {}, "click": {"pos": CLICK}}

    def initialize(self, address, configuration):
        gap = int(configuration.get("gap", 0))
        combo = configuration.get("combo")
        colour, relation = combo if combo else COMBOS[address.index % len(COMBOS)]
        gui_state, anchor, target, draws = draw_screen(address.seed, address.index,
                                                       address.split, colour, relation, gap)
        text = instruction(colour, relation)
        kernel_seed = address.rng("kernel").randrange(2 ** 31)
        events = [{"kind": "write", "path": PRESSED, "text": "none", "time": 0}]
        transport = configuration.get("session")
        result = computer.execute(kernel_seed, events, 0, PROBE_REQUEST, transport)
        return {"time": 0., "tick": 0, "gui": gui_state, "draws": draws, "text": text,
                "colour": colour, "relation": relation, "anchor": anchor, "target": target,
                "gap": gap, "kernel_seed": kernel_seed, "events": events, "result": result,
                "transport": transport, "clicked": None,
                "horizon": int(configuration.get("horizon", 1))}

    def readback(self, s):
        rows = (s["result"].get("probe") or {}).get("contents", ())
        row = next((r for r in rows if r["path"] == PRESSED), None)
        return None if row is None else row["content"]

    def advance(self, s, actions, dt, rng):
        reward = 0.
        for a in actions:
            if a.verb != "click":
                continue
            pos = int(a.arg("pos"))
            clicked = None
            if 0 <= pos < N_PIXELS:
                clicked = owner_map(s["gui"])[pos]
                widget = s["gui"]["widgets"][clicked]
                if KINDS[widget["kind"]] in PRESSABLE:
                    press = Action("press", arguments=(("id", Value.of(FIELD, clicked)),))
                    s["gui"], _, _ = GUI().advance(s["gui"], [press], dt, rng)
            s["clicked"] = clicked
            s["events"].append({"kind": "write", "path": PRESSED,
                                "text": "none" if clicked is None else str(clicked),
                                "time": s["time"] + dt})
            s["result"] = computer.execute(s["kernel_seed"], s["events"], s["time"] + dt,
                                           PROBE_REQUEST, s["transport"])
            reward = float(self.readback(s) == str(s["target"]))
        s["done"] = s["tick"] + 1 >= s["horizon"]
        clicked = s["clicked"]
        return s, {"clicked": Value.of(ID, 0xFFFF if clicked is None else clicked)}, {"goal": reward}

    def observe(self, s):
        observations, _, probes, _ = GUI().observe(s["gui"])
        observations = {"pixels": observations["pixels"],
                        "text": text_value(s["text"], TEXT_CAPACITY)}
        probes = dict(probes)
        probes["target"] = Value.of(ID, s["target"])
        probes["anchor"] = Value.of(ID, s["anchor"])
        content = self.readback(s)
        probes["kernel_pressed"] = text_value(content or "", 16)
        latents = {"draws": Value.of(ID, s["draws"])}
        return observations, latents, probes, {"agent_0": ("wait", "click")}


def make_host(seed=0, index=0, split="train", configuration=None, objective=None):
    return Host(Integrated(), Address("integrated", seed, index, split),
                configuration, objective)


class Ledger(EpisodeLedger):
    """`tcn.search.EpisodeLedger`, building this generator instead of a manifest one."""

    def create(self, generator, **kwargs):
        self.episodes += 1
        return make_host(**kwargs)


def click_action(pos):
    return Action("click", arguments=(("pos", Value.of(CLICK, int(pos))),))


def episode_record(seed, index, split, gap=0, session=None):
    """The cached facts the evaluator needs, read off one live episode.

    Observations are what the actor sees; `owner`, `target`, `anchor` and
    `widgets` are probes, used only to score and to check equivalence.
    """
    configuration = {"gap": gap}
    if session:
        configuration["session"] = session
    host = make_host(seed, index, split, configuration)
    record = host.records[0]
    view = record.actor_view()
    assert set(view.observations) == {"pixels", "text"}, view.observations
    length, data = view.observations["text"].decoded
    height, width, channels, pixels = view.observations["pixels"].decoded
    s = host.state
    return {"seed": seed, "index": index, "split": split, "gap": gap,
            "colour": s["colour"], "relation": s["relation"], "text": s["text"],
            "text_length": length, "text_bytes": list(data), "pixels": list(pixels),
            "width": width, "height": height, "owner": owner_map(s["gui"]),
            "anchor": s["anchor"], "target": s["target"], "draws": s["draws"],
            "widgets": [{"id": w["id"], "parent": w["parent"], "kind": w["kind"],
                         "rect": list(w["rect"]), "colour": list(w["colour"])}
                        for w in s["gui"]["widgets"]]}
