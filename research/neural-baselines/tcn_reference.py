"""The TCN side of every arm, re-measured here rather than transcribed.

`research/baselines/RESULTS.md` set the precedent and the reason: both sides must
be timed on the same host, on the same day, by the same harness, or the cost
columns are not comparable. So each artifact is rebuilt from the frozen
selections its own track recorded, re-scored on the same held-out set the neural
baseline faces, and re-timed with `common.py:bench`.

Nothing is searched, fitted, tuned or re-selected. Nothing under `tcn/` or
`generators/` is modified, and no file belonging to another track is written.

  python tcn_reference.py language
  python tcn_reference.py computer
  python tcn_reference.py visual      # ~15 s per screen, 12 screens
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import common as harness
import langdata


def load_module(path, name, aliases=()):
    """Execute a track module by file path, without letting its directory onto
    sys.path (three tracks ship a `common.py`; the first one on the path wins)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    for alias in aliases:
        sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
def language(args):
    track = ROOT / "research" / "language-capability"
    LC = load_module(track / "common.py", "lc_common", aliases=("common",))
    SC = load_module(track / "scaffolds.py", "lc_scaffolds", aliases=("scaffolds",))
    SB = load_module(track / "run_stage_b.py", "lc_run_stage_b")
    sys.modules["common"] = harness

    stage_b = json.loads((track / "stage_b.json").read_text())
    stage_a = json.loads((track / "stage_a.json").read_text())
    selections = stage_b["enumeration"]["selections"]
    module, registry, frozen = SB.build_module()
    program, signals = SC.stage_b(module, registry)
    exported = program.harden(selections).pruned()

    sp = langdata.splits()
    report = {"artifact": "language-capability stage-B exported program",
              "rebuilt_from": "research/language-capability/stage_b.json",
              "budget": {"train_episodes": stage_b["train_episodes"],
                         "train_distinct_strings": len({e["string"] for e in sp["train"]}),
                         "stage_a_supervised_positions": stage_a["train_positions"],
                         "programs_enumerated": stage_a["enumeration"]["evaluated"]
                                                + stage_b["enumeration"]["evaluated"],
                         "search_seconds": stage_a["enumeration"]["seconds"]
                                           + stage_b["enumeration"]["seconds"],
                         "optimizer_steps": 0},
              "size": harness.program_size_report(exported, registry),
              "quality": {}}

    def counts_match(s):
        return s.count("(") == s.count(")")

    def balanced(s):
        depth = 0
        for ch in s:
            depth += 1 if ch == "(" else -1
            if depth < 0:
                return False
        return depth == 0

    for name in ("train", "val", "test", "current_stream_test"):
        episodes = sp[name]
        ok = 0
        per_length = {}
        for e in episodes:
            try:
                out, _ = exported.run({"text": e["text"]}, registry=registry)
                hit = bool(out["answer"].decoded) == bool(e["label"])
            except Exception:
                hit = False
            ok += hit
            slot = per_length.setdefault(e["length"], [0, 0])
            slot[0] += hit
            slot[1] += 1
        report["quality"][name] = {
            "accuracy": ok / max(1, len(episodes)),
            "per_length": {k: v[0] / v[1] for k, v in sorted(per_length.items())},
            "lengths": sorted({e["length"] for e in episodes}),
            # Two oracles, so a low score can be attributed. `counting` is the rule
            # the frozen program implements; `dyck` is real balancedness. On the
            # legacy stream FINDINGS §19 measured them to agree on 20,000 of 20,000
            # seeds; if they part company on the current stream, the lesson changed
            # under the artifact rather than the artifact failing at its own task.
            "oracle_counting": sum(counts_match(e["string"]) == e["label"]
                                   for e in episodes) / max(1, len(episodes)),
            "oracle_dyck": sum(balanced(e["string"]) == e["label"]
                               for e in episodes) / max(1, len(episodes)),
            **harness.majority_reference([bool(e["label"]) for e in episodes])}
        harness.report(f"language TCN {name}",
                       f"acc {report['quality'][name]['accuracy']:.4f} "
                       f"counting-oracle {report['quality'][name]['oracle_counting']:.4f} "
                       f"dyck-oracle {report['quality'][name]['oracle_dyck']:.4f} "
                       f"majority {report['quality'][name]['majority_constant']:.4f}")

    cases = [e["text"] for e in sp["test"][:20]]
    counter = {"i": 0}

    def call():
        value = cases[counter["i"] % len(cases)]
        counter["i"] += 1
        return exported.run({"text": value}, registry=registry)

    report["latency"] = {"cold_first_call_ms": harness.cold_call_ms(call),
                         "warm": harness.bench(call)}
    report["peak_rss_mb"] = harness.rss_mb()
    harness.dump("tcn_language", report)
    print(json.dumps({k: v for k, v in report.items() if k != "quality"}, indent=1))


# --------------------------------------------------------------------------
def computer(args):
    track = ROOT / "research" / "computer-capability"
    sys.path.insert(0, str(track))
    import program as P
    import task as T
    import closed_loop as CL
    from tcn.agent import Agent
    from tcn.operators import Registry

    found = json.loads((track / "out" / "search.json").read_text())
    registry = Registry()
    transform = P.transform_program(registry)
    policy = P.policy_program(registry)
    frozen = P.agent_program(registry, transform, policy,
                             found["transform"]["enumeration"]["selections"],
                             found["policy"]["enumeration"]["selections"])
    config = CL.configuration()

    report = {"artifact": "computer-capability frozen agent (successor task)",
              "rebuilt_from": "research/computer-capability/out/search.json",
              "budget": {"train_documents": len(T.TRAIN_DOCUMENTS),
                         "train_decision_points": 3 * len(T.TRAIN_DOCUMENTS),
                         "programs_enumerated": found["transform"]["space_size"]
                                                + found["policy"]["space_size"],
                         "search_seconds": found["transform"]["enumeration"]["seconds"]
                                           + found["policy"]["enumeration"]["seconds"],
                         "optimizer_steps": 0},
              "size": harness.program_size_report(frozen, registry)}

    from dataclasses import replace
    rows, program_ms, kernel_ms = [], [], []
    for i, (name, digit) in enumerate(T.TEST_DOCUMENTS):
        host = T.host(name, digit, 3, objective_path=T.TASK_PATH, seed=0, index=1000 + i,
                      split="test")
        agent = Agent(frozen, registry, config, 0)
        agent.reset()
        verbs = []
        for _ in range(3):
            if host.records[-1].done:
                break
            view = host.view("agent_0")
            started = time.perf_counter_ns()
            action = agent.act(view, deterministic=True)
            program_ms.append((time.perf_counter_ns() - started) / 1e6)
            started = time.perf_counter_ns()
            host.step((replace(action, actor="agent_0"),), config.dt)
            kernel_ms.append((time.perf_counter_ns() - started) / 1e6)
            verbs.append(action.verb)
        rows.append({"document": T.document(name, digit), "verbs": verbs,
                     "return": CL.episode_return(host)})
        harness.report(f"computer TCN {T.document(name, digit)}",
                       f"return {rows[-1]['return']} verbs {verbs}")

    report["quality"] = {"episodes": len(rows), "max_return": 2,
                         "mean_return": sum(r["return"] for r in rows) / len(rows),
                         "solved": sum(r["return"] == 2 for r in rows), "rows": rows}
    report["latency"] = {"agent_act_ms_median": statistics.median(program_ms),
                         "agent_act_ms_first": program_ms[0],
                         "kernel_step_ms_median": statistics.median(kernel_ms),
                         "note": "Agent.act runs the program twice per step; the kernel "
                                 "round trip is reported separately and charged to neither "
                                 "method."}
    report["peak_rss_mb"] = harness.rss_mb()
    harness.dump("tcn_computer", report)
    print(json.dumps({k: v for k, v in report.items() if k != "quality"}, indent=1))


# --------------------------------------------------------------------------
def visual(args):
    track = ROOT / "research" / "visual-ladder"
    VL = load_module(track / "common.py", "vl_common", aliases=("common",))
    R = load_module(track / "rung3_widgets.py", "rung3_widgets")
    load_module(track / "rekey.py", "rekey")
    RR = load_module(track / "rung3_root.py", "vl_rung3_root")
    sys.modules["common"] = harness

    from tcn.types import Value
    registry = VL.Registry()
    configuration = dict(VL.FLAT)
    rung3 = json.loads((track / "out" / "rung3.json").read_text())
    root = json.loads((track / "out" / "rung3_root.json").read_text())

    started = time.perf_counter()
    probe = VL.episode(0, "train", **configuration)
    W, H = probe["width"], probe["height"]
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(registry, W, H)
    same = registry.register_module(p0.harden(rung3["s0"]["chosen"]))
    p1 = RR.corner_scaffold_masked(registry, W, H, same, offsets)
    corner = registry.register_module(p1.harden(root["s1"]["chosen"]))
    p2 = RR.rect_scaffold_clamped(registry, W, H, same, offsets)
    rect = registry.register_module(p2.harden(root["s2"]["chosen"]))
    build_seconds = time.perf_counter() - started

    rows, latency = [], []
    parser = None
    for i in range(args.screens):
        ep = VL.episode(200 + i, "test", **configuration)
        BT = VL.bytes_type(ep["width"], ep["height"])
        positions = RR.all_positions(ep)
        parser = R.assembly(registry, BT, positions, corner, rect)
        value = Value.of(BT, ep["pixels"])
        started = time.perf_counter_ns()
        got = parser.run({"observation": value}, registry=registry)[0]
        predicted = [list(map(int, r)) for r in sorted(got["mapped"].decoded)]
        elapsed = (time.perf_counter_ns() - started) / 1e6
        latency.append(elapsed)
        row = RR.score_with_root(predicted, ep) | {"seed": 200 + i, "latency_ms": elapsed}
        rows.append(row)
        harness.report(f"visual TCN seed {200 + i}",
                       f"rects {row['rects_predicted']}/{row['rects_true'] if 'rects_true' in row else '?'} "
                       f"links {row['parent_links_correct']}/{row['widgets_in_probe']} "
                       f"tree_exact={row['tree_exact']} ({elapsed:.0f} ms)")

    total = {"screens": len(rows),
             "widgets_in_probe": sum(r["widgets_in_probe"] for r in rows),
             "rects_predicted": sum(r["rects_predicted"] for r in rows),
             "screens_rects_exact": sum(bool(r["rects_exact"]) for r in rows),
             "roots_predicted": sum(r["roots_predicted"] for r in rows),
             "parent_links_correct": sum(r["parent_links_correct"] for r in rows),
             "parent_links_wrong": sum(r["parent_links_wrong"] for r in rows),
             "trees_exact": sum(bool(r["tree_exact"]) for r in rows)}
    total["link_accuracy"] = total["parent_links_correct"] / max(1, total["widgets_in_probe"])

    report = {"artifact": "visual-ladder S3' screenshot->hierarchy parse (root recovered)",
              "rebuilt_from": "research/visual-ladder/out/rung3.json + out/rung3_root.json",
              "configuration": configuration,
              "budget": {"train_screens": root["arguments"]["train"],
                         "validation_screens": root["arguments"]["validation"],
                         "corner_positions_per_image": root["arguments"]["corner_per_image"],
                         "programs_enumerated": (root["s1"].get("space_size", 400)
                                                 + root["s2"].get("space_size", 25)
                                                 + rung3["s0"].get("space_size", 256)),
                         "optimizer_steps": 0},
              "size": harness.program_size_report(parser, registry),
              "quality": {"episodes": rows, "totals": total},
              "latency": {"per_screen_ms_median": statistics.median(latency),
                          "per_screen_ms_first": latency[0],
                          "per_screen_ms_min": min(latency)},
              "module_build_seconds": build_seconds,
              "peak_rss_mb": harness.rss_mb()}
    harness.dump("tcn_visual", report)
    print(json.dumps({k: v for k, v in report.items() if k != "quality"}, indent=1))
    print(json.dumps(total, indent=1))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("language", "computer", "visual"))
    ap.add_argument("--screens", type=int, default=12)
    args = ap.parse_args()
    {"language": language, "computer": computer, "visual": visual}[args.arm](args)


if __name__ == "__main__":
    main()
