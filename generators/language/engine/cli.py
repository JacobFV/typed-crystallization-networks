"""``langcurriculum`` on the command line: list, show, export, verify, evaluate."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Callable, Sequence

from . import __version__
from .curricula import curriculum_ids
from .curricula import get as get_curriculum
from .dataset import compositional_splits, export
from .evaluate import evaluate, random_agent
from .languages import DEFAULT_LANGUAGE, languages
from .presentation import ANSWER_FORMATS, Presentation
from .registry import all_lessons, by_tag, get, resolve
from .surfaces import (NATIVE_SURFACES, REPRODUCIBILITY, RENDERER_VERSIONS,
                       render_native, renders_natively, surface_names,
                       transcode_example)
from .verify import verify_all, verify_surface

__all__ = ["main"]


def _cmd_ls(args: argparse.Namespace) -> int:
    if args.curriculum:
        c = get_curriculum(args.curriculum)
        order = c.linearize(args.order)
        depth = c.layers()
        for i, node in enumerate(order, 1):
            l = get(node.lesson)
            flag = "" if l.status == "implemented" else f"  [{l.status}]"
            print(f"{i:>4}  d{depth[node.key]:<3} {node.lesson:<36} {l.teaches}{flag}")
        print(f"\n{len(order)} nodes in {c.id} ({args.order} order)", file=sys.stderr)
        return 0
    rows = resolve(args.lessons) if args.lessons else list(all_lessons().values())
    for l in rows:
        flag = "" if l.status == "implemented" else f"  [{l.status}]"
        knob = "~" if l.supports_difficulty() else " "
        print(f"{knob} {l.id:<36} {','.join(l.tags):<44} {l.teaches}{flag}")
    print(f"\n{len(rows)} lessons  (~ = has a difficulty knob)", file=sys.stderr)
    return 0


def _cmd_curricula(args: argparse.Namespace) -> int:
    if not args.curriculum:
        for name in curriculum_ids():
            c = get_curriculum(name)
            print(f"{c.id:<16} {len(c.nodes):>4} nodes  {len(c.edges):>5} edges   {c.title}")
        print(f"\nalso tag:<name> and capability:<name>; "
              f"tags are {', '.join(sorted(by_tag())[:8])}...", file=sys.stderr)
        return 0
    c = get_curriculum(args.curriculum)
    if args.json:
        print(json.dumps(c.to_dict(), indent=2))
        return 0
    print(f"{c.id}: {c.title}\n{c.description}\n")
    print(f"{len(c.nodes)} nodes, {len(c.edges)} edges, "
          f"{len(c.roots())} roots, {max(c.layers().values(), default=0) + 1} layers")
    orders = list(c.linearizations(args.orders))
    print(f"{len(orders)} distinct flattenings shown:")
    for i, order in enumerate(orders, 1):
        head = " -> ".join(n.lesson for n in order[:6])
        print(f"  {i}. {head} ...")
    return 0


def _cmd_languages(args: argparse.Namespace) -> int:
    for lang in languages():
        default = "  (default)" if lang["code"] == DEFAULT_LANGUAGE else ""
        print(f"{lang['code']:<18} {lang['kind']:<9} {lang['name']}{default}")
        print(f"{'':<18} {lang['description']}")
    return 0


def _cmd_surfaces(args: argparse.Namespace) -> int:
    print(f"{'surface':<10} {'renderer':<12} {'kind':<10} reproducibility")
    for name in surface_names():
        kind = "native" if name in NATIVE_SURFACES else "transcode"
        print(f"{name:<10} {RENDERER_VERSIONS[name]:<12} {kind:<10} "
              f"{REPRODUCIBILITY[name]}")
    print("\na transcode carries the same string in another medium, so it inherits "
          "the lesson's floor;\na native surface draws the episode's structure and "
          "has to be verified in its own right.", file=sys.stderr)
    print(f"\nanswer formats: {', '.join(sorted(ANSWER_FORMATS))}", file=sys.stderr)
    return 0


def _presentation(args: argparse.Namespace) -> Presentation:
    p = Presentation.parse(getattr(args, "presentation", None))
    if getattr(args, "language", None):
        p = p.with_(language=args.language)
    if getattr(args, "format", None):
        p = p.with_(answer_format=args.format)
    if getattr(args, "surface", None):
        p = p.with_(surface=args.surface)
    return p


def _cmd_show(args: argparse.Namespace) -> int:
    lesson = get(args.lesson)
    info = lesson.info()
    pres = _presentation(args)
    if args.json:
        ex = None if lesson.status != "implemented" else lesson.example(
            args.seed, presentation=pres, difficulty=args.difficulty).to_dict()
        print(json.dumps({"info": info, "example": ex}, indent=2))
        return 0
    print(f"{info['id']}  (level {info['level']}, tags: {', '.join(info['tags']) or '-'})")
    print(f"teaches: {info['teaches']}")
    if info["capabilities"]:
        print(f"capabilities: {', '.join(info['capabilities'])}")
    if info["axes"]:
        print("axes: " + ", ".join(f"{k}={v}" for k, v in sorted(info["axes"].items())))
    print(f"difficulty knob: {'yes' if info['supports_difficulty'] else 'no'}")
    if info["description"]:
        print("\n" + info["description"])
    if lesson.status != "implemented":
        print(f"\nstatus: {lesson.status}\n{lesson.note}")
        return 0
    for i in range(args.n):
        ex = lesson.example(args.seed + i, presentation=pres, difficulty=args.difficulty)
        print(f"\n--- seed {ex.seed} ({ex.presentation}) " + "-" * 30)
        if pres.surface == "text":
            print(ex.prompt)
        else:
            content = (render_native(lesson, ex.seed, language=pres.language,
                                     difficulty=args.difficulty, surface=pres.surface)
                       if pres.surface in NATIVE_SURFACES
                       else transcode_example(ex, pres.surface))
            print(content.text if pres.surface in ("spoken", "audio")
                  else repr(content))
            if args.write:
                for a in content.assets:
                    path = f"{args.write}/{lesson.id}.{ex.seed:03d}.{a.name('x')}"
                    import pathlib as _p
                    _p.Path(path).parent.mkdir(parents=True, exist_ok=True)
                    _p.Path(path).write_bytes(a.data)
                    print(f"wrote {path} ({a.size} bytes)")
            if not content.fidelity.lossless:
                print(f"! fidelity: {content.fidelity.to_dict()}")
        print(f"target: {ex.target}   (answer: {ex.answer})")
    return 0


def _cmd_structure(args: argparse.Namespace) -> int:
    lesson = get(args.lesson)
    for i in range(args.n):
        print(json.dumps(lesson.structured(args.seed + i, difficulty=args.difficulty),
                         indent=2 if args.n == 1 else None, sort_keys=True))
    return 0


def _cmd_splits(args: argparse.Namespace) -> int:
    rows = compositional_splits(args.curriculum)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for r in rows[:args.limit]:
        print(f"{r['node']:<36} train {len(r['train']):>4}  eval {r['eval'][0]}")
    print(f"\n{len(rows)} compositional splits in {args.curriculum}", file=sys.stderr)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    total = export(args.out, args.lessons, n=args.n, seed0=args.seed,
                   presentation=_presentation(args), difficulty=args.difficulty,
                   include_hidden=not args.no_hidden, per_lesson=args.per_lesson,
                   verify_first=args.verify)
    print(f"wrote {total} records to {args.out}", file=sys.stderr)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    if args.surface:
        rows = [verify_surface(l, args.surface, episodes=args.episodes,
                               language=args.language or DEFAULT_LANGUAGE)
                for l in resolve(args.lessons)]
        failed = [r for r in rows if r.get("ok") is False]
        # `ok is None` means the surface does not apply to this lesson at all --
        # a native renderer with nothing to draw. Counting that as a failure
        # would make a lesson look broken for the crime of being text.
        skipped = [r for r in rows if r.get("ok") is None]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                if r.get("ok") is None:
                    print(f"{r['lesson']:<36} n/a   {r.get('note', '')}")
                    continue
                mark = "ok  " if r["ok"] else "LOSSY"
                print(f"{r['lesson']:<36} {mark} lossy={r.get('lossy_episodes')} "
                      f"dropped={''.join(r.get('dropped') or []) or '-'}")
        checked = len(rows) - len(skipped)
        print(f"\n{checked - len(failed)}/{checked} keep the episode answerable "
              f"in {args.surface}" + (f"; {len(skipped)} n/a" if skipped else ""),
              file=sys.stderr)
        return 1 if failed else 0

    rows = verify_all(args.lessons, episodes=args.episodes, difficulty=args.difficulty)
    failed = [r for r in rows if r.get("ok") is False]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            if r.get("ok") is None:
                print(f"{r['lesson']:<36} spec-only")
                continue
            mark = "ok " if r["ok"] else "FAIL"
            print(f"{r['lesson']:<36} {mark}  uniform={r.get('uniform')} "
                  f"constant={r.get('constant')} limit={r.get('limit')}")
    print(f"\n{len(rows) - len(failed)}/{len(rows)} passed", file=sys.stderr)
    return 1 if failed else 0


def _load_agent(spec: str) -> Callable[[str], str]:
    """``module:function`` -> the callable, imported at call time."""
    if spec == "random":
        return random_agent()
    if ":" not in spec:
        raise SystemExit(f"agent must be 'module:function' or 'random', got {spec!r}")
    mod, _, fn = spec.partition(":")
    obj: Any = importlib.import_module(mod)
    for part in fn.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise SystemExit(f"{spec} is not callable")
    return obj


def _cmd_eval(args: argparse.Namespace) -> int:
    agent = _load_agent(args.agent)
    report = evaluate(agent, args.lessons, n=args.n, seed0=args.seed,
                      presentation=_presentation(args), difficulty=args.difficulty,
                      strict=args.strict)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.table())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="langcurriculum",
                                description="A procedurally generated curriculum "
                                            "for developing and evaluating symbolic AI.")
    p.add_argument("--version", action="version", version=f"langcurriculum {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _presentation_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-L", "--language", default=None,
                        help=f"language to read the episode in (default {DEFAULT_LANGUAGE})")
        sp.add_argument("-F", "--format", default=None,
                        help=f"answer format: {', '.join(sorted(ANSWER_FORMATS))}")
        sp.add_argument("-S", "--surface", default=None,
                        help=f"modality: {', '.join(surface_names())}")
        sp.add_argument("-P", "--presentation", default=None,
                        help="all three at once, as language/format/surface")
        sp.add_argument("-d", "--difficulty", type=float, default=None,
                        help="difficulty in [0, 1], for lessons that have a knob")

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-l", "--lessons", default=None,
                        help="lesson ids, a curriculum name, tag:x, capability:x, or all")
        _presentation_args(sp)
        sp.add_argument("--seed", type=int, default=0, help="first seed")

    sp = sub.add_parser("ls", help="list lessons")
    sp.add_argument("-l", "--lessons", default=None)
    sp.add_argument("-c", "--curriculum", default=None,
                    help="list in a curriculum's order instead")
    sp.add_argument("--order", default="default",
                    choices=["default", "level", "breadth", "depth"])
    sp.set_defaults(fn=_cmd_ls)

    sp = sub.add_parser("curricula", help="list curricula, or describe one")
    sp.add_argument("curriculum", nargs="?", default=None)
    sp.add_argument("--orders", type=int, default=4,
                    help="how many distinct flattenings to show")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=_cmd_curricula)

    sp = sub.add_parser("languages", help="list the languages an episode can be read in")
    sp.set_defaults(fn=_cmd_languages)

    sp = sub.add_parser("surfaces", help="list the modalities and what they guarantee")
    sp.set_defaults(fn=_cmd_surfaces)

    sp = sub.add_parser("show", help="show one lesson and sample episodes")
    sp.add_argument("lesson")
    sp.add_argument("-n", type=int, default=1, help="episodes to print")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--write", metavar="DIR",
                    help="save the rendered assets (images, audio) into a directory")
    _presentation_args(sp)
    sp.set_defaults(fn=_cmd_show)

    sp = sub.add_parser("structure", help="dump an episode's structure, not its rendering")
    sp.add_argument("lesson")
    sp.add_argument("-n", type=int, default=1)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("-d", "--difficulty", type=float, default=None)
    sp.set_defaults(fn=_cmd_structure)

    sp = sub.add_parser("splits", help="compositional train/eval splits from a curriculum")
    sp.add_argument("-c", "--curriculum", default="progressive")
    sp.add_argument("--limit", type=int, default=30)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=_cmd_splits)

    sp = sub.add_parser("export", help="write a JSONL dataset")
    sp.add_argument("out")
    sp.add_argument("-n", type=int, default=100, help="episodes per lesson")
    sp.add_argument("--per-lesson", action="store_true",
                    help="write one file per lesson into a directory")
    sp.add_argument("--no-hidden", action="store_true",
                    help="omit the hidden ground truth from metadata")
    sp.add_argument("--verify", action="store_true",
                    help="refuse to write if any selected lesson fails verification")
    _common(sp)
    sp.set_defaults(fn=_cmd_export)

    sp = sub.add_parser("verify", help="check floors, determinism, generation and fidelity")
    sp.add_argument("-l", "--lessons", default=None)
    sp.add_argument("--episodes", type=int, default=200)
    sp.add_argument("-S", "--surface", default=None,
                    help="check a transcode keeps episodes answerable, instead")
    sp.add_argument("-L", "--language", default=None)
    sp.add_argument("-d", "--difficulty", type=float, default=None)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=_cmd_verify)

    sp = sub.add_parser("eval", help="evaluate a text agent")
    sp.add_argument("agent", help="'module:function', or 'random' for the floor")
    sp.add_argument("-n", type=int, default=20, help="episodes per lesson")
    sp.add_argument("--strict", action="store_true", help="require exact-match replies")
    sp.add_argument("--json", action="store_true")
    _common(sp)
    sp.set_defaults(fn=_cmd_eval)

    args = p.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
