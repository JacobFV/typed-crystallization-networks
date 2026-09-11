"""Negative tests: prove the provenance and promise guards FAIL when they should.

A guard that has never been seen to fail is a guard nobody has tested. Both
incidents this machinery exists to prevent were cases where a check passed on
an object it was not actually checking, so each guard here is driven to its
failure state on a temporary copy and asserted to raise.

Nothing under `out/` is written or modified: every case runs on an in-memory
copy or in a temporary directory.

    python selftest_guards.py
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import stamp

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
cases = []


def case(name, fn):
    """`fn` must raise stamp.StampError. Passing silently is the failure."""
    try:
        fn()
    except stamp.StampError as exc:
        cases.append((True, name, str(exc).split(":")[0]))
        return
    except Exception as exc:                                   # noqa: BLE001
        cases.append((False, name, f"raised the wrong type: {exc!r}"))
        return
    cases.append((False, name, "did NOT raise — the guard does not bite"))


def main():
    live = json.loads((OUT / "phase0.json").read_text())

    case("an artifact with no provenance block is refused",
         lambda: stamp.present({k: v for k, v in live.items() if k != "provenance"}))

    case("an artifact that is not a JSON object is refused",
         lambda: stamp.present([1, 2, 3]))

    def unknown_format():
        bad = copy.deepcopy(live)
        bad["provenance"]["format"] = "some-other-track/provenance-9"
        stamp.present(bad)
    case("an unknown provenance format is refused", unknown_format)

    def missing_field():
        bad = copy.deepcopy(live)
        del bad["provenance"]["sources"]
        stamp.present(bad)
    case("a provenance block missing a required field is refused", missing_field)

    def empty_inputs():
        bad = copy.deepcopy(live)
        bad["provenance"]["inputs"] = {}
        stamp.present(bad)
    case("a provenance block naming no inputs is refused", empty_inputs)

    def edited_parameters():
        bad = copy.deepcopy(live)
        bad["provenance"]["parameters"]["s0_offsets"] = [6, 51, 3, 48, 96]
        stamp.present(bad)
    case("a provenance block whose parameters were edited after writing is refused",
         edited_parameters)

    def grammar_absent_without_reason():
        bad = copy.deepcopy(live)
        bad["provenance"]["grammar_digest"]["absent_because"] = None
        stamp.present(bad)
    case("an absent mutation-grammar digest with no reason given is refused",
         grammar_absent_without_reason)

    def grammar_field_omitted():
        bad = copy.deepcopy(live)
        del bad["provenance"]["grammar_digest"]
        stamp.present(bad)
    case("an OMITTED mutation-grammar digest is refused (omission is not absence)",
         grammar_field_omitted)

    def no_revision():
        bad = copy.deepcopy(live)
        bad["provenance"]["revision"] = ""
        stamp.present(bad)
    case("a missing pre-registration revision is refused", no_revision)

    def dirty_producer():
        bad = copy.deepcopy(live)
        bad["provenance"]["commit"]["dirty"] = True
        stamp.present(bad)
    case("an artifact produced from a DIRTY tree is refused", dirty_producer)

    def dirty_sources_despite_clean_outputs():
        bad = copy.deepcopy(live)
        bad["provenance"]["commit"]["dirty"] = True
        bad["provenance"]["commit"]["dirty_tracked_paths"] = [" M research/schema-induction/grammar.py"]
        bad["provenance"]["commit"]["dirty_outputs"] = []
        stamp.present(bad)
    case("a modified SOURCE is refused even when the outputs are clean",
         dirty_sources_despite_clean_outputs)

    def unknown_dirtiness():
        bad = copy.deepcopy(live)
        bad["provenance"]["commit"]["dirty"] = None
        stamp.present(bad)
    case("an artifact whose tree state is unknown is refused (not-false, not just true)",
         unknown_dirtiness)

    def untracked_source():
        bad = copy.deepcopy(live)
        key = next(iter(bad["provenance"]["sources"]))
        bad["provenance"]["sources"][key]["tracked"] = False
        stamp.present(bad)
    case("an artifact produced by an UNTRACKED source is refused", untracked_source)

    def no_commit():
        bad = copy.deepcopy(live)
        bad["provenance"]["commit"] = {"head": None}
        stamp.present(bad)
    case("a missing producing commit is refused", no_commit)

    def base_omitted():
        bad = copy.deepcopy(live)
        del bad["provenance"]["base_digest"]
        stamp.present(bad)
    case("an omitted base scaffold digest is refused", base_omitted)

    def base_edited():
        bad = copy.deepcopy(live)
        bad["provenance"]["base"]["step_offsets"] = [6, 51, 3, 48, 96]
        stamp.present(bad)
    case("a base scaffold edited after the artifact was written is refused", base_edited)

    def splits_edited():
        bad = copy.deepcopy(live)
        bad["provenance"]["splits"]["gap1.admission"] = "0" * 64
        stamp.present(bad)
    case("a split digest edited after the artifact was written is refused", splits_edited)

    case("an input digest that disagrees with disk is refused",
         lambda: stamp.require(live, inputs={"episodes_gap0.file": "0" * 64}))

    case("an input the block does not name is refused",
         lambda: stamp.require(live, inputs={"episodes_gap9.file": "0" * 64}))

    case("a source that changed since the artifact was written is refused",
         lambda: stamp.require(live, sources=[HERE / "verify.py"]))

    case("the wrong `kind` is refused", lambda: stamp.require(live, kind="final"))

    case("a comparison with nothing to compare is refused (an assertion that "
         "checks nothing must fail, not pass)", lambda: stamp.require(live))

    case("stamp.write refuses to overwrite an existing provenance block",
         lambda: stamp.write(OUT / "unreachable.json", live, live["provenance"]))

    case("a missing input cannot be fingerprinted silently",
         lambda: stamp.digest_file(HERE / "no-such-file.json"))

    width = max(len(c[1]) for c in cases)
    for ok, name, detail in cases:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}s}  [{detail}]")
    failed = [c for c in cases if not c[0]]
    print(f"\n{len(cases) - len(failed)} PASS, {len(failed)} FAIL")
    assert not (OUT / "unreachable.json").exists(), "a refused write still created a file"
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
