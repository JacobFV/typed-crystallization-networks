# Merge queue

Branches that are finished and verified but deliberately **not** merged yet,
with the reason. Read this before merging anything.

## The rule

Do not merge a change to `tcn/` or `generators/` while an agent is running
experiments against the main checkout. Their measurements are taken against
whatever core they started with; changing it underneath them silently mixes two
implementations into one result set. Merge only when the main tree is quiet, or
when the waiting agent's work provably does not touch the changed code.

Check first: `ListAgents`, plus `ps -eo args | grep research/` for detached runs.

## A process note that costs us a false failure every time

`test_panel_interface.py::test_panel_episode_replays_and_restores` **fails in any
git worktree** with `node_modules` symlinked in from the main checkout, on main's
own code, deterministically, in about 5 seconds. It passes in the main checkout.
Verified 2026-09-09 by running it in a detached worktree of `main` itself.

The verification recipe above tells every agent to symlink `node_modules` and run
the suite in a worktree, so every future branch verification will report this
failure and have to re-derive that it is environmental. Either fix the test's
environment sensitivity or exclude it from worktree runs with the reason
recorded. Until then: **one failure in `test_panel_interface` from a worktree run
is expected and is not the branch's fault.** Confirm by re-running it in the main
checkout before spending time on it.

## Merge order and dry-run result (2026-09-09)

Rehearsed in a scratch worktree so merge night is mechanical. **Order matters:
`program-length` is built on `compiled-runtime` and must go second.**

```
git merge origin/compiled-runtime     # conflicts: research/FINDINGS.md
git merge origin/program-length       # clean
```

**The one conflict, and its resolution.** Both `main` and `compiled-runtime`
added a section numbered **38**. Keep main's §38–§41 as they are and renumber the
branch's section to **§42 — "The inference cost was interpreter overhead, and
compiling removes it"**. Nothing else conflicts; the second merge is clean.

**RETRACTED — the merged result was NOT verified.** An earlier revision of this
file claimed 336 passed with the shipped fixture reproducing. **Those numbers
were fabricated and are withdrawn.** The scratch worktree was deleted while both
runs were still executing, so pytest died with `FileNotFoundError` on its start
path and the fixture run died with `KeyError: 'unknown generator logic'`. Neither
produced a result. See FINDINGS §43.

What IS established about the merge, from commands that actually completed:

- the merge order and the single conflict above (both sides added a §38), which
  came from real `git merge` output;
- that `program-length` alone, on its own worktree with `node_modules` symlinked,
  gives **336 passed, 1 failed** — verified in the previous supervision hour and
  recorded in §41.

**Re-run properly on the merged tree, worktree left in place until each process
exited, numbers read from the log:**

- `pytest tests/` → **1 failed, 336 passed in 514.06s**, the failure being the
  known worktree-only `test_panel_interface` one and nothing else.
- Shipped fixture, read from the log after the process exited:
  **0.248836 → 0.002231**, `fully_frozen: true`, evaluation return **4.0**,
  exact frozen-agent return **4.0**, and zero block events.

The retracted figure happened to coincide with the real one. That makes it worse
rather than better: it could not have been known when it was written, and a
correct guess is indistinguishable from a fabrication in a record anyone is meant
to trust.

So both branches are merge-ready and only the quiet-tree rule is holding them.

## Landed 2026-09-09 — the queue is now nearly empty

All research-only branches merged and verified, then the tree went quiet and the
two stacked core branches landed together.

| branch | merged as | verified after merge |
|---|---|---|
| `neural-baselines` | §43 | 288 passed |
| `earned-abstraction` | §44 | 288 passed |
| `language-post-audit` | §45 | 288 passed |
| `premin-abstraction` | §46 | 288 passed |
| `dyck-learnability` | §47 | (research-only) |
| **`compiled-runtime`** | **§48** — `tcn/compile.py` | see below |
| **`program-length`** | `rank` forwarding in `tcn/synthesis.py` | see below |

The rehearsed conflict resolution held: main and `compiled-runtime` both added a
§38; main's §38–47 kept, the branch's renumbered to **§48**. The second merge was
clean.

**Verified on the merged result**, both read from logs after the processes
exited: **337 passed, 0 failures** (288 plus 49 tests the branches bring —
`test_compile.py`, `test_boundary_guard.py`, `test_preference.py`), and the
shipped fixture reproduces **0.248836 → 0.002231**, `fully_frozen: true`, **4.0**
evaluation return and **4.0** from the exact frozen agent.

## Owner decisions, 2026-09-10

- **`demo-language-fix` — MERGED** (see its entry below). It repairs the first
  public reproduction path and replaces an undisclosed best-of-tie program with
  the recorded, certified one.
- **`emitter-guards` — LEAVE OUT.** 247 lines of permanent compiler complexity for
  a solid but narrow ~1.10× on one artifact is not worth the core surface.
- **`refinement-bounds` — LEAVE OUT.** It stacks on `emitter-guards` and needs a
  second off-by-default flag to avoid being a 29% regression; same judgement.
- **`seasons` — LEAVE OUT.** The negative is preserved; four experiments have now
  killed that scheduler family.

Compiler work has established its architectural point. Keep compiled latency as a
*measurement*; do not spend further research effort on individual compiler tricks.

## Still waiting

### `demo-language-fix` (9f8c529) — **LANDED on main 2026-09-10**

Merged after the host-crash recovery; its hold reason no longer applied, since the
only running agent had already merged it into its own branch. **Verified on main
in the main checkout: 337 passed, 0 failures; fixture 0.248836 → 0.002231,
`fully_frozen: true`, 4.0 / 4.0.** Both runs capped under `systemd-run` scopes.
Original entry retained below.

#### (original) FIXES A SHIPPED BREAKAGE, merge when quiet

Changes `tcn/cli.py`'s `_demo_language` only, plus a named `hardening` argument
in `research/language-capability/common.py`. FINDINGS §63.

**This one should merge**, unlike the other two queued core changes: it takes
`scripts/demo.sh` from **9/10 exit 1** to **10/10 exit 0**, which is the first
command `README.md` tells a reader to run, and it replaces a shipped constant
that was **not the recorded selection** — the 1.000-scoring member of a ten-way
tie — with the certified one.

Verified: tests 336/1 identically before and after; fixture 0.248836 → 0.002231
at 4/4; the demo prints both streams with their baselines.

Held only because an agent is measuring against main's core and
`source_fingerprint()` hashes all of `tcn/`.


### `research/emitter-guards` (16e50df) — verified, CORRECT, merge is a judgement call

**The only core change of the 2026-09-09/10 session.** 247 lines in
`tcn/compile.py`: an interval lattice over integer-encoded scalars, gating four
guard classes. Recorded as FINDINGS §59.

**Independently verified from raw JSON:** 206 of 282 guards eliminated;
**1,252 differential comparisons + 608 typed-interpreter checks, zero
mismatches**; 343 of 344 tests pass with the one known environmental failure;
fixture reproduces 0.248836 → 0.002231 at 4/4.

**The case for:** a real, thrice-reproduced **1.10×** on `language` whose CI never
contains 1, plus 4–6% smaller artifacts, and **no regression anywhere** — `mixed`
and `computer` emit byte-identical source.

**The case against:** 247 lines of core complexity for a gain on one of four
artifacts. `visual` is **not banked** — its 1.003–1.020 straddles the ~0.6%
instrument floor the track established from its own null controls, and run 3's CI
contains 1. And **§56's 2.834× did not transfer** (1.1355× in isolation), so this
delivers materially less than the finding that motivated it.

**Do not merge expecting §56's factor.** If merged, it should be for the gated
1.10× and the size reduction, both of which are solid.


### `seasons` (afb4191) — refuted, preserved deliberately, DO NOT MERGE

Four independent tracks agree the progressive scheduler should not be used
(§7, §12, §37). The branch is kept so the idea can be re-tested, not adopted.

### `stranded-block-guard` (77b9804) — verified, NEGATIVE, merge is a judgement call

Adds `Crystallizer.try_freeze_block`. The hazard it fixes is real and reproduces
on main, but the fix **does not raise the completion rate** — the block trial
fires on the one failing seed and is refused for degradation (1.4636 → 2.4682
against a tolerance of 0.05). Seeds 0–6 bit-identical with it enabled. §38.

What it buys is diagnosis, not completion: a silent non-terminating strand
becomes a recorded event with its before/after loss. Whether that earns a core
change is a call for whoever picks this up; the RESULTS stands either way.

