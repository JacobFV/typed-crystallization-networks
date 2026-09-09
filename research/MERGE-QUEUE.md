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

## Waiting

### `compiled-runtime` (68db06a) — verified, blocked on quiet tree, and POSITIVE

Adds `tcn/compile.py` (frozen `Program` + `Registry` → standalone stdlib Python),
`tests/test_compile.py`, and `research/compiled-runtime/`.
`research/compiled-runtime/RESULTS.md`.

**Independently verified from the supervising session**, not taken on the agent's
word:

- **Pure addition confirmed.** `git diff --name-only main...HEAD -- tcn/` lists
  only `tcn/compile.py`. `types.py`, `graph.py` and `operators.py` are untouched,
  so the falsification stays clean.
- **Equivalence re-run here.** `check.py` hard-asserts arm C against the
  interpreter's decoded output before any timing; re-running `mixed` gives
  `A==C exact` on all four cases.
- **The generated code is what was asked for.** `v4 = _k0[2 * v1 + v2]`,
  `v6 = _c1(v5 + v3)` — native values, straight-line SSA, **zero** `Value(`
  constructions in the generated visual source, stdlib only, and every line
  carries its typed node and operator in a comment and in `PROVENANCE`.
- **Test suite: 313 passed, 1 failed — and the failure is not this branch.**
  `test_panel_interface.py::test_panel_episode_replays_and_restores` fails
  identically on **main's own code** in a detached worktree with symlinked
  `node_modules`, and still fails with `tcn/compile.py` moved aside. See the
  process note below; the agent's "314 passed" was optimistic by one.

Blocked because `source_fingerprint()` hashes all of `tcn/`, so even a new file
invalidates every recorded episode while the neural-baselines agent is measuring.
Merge when that agent is done.


### `stranded-block-guard` (77b9804) — verified, blocked on quiet tree, and NEGATIVE

Isolates the guard hazard FINDINGS §37 found inside the seasons branch, so that
none of seasons' +345 lines of scheduler machinery has to be merged to keep it.
Adds `Crystallizer.try_freeze_block` (~45 lines) plus `close_block`, and 7 tests
in `tests/test_stranded_block.py`. `research/stranded-block/RESULTS.md`.

**Read the verdict before merging: the fix does not raise the completion rate.**
The hazard is real and reproduces on `main` (seed 7 of 8 ends 9/13 frozen behind
29 refusals; rounds=48 buys 77 refusals and no progress). The block trial fires
on exactly that seed and is *refused for degradation*, 1.4636 → 2.4682 against a
tolerance of 0.05. Seeds 0–6 are bit-identical on the committed selections map
with it enabled and no trial fires. What it buys is diagnosis, not completion.

Verified in-branch: 295 tests pass; `python -m tcn train --episodes 160` gives
0.24884 → 0.00223, fully frozen, 4.0 evaluation and 4.0 exact frozen return,
with zero block events — the fix never fires on the shipped path.

Blocked because it changes `tcn/crystallize.py` while the compiled-runtime and
neural-baselines agents are measuring against main's core. Merge when the tree
is quiet, or decide not to: a change measured to alter no outcome is a legitimate
thing to leave on a branch, and the RESULTS stands on its own either way.


### `perturbation-selection` (90dd08a) — SUPERSEDED by `loss-gated-eligibility`

`loss-gated-eligibility` (ec953bc) is built on this commit and contains it
(`git merge-base --is-ancestor` confirms). Merge that branch instead; merging
this one separately is redundant.

Original entry retained below for its verification record.

### `perturbation-selection` (commit 90dd08a) — verified, blocked on quiet tree

Replaces entropy+stability selection with the DARTS-PT removal rule, and adds
`Objective(total, task)` so the connectivity guard probes the unregularized task
loss. 96 tests pass in its worktree.

**Independently verified from the supervising session**, not taken on the agent's
word: `96 passed in 360.92s` run against that worktree, with the gitignored
`node_modules` symlinked in for the four computer tests and the symlink removed
afterwards, leaving the worktree clean at 90dd08a.

Blocked because it changes `tcn/crystallize.py`, `tcn/synthesis.py`, `tcn/cli.py`
and `JointTrainer.episode`, while the perception-ladder and
recursive-abstraction-retest agents are measuring against main's current core.

Merge when both have finished and their results are landed. After merging,
re-run `tcn train --episodes 160` and confirm it still reports 0.2488 -> 0.0022,
4/4 deterministic and 4/4 frozen; the agent reports three disconnection
deferrals instead of four, which is expected from the corrected guard and is
already reflected in its `docs/VALIDATION.md` edit.

Note for a fresh worktree: the four `generators/computer` tests need
`generators/computer/engine/node_modules`, which is gitignored. Symlink it from
the main checkout to run the full suite.

### `abstraction-preference` (commit on branch `abstraction-preference`) — blocked on quiet tree

Makes the objective able to prefer the cheaper program. Adds
`SoftProgram.description_cost()` (ARCHITECTURE section 8's
`L_program_description`: expected description length in bits of the pruned
hardened program, exact at any one-hot selection), exposes it as
`synthesis.fit(mdl_weight=...)` and `TrainConfig.description_weight` (both
default 0), deduplicates argument rows in `exact_tensor`, adds
`Program.pruned()` applied by `Registry.register_module` and
`runtime.save_program`, and gives `enumerate_fit` a `rank` of
`order`/`description`/`cost` plus a `conforming` count.

108 tests pass (93 shipped plus 15 new in `tests/test_preference.py`), and
`tcn train --episodes 160` still reports `0.248836 -> 0.002231`, fully frozen,
4.0/4.0 return.

**Two things that change behaviour even with every new weight at zero**, and both
need re-verifying against whatever else is in flight before merging:

- **Pruning changes content addresses.** A registered module and a saved artifact
  are now the pruned program, so a module learned in a scaffold with a dead gate
  gets a different `module:<digest>` than it did before, and `save_program`
  writes a different digest than `program.digest` of the unpruned export. Any
  recorded digest that straddles the merge is not comparable.
- **`exact_tensor` output is bit-identical** (checked on the Boolean and float
  paths) but roughly 4.5x faster on module-heavy scaffolds and 6-8% slower where
  argument rows never repeat, so wall-clock figures straddling the merge are not
  comparable either.

Blocked because it changes `tcn/learning.py`, `tcn/graph.py`, `tcn/operators.py`,
`tcn/runtime.py`, `tcn/search.py`, `tcn/synthesis.py` and `tcn/training.py`,
which overlaps `perturbation-selection` (`tcn/synthesis.py`) and
`positional-reuse` (`tcn/graph.py`). Merge order has **not** been dry-run against
those two; do that first. Results are in
`research/abstraction-preference/RESULTS.md`.

### `positional-reuse` — verified, blocked on quiet tree

Refutes the positional-reuse blocker: one crystallized module applies at every
position of an arbitrarily wide tuple with **three caller nodes**, independent of
width and position count, using only operators already in the registry
(`insert` to bridge into set-land, `pair` for the cartesian product with a
constant set of positions, `map` with the module parameter). No operator, type,
relaxation or loss term is added, and no ARCHITECTURE section 2 amendment is
proposed.

Also fixes a genuine search bug: `legal_candidates` resolved every candidate
with empty parameters, so no parameterized operator could ever be proposed.

**Independently verified from the supervising session, against main's operators
rather than the branch:**

- `legal_candidates` on main proposes **0** candidates for `project`, `map`,
  `filter` and `join`, and 1 for `and`. `project` with empty parameters raises
  `TypeError: operator signature mismatch`; with `index=0` it resolves. The
  entire structural and recursive-abstraction family was outside the search
  space, so positional reuse could only ever be hand-supplied, never discovered.
- The three-node composition runs on **unmodified main**: a registered
  `(index, wide) -> bool` module applied across an 8-wide observation by exactly
  3 caller nodes. Confirmed the claim needs no core change to be *expressible* —
  the branch's changes are what let search *find* it.
- Confirmed the position tag is load-bearing: an untagged `set[BOOL]` output
  collapses to `[False, True]`, losing per-position information, exactly as the
  duplicate-free set semantics require.

Blocked because it changes `tcn/graph.py` and `tcn/scaffold.py` while the
perception-ladder and recursive-abstraction-retest agents measure against main.

Its own reported caveats to preserve when landing: the geometry demonstration's
supervision does not identify the program (1,584 of 32,000 candidates survive
held-out, the renderer's own among them), the whole pattern sits behind a
declared gradient boundary since `map`/`pair`/`insert` have no relaxation, and
73% of apply time is `Value.of` re-encoding because `pair` replicates the
observation per position.

## Merge order, checked in advance

Dry-run with `git merge-tree --write-tree`, no working tree touched:

- `positional-reuse` against main: **clean**
- `perturbation-selection` against main: **clean**
- the two against each other: **clean**

They overlap on `generators/logic/generator.py` and `tests/test_generators.py`
only, and git resolves both without conflict. Either order works.

Merge `perturbation-selection` first: it is the smaller behavioural change to
the shipped path and its re-verification is the sharper tripwire (`tcn train
--episodes 160` must still give 0.2488 -> 0.0022, 4/4 deterministic, 4/4
frozen, with three disconnection deferrals rather than four). If that holds,
merge `positional-reuse` and re-run the suite again.

After both land, two things follow immediately and should not be forgotten:

- The `legal_candidates` parameter fix changes the candidate space, so any
  scaffold that enumerated candidates before the merge may now see more of
  them. Re-run the shipped fixtures and confirm the numbers before trusting
  any comparison that straddles the merge.
- `research/discrete-perception` was measured against pre-merge main and was
  explicitly told to supply `project`/`map`/`filter` candidates by hand. Its
  numbers remain valid for what they measured; do not silently restate them as
  post-merge results.

### `loss-gated-eligibility` (ec953bc) — verified, blocked on quiet tree

Contains `perturbation-selection` and supersedes it. Adds `eligibility` and
`anneal` rules to `Crystallizer`, both **defaulting to the pre-existing
behaviour** (`eligibility="immediate"`, `anneal="round"`), verified by reading
the constructor on the branch. Note the inherited `selection="perturbation"`
default from 90dd08a IS a behavioural change; that belongs to the parent commit.

Verified from the supervising session: contains 90dd08a,
`git merge-tree` shows it conflict-free against `positional-reuse`, and
**`104 passed in 64.89s`** run against that worktree rather than taken on the
agent's word (gitignored `node_modules` symlinked in for the four computer
tests, symlink removed afterwards, worktree left clean at ec953bc).

Its verdict is the third independent confirmation that the scheduler does not
beat plain argmax, and the agent killed its own apparent win: not-annealing
conformed 48/48 against step-matched argmax's 42/48 (p = 0.0265), but that arm
spends about twice argmax's forward passes, and a forward-matched argmax
conforms 32/32. The gain did not survive the second compute axis.

The mechanism it identified is the more general result and should survive into
any future scheduler work: under the gate, perturbation and argmax both name
the reference candidate 11/16 and agree on 9 of 16, whereas at the incumbent
schedule it is 5/16 against 0/16. **The interval in which the perturbation
measurement beats argmax is exactly the interval in which committing is a
mistake.** For DARTS-PT-style selection to pay here, the freeze would have to
be reversible.

Also worth keeping: gating eligibility while leaving the temperature on the
round clock is actively broken, because waiting rounds are counted by the
anneal, so the temperature floors before the objective settles. The two clocks
must come off together.

### `abstraction-preference` — verified, CONFLICTS, merge by hand

Adds `SoftProgram.description_cost()` (expected description bits of the pruned
hardened program under the choice distribution, equal to
`export().pruned().description_bits()` at any one-hot), wires it into
`synthesis.fit(mdl_weight=)` and `TrainConfig.description_weight`, both off by
default; deduplicates `exact_tensor` rows; prunes dead nodes at
`register_module` and `save_program`; and adds
`enumerate_fit(rank='order'|'description'|'cost')` with a `conforming` count.

**`git merge-tree` shows two conflicts against current main**, unlike the other
branches: `tcn/training.py` and `research/FINDINGS.md`. It was cut before the
positional-reuse and loss-gated merges landed, so it does not contain them.
Merge by hand, keeping both sides: main has the merged `Objective(total, task)`
signature and the gate rules in `training.py`, the branch adds
`description_weight`.

Verified from the supervising session: `complexity()` charges a module call its
whole body cost and counts dead nodes, so the branch's correction to the
re-test's "20.43 for both" is right and is now recorded in FINDINGS section 12.

Report its trade honestly when landing: the description term takes the tight
scaffold from 19/24 to 2/24 conformance (p = 1.1e-6), because the pressure is
"make fewer nodes live", which is aligned on an over-provisioned scaffold and
opposed on an exactly-sized one. Defaults are off precisely because of that.

### `worktree-agent-afc0484c7c0dc4724` (6178867) — verified, held

Automatic search-mode selection. Adds `tcn/select.py` (`liveness`,
`enumeration_cost`, `hybrid_cost`, `select_backend`, `hybrid_fit`), a
`mode="auto"` on `synthesis.fit` and `tcn synthesize --mode`, both defaulting to
the shipped path. Reported: agrees with what the tracks measured on 11 of 11
decisions and 10 of 11 end to end, 197 tests passing, `examples/mixed.py`
unchanged at relaxed 1.0066e-06 / exact 0.0.

**It also implements Defect 2 independently**: `SoftProgram.surrogate_scale`
multiplies the temperature only on the relaxation path, so a surrogate widens
without flattening the node's choice distribution, with the carrier scaling as
an opt-in `carrier_temperature` in `relaxed`. The core-gradient-fixes track has
been told to adopt this rather than write a second mechanism.

Held because `tcn/` changes cannot land while the lesson-audit track is running
experiments against the main checkout, and because the core-gradient-fixes
branch will touch the same files. **Merge that branch's Defect 1 fix together
with this one**, not separately.

Verified from the supervising session: its claim that the shipped scaffolds are
partly unreachable by relaxation is correct and now measured directly —
`truth_0` and `truth_15` are constant functions whose input gradient is
identically zero, giving a reachable fraction of exactly **0.875** of the
16-table family. So one eighth of every truth-table node in every shipped
scaffold has always been invisible to gradient descent, while enumeration
searches it normally.

### `module-library` — verified, blocked on quiet tree

The persistent module library and the curriculum artifact flow. Adds
`tcn/library.py` (new file: `Library`, `Entry`, content-addressed storage, a
manifest, versions, staleness on relearning, and a strict/revalidate
source-fingerprint policy backed by recorded conformance fixtures), extends
`tcn/curriculum.py` with `Artifacts`, `Stage.inherits`, `Stage.publishes` and
publication as an evidence gate, and adds `tcn library list/show/verify`, a
`--library` flag on `tcn curriculum`, a publish hook inside the existing
`synthesize` stage and a `reuse` stage operation to `tcn/cli.py`.
`curricula/system.json` gains a `module_reuse` stage that inherits
`typed_synthesis`'s crystallized program as a candidate operator.

**193 tests pass** (179 shipped plus 14 new in `tests/test_module_library.py`),
with the four `generators/computer` tests enabled via a temporary `node_modules`
symlink, removed before committing. All fifteen stages of `curricula/system.json`
pass end to end, and `typed_synthesis` still reports relaxed loss 1.0066e-06 and
exact conformance.

**One shipped-interface change**: `Curriculum.run`'s runner is now called as
`runner(stage, path, artifacts)` rather than `runner(stage, path)`, because the
inherited module references have to reach the stage somehow and passing them is
what makes the restriction enforceable rather than conventional. Two existing
tests define runners and were updated (3 lines).

Deliberately avoids `tcn/learning.py`, `tcn/select.py`, `tcn/graph.py`,
`tcn/operators.py`, `tcn/runtime.py`, `tcn/search.py`, `tcn/synthesis.py` and
`tcn/training.py`, all of which queued branches change. The only overlap with a
queued branch is `tcn/cli.py` (`perturbation-selection`), and the changes here
are additive: one new subcommand block, one new stage operation, and a publish
hook inside the existing `synthesize` branch. **`git merge-tree` has not been
dry-run against the queue**; do that before landing.

Two things to know when landing:

- **The library's stored `source` is the fingerprint of the tree it was
  published against.** Merging anything into `tcn/` or `generators/` will make
  `tcn library verify --root research/module-library/library` report
  `source_current: false`. It should still pass, because verify goes through the
  recorded fixtures; if it does not, the merge changed operator semantics and
  that is the point of the check.
- `research/module-library/RESULTS.md` records a finding that constrains any
  future chain: **a chain needs exactness at the interface, not accuracy.** A
  stage-1 module wrong at 1 of 384 positions took stage 2 from a unique solution
  to zero conforming programs out of 48.
