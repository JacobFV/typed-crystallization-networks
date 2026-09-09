# Handoff — 2026-09-09

One overnight session. Read this, then `research/FINDINGS.md`, then `STATUS.md`.
`research/FINDINGS.md` is the authoritative record: 30 numbered sections, each
with measurements, and it supersedes older claims wherever they disagree.

## The one-paragraph version

The substrate had four defects that silently prevented gradient learning; all
four are fixed and merged. Composition now works end to end: a module learned in
one curriculum stage persists and is consumed as a candidate operator by a later
stage, demonstrated on the shipped 15-stage curriculum. Thirteen capabilities are
measured, several with uniqueness certificates, and a task now poses genuine
delayed credit assignment with two things solving it. The largest remaining gaps
are that the visual ladder stops at edges and that the action hierarchy is
written but unmeasured.

## What is true now, with evidence

Everything below is verified on merged `main` and reproducible from a command or
a file under `research/`.

| capability | evidence |
|---|---|
| typed synthesis, exact export | held-out 5.9e-08 on 584 unfitted points; unique of 96 |
| structural generalization | 3.38-4.00 on unseen gate families vs 2.13-2.56 best constant, interpreter candidate chosen 8/8 seeds unprompted |
| depth generalization | one fixed graph trained at depths 1-2 scores 4.00 at 3, 4, 6, 8 |
| recursive abstraction | module on the output path in 27/27 successes; flat space of 230,400 exhausted with no solution |
| positional reuse | one module at thousands of positions by **three** caller nodes, width-independent |
| pixel segmentation | held-out 0.0, unique among 65,536 |
| edge detection | held-out 0.0, unique among 48 staged |
| same-object relation | 1.0000 held-out vs 0.8842; transfers to `world_2d`/`world_3d` unchanged |
| convolution from raw bytes | 3x3 Sobel-x, 4/4 seeds recover the exact kernel |
| language | 0.9986 on 724 held-out episodes of unseen lengths vs 0.548 majority (one error in 724; corrected from 1.000, FINDINGS §43) |
| computer use | 10/10 held-out documents, generalizing to unseen commands and formats |
| reward learning | 4.00/4 on 8/8 seeds from reward alone, no supervision, no hand-init |
| module library | stage 2 given stage 1: 48 programs unique vs 4.9e10 without |

Run `python -m tcn demo` for ten of these against their baselines. Note
`.venv/bin/tcn` was broken for the life of the project (stale shebang from the
repo rename) and is fixed; `python -m tcn` always worked.

## What is not true, and was corrected

These were believed and are now refuted. Do not reintroduce them.

- **Crystallization does not earn its complexity.** Three independent tracks:
  inert at shipped budgets, harmful at tight ones, and neither DARTS-PT
  selection nor loss-gated eligibility beats plain argmax at equal compute.
  The mechanism: the interval where the better measurement wins is exactly the
  interval where committing is a mistake.
- **The address wall was a dead surrogate, not a relaxation limit.** `eq`'s
  surrogate is exactly 0.0 past |a-b| >= 11 and `lt`'s past 17; the failing
  experiment's gradients were identically zero and a probe was averaging over
  dead nodes.
- **Policy learning does work.** The earlier claim came from a baseline that
  never ran REINFORCE on the typed program at all.
- **179 language lessons was never evidence.** 63 of them are exploitable, 31 by
  copying, 7 solved at exactly 1.000. Fourteen are re-drawn; the default
  distribution changed deliberately and `hardening="none"` preserves the old
  stream bit-identically.

## Start here

1. **The visual ladder stops at edges.** Rungs two and three (glyphs, widget
   kind) were dispatched and did not finish. The bounds are already measured: a
   glyph **is** determined by its bounding box (0.9625, transferring through an
   ink reduction); a widget's `kind` is **not** determined by colour (0.4620 vs
   0.2982); the parent relation **is** exactly determined by geometry alone
   (1.0000 at 368 widgets, zero ties). That last one means **hierarchy assembly
   is well-posed once widgets exist** — the screenshot-to-hierarchy parse is
   two rungs away, not a research program. See `research/visual-ladder/`.
2. **Credit assignment is solved as a task and half-solved as a capability.**
   `generators/computer` at `interface='panel'` poses it, proved by exact DP:
   V*(6) = 1.0000 against a myopic 0.0625, detector run and cleared. Two things
   solve it — model-based enumeration in 89 episodes, and reward-only REINFORCE
   recovering the delayed sequence on 6/6 seeds. **What is not measured is the
   action hierarchy**: reward-only fails to select the argument sub-action, and
   the composite-action arm is written but never run
   (`research/credit-assignment/run_macro.sh`). That is the first thing to run.
3. **Composition is brittle at the interface.** A stage-1 module wrong at 1 of
   384 positions took stage 2 from a unique solution to zero conforming. Chains
   need exactness, not accuracy. This is the most important open risk in the
   composition story.
4. **A stored module is width-specific.** A module hardened at one observation
   width cannot be registered against another — its input type names the
   observation, so `map` raises a signature mismatch. What transfers is the
   *selections*: rebuild the scaffold at the new width and reuse the chosen
   indices. Generalising across widths needs either stored selections or an
   observation type that does not fix the width; neither exists.
5. **`index`'s temperature does two jobs** — the value that makes the gather
   exact kills the address gradient. A fourth instance of the coupling that was
   fixed for `eq`; the same separation has not been applied here.

## Discipline that paid off, and should continue

Every agent's headline claim was independently spot-checked before being
recorded. That caught **six** confident wrong conclusions, three of them mine:
a crystallizer "improvement" that was extra compute; a generator pool that
silently ignored an explicit config, making a held-out run evaluate on its own
training distribution; a lexicographic tie-break presented as a solution; the
address-wall misattribution; a constant-baseline figure that did not reproduce;
and a demo claim read from a still-running track's in-progress files.

Two shipped commands broke without any test noticing — the CLI entry point and
the curriculum at `--workers 2`. **`README.md`'s commands are not covered by the
suite.** Closing that gap is cheap and overdue.

Standing rules that are worth keeping: enumeration beside every synthesis
number; constant and random baselines beside every return; probes are
supervision and never model inputs; a number that looks too clean probably is;
and hand-initialization is fine provided it is stated, ablated, and where the
space is exhaustible, certified by enumeration.

## Mechanics

- `research/MERGE-QUEUE.md` records the merge discipline. The queue is empty.
- Two agent worktrees remain under `.claude/worktrees/`, locked by a live
  process; they are harmless and can be pruned with `git worktree prune` once
  that process exits.
- 288 tests pass. The shipped fixture reproduces at 0.24884 to 0.00223, 4/4
  deterministic, fully frozen, 4/4 from the exact frozen agent.
- `source_fingerprint()` hashes all of `tcn/` and `generators/`, so every core
  change invalidates every recorded episode. That is intended, and coarse.
