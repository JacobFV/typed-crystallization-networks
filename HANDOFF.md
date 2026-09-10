# Handoff — start here

**This file is an index and stays short.** It is updated in place; everything it
points to is append-only. Nothing here is the record — the record is the files
below.

## First 10 minutes on a new machine

```bash
git clone <repo> && cd typed-crystallization-networks
./scripts/setup.sh
.venv/bin/python -m pytest -q          # expect: 288 passed
.venv/bin/python -m tcn train --episodes 160
# expect: 0.248836 -> 0.002231, fully frozen, 4.0 evaluation, 4.0 exact frozen
.venv/bin/python -m tcn demo           # ten capabilities against their baselines
```

## Where everything lives

| you want | read |
|---|---|
| what was measured, authoritatively | [`research/FINDINGS.md`](research/FINDINGS.md) — 44 numbered sections; supersedes everything else |
| **how the project got here, chronologically** | [`docs/PROJECT-LOG.md`](docs/PROJECT-LOG.md) — five phases, with commit SHAs |
| **what we believed and got wrong** | [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) — 14 reversals + process failures |
| what a given session did | [`docs/handoffs/`](docs/handoffs/) — one dated file per session, never edited after the fact |
| the short hostile summary | [`STATUS.md`](STATUS.md) |
| the constitutional spec | [`ARCHITECTURE.md`](ARCHITECTURE.md) — §5 (commitment), §8.1 (four costs, three sizes) |
| what may be merged and when | [`research/MERGE-QUEUE.md`](research/MERGE-QUEUE.md) — **read before merging anything** |
| naming crystallized programs | [`NOMENCLATURE.md`](NOMENCLATURE.md) |
| open questions, with closed ones marked | [`research/AGENDA.md`](research/AGENDA.md) |
| **the next architectural direction** | [`research/algorithm-resynthesis/DESIGN.md`](research/algorithm-resynthesis/DESIGN.md) — post-crystallization algorithm resynthesis; design only, experiment pre-registered but not run |

Per-track raw data and reproduction scripts are under `research/<track>/`.

## State as of 2026-09-09

**Settled:** the inference overhead was *interpreter* overhead, not the typed
representation. Compiling a frozen program to stdlib Python takes the visual
parse from 103,487,972 element operations to 6,144 (§42, branch
`compiled-runtime`).

**Refuted four times:** the progressive irreversible freezing schedule, most
recently in the reversible form its third refutation asked for (§7, §12, §37).

**Open, highest value first:**

1. **Mine abstractions from *pre*-minimisation programs.** §44 measured why the
   obvious rule fails: exact minimisation fuses the reusable fragment into its
   wrapper, so `MAJ3` survives in 1 of 6 minimised programs and the frequency
   statistics never see it. Harness is built and verified in
   `research/earned-abstraction/`; arm 3 proves the task *is* solvable there.
2. **Shorten the programs.** §41 certified that ranking within a fixed scaffold
   cannot (`none exists` at spans 4–29, `unique` at 30). The scaffold is the lever.
3. **Fix or retire `execution_cost`.** It is constant across programs whose
   executed bytecodes differ — no signal, not merely a poor predictor.
4. **The language capability on the post-audit stream** — never measured; see
   the stream trap below.

## Traps that have cost real time

- **The host crashed on 2026-09-10 from memory exhaustion — cap every heavy job.**
  The machine is an NVIDIA GB10 with **unified memory**: CPU and GPU share one
  121 GB pool, so exhausting RAM shows up as `NVRM ... NV_ERR_NO_MEMORY` in the
  kernel log and takes the whole host down. The venv's torch is **CPU-only**
  (`2.14.0+cpu`), so this was plain RAM, from several agents running parallel
  enumeration and sweep jobs (up to 11-way shards) alongside pytest. Rules:
  run **one heavy agent at a time**; wrap heavy jobs in
  `systemd-run --user --scope -p MemoryMax=40G -p CPUQuota=800% <cmd>` so a
  runaway job is killed rather than the host; cap worker processes at ~4 with
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; never run the full suite concurrently
  with an agent's heavy phase.

- **`test_panel_interface.py::test_panel_episode_replays_and_restores` fails in
  any git worktree** with `node_modules` symlinked from the main checkout, on
  main's own code, deterministically. It passes in the main checkout. One such
  failure in a worktree run is expected and is **not your branch's fault**.
- The four `generators/computer` tests need gitignored `node_modules`. Symlink it
  from the main checkout; remove before committing. Twelve further worktree
  failures are just this.
- **Never quote §19's language number without saying which stream.** It
  reproduces only with `hardening='none'`; §24 re-drew the lesson and the track's
  training split is empty on today's default (§39).
- `main`'s FINDINGS jumps **§41 → §43**. §42 is on branch `compiled-runtime` and
  arrives when it merges.
- **Never merge a change to `tcn/` or `generators/` while an agent is measuring
  against main** — `source_fingerprint()` hashes both, invalidating every
  recorded episode.

## Working discipline that keeps paying

Independently spot-check every headline against **raw data**, not against a
summary. That has caught nine wrong conclusions. Two were caught only because an
agent reported a number contradicting a recorded one *instead of routing around
it* — ask for that behaviour explicitly in briefs. Enumeration beside every
synthesis number; constant and random baselines beside every return; probes are
supervision and never model inputs; preserve negative results and invalid logs
rather than deleting them.

See [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) for what happens when this
slips — including a fabricated verification committed by the assistant and
retracted in place.
