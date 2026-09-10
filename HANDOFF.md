# Handoff — start here

**This file is an index and stays short.** It is updated in place; everything it
points to is append-only. Nothing here is the record — the record is the files
below.

## First 10 minutes on a new machine

```bash
git clone <repo> && cd typed-crystallization-networks
./scripts/setup.sh
.venv/bin/python -m pytest -q          # expect: 338 passed (a worktree: 337 + the known panel-replay failure)
.venv/bin/python -m tcn train --episodes 160
# expect: 0.248836 -> 0.002231, fully frozen, 4.0 evaluation, 4.0 exact frozen
.venv/bin/python -m tcn demo           # ten capabilities against their baselines
```

## Where everything lives

| you want | read |
|---|---|
| what was measured, authoritatively | [`research/FINDINGS.md`](research/FINDINGS.md) — ~~44~~ numbered sections (the number 14 is headed §14a and §14b; §42 is a tombstone); supersedes everything else, and `research/record-audit/verify.py --gate` checks it against the artifacts |
| **how the project got here, chronologically** | [`docs/PROJECT-LOG.md`](docs/PROJECT-LOG.md) — five phases, with commit SHAs |
| **what we believed and got wrong** | [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) — ~~14~~ every reversal and corrections, plus process failures |
| what a given session did | [`docs/handoffs/`](docs/handoffs/) — one dated file per session, never edited after the fact |
| the short hostile summary | [`STATUS.md`](STATUS.md) |
| the constitutional spec | [`ARCHITECTURE.md`](ARCHITECTURE.md) — §5 (commitment), §8.1 (four costs, three sizes) |
| what may be merged and when | [`research/MERGE-QUEUE.md`](research/MERGE-QUEUE.md) — **read before merging anything** |
| naming crystallized programs | [`NOMENCLATURE.md`](NOMENCLATURE.md) |
| open questions, with closed ones marked | [`research/AGENDA.md`](research/AGENDA.md) |
| **the next architectural direction** | [`research/algorithm-resynthesis/DESIGN.md`](research/algorithm-resynthesis/DESIGN.md) — post-crystallization algorithm resynthesis; design only, experiment pre-registered but not run |

Per-track raw data and reproduction scripts are under `research/<track>/`.

## Direction as of 2026-09-10 — supersedes the older lists below

**The bottleneck is now the research process, not the architecture.** Results are
being produced faster than the evidence layer can absorb them: the §62 audit found
85 discrepancies in 794 cited figures, and two entries in `CORRECTIONS.md` were
themselves wrong.

**1. Evidence pipeline first, before more research.**
- Machine-generate headline numbers from committed artifacts; no hand-transcribed
  figure where a script can emit it.
- Every `FINDINGS` section should have a verifier whose pass/fail is computed from
  committed raw artifacts. `research/record-audit/verify.py` is the start.
- The record audit is a **routine gate**, not a periodic heroic cleanup.
- Label single-family / single-width / single-configuration evidence explicitly.
  The audit found 25 load-bearing claims resting on one, 16 undisclosed.

**2. The current abstraction hypothesis is schema + specialization** (§60, §53).
The schema crosses domains and widths; the frozen selection vector does not. So
the reusable unit looks like *semantic class + parametric structural schema +
domain/task specialization + multiple certified implementations*, not "a frozen
module". **Keep a hierarchy of identities** — semantic class → structural schemas
→ implementations — and do not collapse it: semantic equality is information and
must not erase factorization information. **Do not invent a single "abstraction
score"** — §57 showed each candidate so far was family-specific. Do not add
polymorphism to the core type system; the sidecar mechanism already works.

**3. No compiler archaeology.** Individual compiler tricks now yield ~1.1× and an
obvious-looking bound was unsound (§59, §61). Move effort up, to better algorithms
and reusable structure.

**4. Scaffold induction is the concrete research loop** (§45, §47, §50): when a
scaffold has no exact solution, enumerate small typed structural edits, add one
reduction/state/path/operator family, test exact solvability and held-out
behaviour, and keep the smallest useful expansion. A dumb typed operator sweep
already repaired 21 of 24 failed scaffolds (§50). This attacks the biggest
remaining human prior: someone currently hand-picks the coarse graph.

**5. Flagship, after the cleanup: one integrated benchmark.** Raw screenshot +
raw textual instruction → identify a relationally-specified widget → act on it in
the computer environment. Compare flat substrate vs inherited learned library vs a
wrong/distractor library, measuring task quality, search space, environment
episodes, compiled work, and which inherited abstractions land on the execution
path. The question it answers: *does the accumulated instruction set make
increasingly integrated intelligence cheaper to acquire?* Stay off Boolean-family
microbenchmarks as the main venue — §58 found zero non-trivial semantic classes
shared across the real artifacts.

**Operational:** one heavy agent at a time, capped (see Traps). Slow the branching
rate.

**The record gate is live (2026-09-10).** `tests/test_record_gate.py` runs
`research/record-audit/verify.py --gate` (~3 s, reads committed artifacts only), so
`pytest` and `scripts/check.sh` fail when a document claim disagrees with its artifact,
a checked claim is reworded away, a section is cited that does not exist, or a
**new FINDINGS section has neither a verified claim nor a `SECTION_UNCHECKABLE`
reason**. Writing a section now means adding its check. Quote headline figures from
`research/record-audit/HEADLINES.md`, which is generated from the artifacts. The
slow audit (`verify.py --demo --tests`) is separate and is not run by pytest.

## Direction update, 2026-09-10 (after §64)

**Index documents do not quote counts** of FINDINGS sections or CORRECTIONS rows;
those drifted silently while the numerical claims stayed verified. Read the files.

**Schemas are falsifiable conjectures — CEGIS over schemas.** A schema is a
conjectured parametric program family; each certified instance is evidence, not
proof. At a new width or domain: instantiate, conformance-check, and on failure
treat it as a counterexample — refine or split the schema class. §64 is the
textbook case: a wrong schema passed two-width certification and failed above the
certified range. Never extrapolate a finite-width certificate to a universal
claim. For finite shape domains, exhaust the parameter values; theorem proving
earns a role only for genuinely parametric families.

**Five levels of reusable knowledge — none of them is "the module":** semantic
transformation → parametric structural schema → specialization policy / prior →
concrete selection vector → frozen implementation. §60 + §64 locate transfer:
within a domain at a new width, schema *and* vector transfer; across domains, the
schema transfers and the vector does not. So cross-domain reuse needs a
**learned specialization prior**, not a copied hard vector.

**Inherited knowledge must earn its name.** Measure programs/episodes to solution
for: no library (N), schema only (N′), schema + learned specialization prior (N″),
and the quality of a hard-transferred vector. If N′ = N the schema is
organisational reuse, not learned intelligence; N″ ≪ N on a genuinely held-out
domain is what begins to count as transfer.

**The library is evidence-carrying and Pareto-shaped, not scored.** Per class:
semantic identity, schemas, implementations, observed domains and widths,
downstream reuse successes and failures, specialisation / execution / description
cost. Let later task evidence decide usefulness. **Stop searching for a universal
scalar ranking objective** — each candidate so far was corpus- or family-specific
(§57).

**Priority order:** (1) the integrated flagship, before any further Boolean-family
study; (2) make inherited knowledge earn its name, per the ladder above; (3)
schemas as falsifiable conjectures; (4) scaffold induction as a cross-domain outer
loop — typed edits enumerated without knowing the repair; (5) keep semantic class
→ schema → implementation separate; (6) keep the evidence gate boring and
automatic — no new audit research program; (7) resource-cap everything heavy, with
*available* memory as the budget.

**Flagship success criterion:** the inherited library makes a previously unseen
integrated visual-language-computer task materially cheaper to acquire than the
same substrate without it, **while a matched distractor library does not** —
instrumented for which schema/class was selected, how much re-specialisation it
needed, how many candidate programs and environment episodes disappeared, and
whether compiled execution got cheaper.

## State as of 2026-09-09

**Settled:** the inference overhead was *interpreter* overhead, not the typed
representation. Compiling a frozen program to stdlib Python takes the visual
parse from 103,487,972 element operations to 6,144 (~~§42, branch
`compiled-runtime`~~ §48 — §42 was never assigned).

**Refuted four times:** the progressive irreversible freezing schedule, most
recently in the reversible form its third refutation asked for (~~§7, §12, §37~~
§1/§3, §10, §37 — §7 and §12 are not about crystallization; §62 audit).

**Open, highest value first:**

1. ~~**Mine abstractions from *pre*-minimisation programs.** §44 measured why the
   obvious rule fails: exact minimisation fuses the reusable fragment into its
   wrapper, so `MAJ3` survives in 1 of 6 minimised programs and the frequency
   statistics never see it. Harness is built and verified in
   `research/earned-abstraction/`; arm 3 proves the task *is* solvable there.~~
   **SUPERSEDED — this restates §44's mechanism, which §46 overturned (the tie-break,
   not minimisation; CORRECTIONS row 15), and §46/§52/§54/§57 have since done the work.**
2. **Shorten the programs.** §41 certified that ranking within a fixed scaffold
   cannot (`none exists` at the eight spans enumerated between 4 and 29, `unique` at 30). The scaffold is the lever.
3. **Fix or retire `execution_cost`.** It is constant across programs whose
   executed bytecodes differ — no signal, not merely a poor predictor.
4. ~~**The language capability on the post-audit stream** — never measured; see
   the stream trap below.~~ **CLOSED — §45 measured it (the counting scaffold has no
   conforming program; the min-prefix program reaches 1.000 on 859) and §63 ships it.**

## Traps that have cost real time

- **The host crashed on 2026-09-10 from memory exhaustion — cap every heavy job.**
  The machine is an NVIDIA GB10 with **unified memory**: CPU and GPU share one
  121 GB pool, so exhausting RAM shows up as `NVRM ... NV_ERR_NO_MEMORY` in the
  kernel log and takes the whole host down. The venv's torch is **CPU-only**
  (`2.14.0+cpu`). **Correction, same day:** the failing allocation was an NVRM
  *GPU* allocation, which CPU-only torch cannot make, so the allocation that failed
  belonged to a GPU-using process. After the reboot a separate project's training
  (`/home/brandonin/Documents/IBM-1`, `train_proprioceptive_motor.py`) was observed
  holding **~66 GB of GPU memory** — the most probable source of the original
  failure, though it was only observed post-reboot. This project's parallel CPU
  jobs (up to 11-way shards, alongside pytest) added RAM pressure on the same
  shared pool; both matter. **Available memory, not the 121 GB total, is the
  budget — run `nvidia-smi` and `free -g` before any heavy dispatch.** Rules:
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
- ~~`main`'s FINDINGS jumps §41 → §43. §42 is on branch `compiled-runtime` and
  arrives when it merges.~~ **Corrected 2026-09-10 (§62 audit):** §42 never existed
  in any commit; the result it was reserved for is **§48**, and FINDINGS now carries a
  §42 tombstone saying so. §14 was used twice; the two sections are headed §14a
  (discrete perception) and §14b (preference). Nothing was renumbered.
- **Never merge a change to `tcn/` or `generators/` while an agent is measuring
  against main** — `source_fingerprint()` hashes both, invalidating every
  recorded episode.

## Working discipline that keeps paying

Independently spot-check every headline against **raw data**, not against a
summary. That has caught ~~nine~~ most of the rows in `docs/CORRECTIONS.md`. Two were caught only because an
agent reported a number contradicting a recorded one *instead of routing around
it* — ask for that behaviour explicitly in briefs. Enumeration beside every
synthesis number; constant and random baselines beside every return; probes are
supervision and never model inputs; preserve negative results and invalid logs
rather than deleting them.

See [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) for what happens when this
slips — including a fabricated verification committed by the assistant and
retracted in place.
