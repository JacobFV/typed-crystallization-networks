# Typed Crystallization Networks

TCN learns typed programs from synthetic transition systems, supervises intermediate
regions, jointly trains prediction and control, and commits the result to immutable
exact programs that later tasks reuse. This repository contains the implementation,
synthetic generators, training and curriculum runners, exact deployment runtime, and
behavioral checks.

*Crystallization* names the conversion of a learned or searched soft structure into
an immutable exact program. It does **not** name a freezing schedule: the specific
strategy of progressive irreversible freezing has been refuted four times over and
plain argmax at equal compute currently wins (ARCHITECTURE section 5).

The [architecture](ARCHITECTURE.md) defines the substrate. The
[blueprint](docs/BLUEPRINT.md) maps every system box to code; the
[implementation guide](docs/IMPLEMENTATION.md) describes contracts, extension points,
and experimental limits. [Implementation rules](AGENTS.md) apply throughout.

## Run

Python 3.11+, Node 22+, npm, and uv are required. The locked setup installs CPU
PyTorch, MuJoCo, Pillow, and the local computer engine's npm dependencies.

```bash
./scripts/setup.sh
./scripts/check.sh
.venv/bin/tcn generators
.venv/bin/tcn curriculum curricula/system.json --workers 2 --out artifacts/system
```

The curriculum runs independent ready stages in separate processes, writes a
resumable evidence journal, and blocks descendants when a prerequisite fails.
Sampling gates establish generator behavior; learning gates separately require
synthesis, crystallization, and closed-loop return.

```bash
# Learn logic → explicit conversion → arithmetic → sine; crystallize and export.
.venv/bin/tcn synthesize --out artifacts/mixed

# Simultaneous latent prediction and policy reinforcement over two objectives.
.venv/bin/tcn train --episodes 160 --out artifacts/joint

# Execute that frozen agent through the same generator host.
.venv/bin/tcn agent artifacts/joint/program.json \
  --config artifacts/joint/agent.json --deterministic --out artifacts/agent.json.gz

# Sample, replay, and inspect the embodied generator.
.venv/bin/tcn sample embodied_world --steps 4 --out artifacts/world.json.gz
.venv/bin/tcn replay artifacts/world.json.gz
.venv/bin/tcn render artifacts/world.json.gz --out artifacts/world-frames
```

Frozen `.pyz` programs run with standard-library Python, without PyTorch or this
repository installed. They accept JSON lines containing typed `inputs`, with
optional `reset` or explicit `state`, and emit typed outputs and recurrent state.

```bash
python3 -I artifacts/mixed/program.pyz < inputs.jsonl
```

For custom tasks, supply a serialized `Program` and `TrainConfig`:

```bash
.venv/bin/tcn train --program program.json --config training.json --out artifacts/custom
.venv/bin/tcn train --resume artifacts/custom/checkpoint.pt --episodes 256 --out artifacts/custom
```

Use `tcn.synthesis.fit` for arbitrary supervised programs and explicit `Signal`
bindings. The CLI synthesis example is a small reproducible acceptance experiment.

## Implemented scope

- Four carriers, explicit encodings/refinements, hard signature checking, universal
  logic/discrete/analytic/structural/temporal/spectral/representation operators.
- Differentiable operator and wiring choices, trainable constants, recurrent state,
  region-constrained probes, progressive quantization pressure, transactional
  hardening, gradient viability checks, immutable callable module reuse.
- A single typed generator host with privileged-state separation, logical time,
  seeded streams, source-pinned snapshots, replay, and delayed composition links.
- Peer generators for logic, arithmetic, relations, oscillators, symbolic language,
  raster text, mesh images, computers, 2D/3D physics, embodied worlds, and composition.
- Joint prediction/probe/policy/value losses, objective distributions, multi-horizon
  targets, probabilistic targets, typed action distributions, checkpoints, and
  prediction/policy gradient-conflict measurements.
- Exact frozen agent execution, standalone program exports, a bit-parallel Boolean
  backend, and explicit model latency/description-size measurements.

Language and computer mechanisms are local source owned by this repository.
There are no sibling-repository imports or runtime links. The computer kernel is
connected to real synthetic shell/file/app state transitions. Embodied agents
have local raster observations, articulated physical links, reachable interaction,
communication, paper, computers, containers, tools, doors, and buttons.

The implementation is a research system, not a trained general language model.
[Validation results](docs/VALIDATION.md) distinguish tested mechanics, demonstrated
learning, unsuccessful experiments, and capability that remains unestablished.

## Measured findings

An eight-track measurement pass ran against this implementation on 2026-09-08.
**[research/FINDINGS.md](research/FINDINGS.md) is the consolidated record** and
supersedes the summary claims above wherever the two disagree; per-track detail
and raw data are under `research/<track>/`, indexed by
[research/AGENDA.md](research/AGENDA.md).

Read it before building on the architecture. In short:

- **Holds up under measurement.** Exact typed execution; dense hierarchical
  supervision, which is the mechanism that actually works; recursive abstraction,
  re-tested after an early negative and now on the output path in 27/27 successes
  with the flat space of 230,400 exhausted without a solution (section 12); the
  persistent module library and curriculum artifact flow (section 25).
- **Refuted.** The *progressive irreversible freezing schedule*, four times over,
  most recently in its reversible form (sections 7, 12, 37).
- **Settled since, and it was the interpreter.** The complete inference path was
  measured at 146x to 90,400x slower than plain Python, with **97.7%** in
  decode/encode/validate marshalling against **0.56%** in operator semantics plus
  the graph walk (section 36). Compiling a frozen program to straight-line stdlib
  Python takes the visual parse from 103,487,972 element operations to **6,144**,
  none of them internal, and flips attribution to 66.7% operator work
  (section 42). What remains is program length, not representation: generated
  Python is 3.1x / 7.1x / 27.6x hand-written, and the visual figure is 11.1x more
  bytecodes at 2.25x each. Ranking the search by description does not fix it and
  is certified not to (section 41).
- **Measured, and mixed — read it before quoting a cost.** Matched neural
  baselines now exist for all three artifacts (section 43). Typed wins *quality*
  on all three: nine CNN arms score **0 exact parse trees** at every width and
  budget against 20/20 rectangles and links from six screens; language 0.9986
  against a best validation-selected 0.657; computer 10/10 against a computed
  5/10 ceiling. But the CNN wins *execution* on the visual artifact by 300-556x
  and *size* by 250x, and the computer artifact is a cost wash. The typed side
  wins deployment footprint across the board: 24-60 MB stdlib-only against
  ~270 MB to import torch. **"Cheaper than a model" is true on some axes and
  false on others**, and which is which depends on the artifact.
- **Where the differentiable path earns its place.** Enumeration settles both
  flagship results in milliseconds, so gradients currently pay only where search
  is coupled to an environment.

Several claims in `docs/VALIDATION.md` were corrected as a result, and the
findings list ten proposed core changes that have been recorded but deliberately
not applied.
