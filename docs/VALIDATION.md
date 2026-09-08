# Validation record — 2026-09-08

These results come from executions in this workspace. They establish reference
implementation behavior and small learned programs, not a trained general agent.

| Check | Result | Evidence |
|---|---|---|
| Python behavioral tests | 86 passed, 53.81 seconds | `tests/`; `pytest -q` |
| Local computer engine tests | 372 passed across 10 suites | `npm --prefix generators/computer/engine test` |
| Computer TypeScript build | Passed | `npm --prefix generators/computer/engine run typecheck` |
| Locked environment installation | Passed | `uv sync --locked --extra test` |
| Wheel build | Passed; includes local TS bridge/kernel/lock, excludes node_modules | `uv build --wheel --no-build-isolation` |
| Full curriculum | All 14 stages passed, two process workers | `artifacts/validated-system/curriculum.json` |
| Language catalog | 179 executable lessons × two seeds generated nonempty prompt/answer | `tests/test_integration_complete.py` |
| Frozen agent CLI | 4 steps, return 4/4, replay identical | `artifacts/validated-agent.json.gz` |
| Embodied raster export | Two actor streams and stable focus streams exported | `artifacts/validated-world-frames/` |

The Python checks include signature rejection, exact/relaxed behavior, set order
invariance, recurrence, library reuse without internal gradients, rollback after
gradient disconnection, standalone execution with isolated Python, typed continuous
action conditioning, probabilistic losses, frozen checkpoint restoration,
privileged-data separation, timing, process isolation, computer causal effects,
physical interaction, occlusion, and episode replay. The bit-parallel backend is
compared against all 16 exact truth tables across 1,024 lanes and recurrent state.

## Learned mixed program

The synthesis fixture learns Boolean XOR, an explicit numeric conversion,
arithmetic, and sine within a supplied coarse graph. All nodes crystallized;
exact conformance passed. Tests also evaluate numeric inputs held out from fitting.
The final relaxed probe loss was approximately `1.01e-6`.

Its exact Python model benchmark measured median **0.0487 ms** and p95 **0.0591 ms**
over 100 batch-one invocations. This includes typed-input validation through typed
output, but excludes observation acquisition, simulation, JSON transport, and
process startup. It is a four-operation program, not an LLM throughput result.
Serialized description size was 17,728 bits; the cost proxy was four operations.
Hardware/software details and the source fingerprint are in
`artifacts/validation-environment.json`.

## Joint prediction and reinforcement

A typed truth-table scaffold learns a latent relation and its goal-conditioned
answer under two objectives. It receives the task bits and the objective bit;
privileged generator targets supply training losses. The supplied graph constrains
possible depth and operators; the correct truth tables are learned. Initialization
copies a predecessor to avoid cancellation from a perfectly uniform gate mixture.

After 160 episodes, average prediction loss over the first/final eight episodes
fell from **0.24884 to 0.00223**. Deterministic evaluation achieved **4/4** mean return
in 16 held-out episodes, across both objectives. Progressive crystallization
accepted every node after deferring four attempts that would have disconnected
remaining gradients. The exact frozen agent subsequently achieved **4/4** mean
return on another 16 test episodes. The CLI also replayed a frozen-agent episode.
Reports, checkpoint, frozen JSON, standalone `.pyz`, and agent configuration are
under `artifacts/validated-system/joint_prediction_policy/`.

This is a small fixed-structure task with held-out episode addresses. It does not
test unseen relation families, natural language transfer, visual control, or
long-horizon embodiment. No such capability is claimed.

## Unsuccessful and unestablished results

An earlier generic arithmetic/sine scaffold trained for 640 episodes stayed near
chance: mean return **2.03125/4**, with final prediction loss around **0.2806**.
Its report remains at `artifacts/joint-long/report.json`. Replacing that scaffold
with the typed logic experiment established a working joint-learning path; it
was not evidence that arbitrary scaffolds learn equally well.

The current system curriculum samples and checks replay of image, language,
computer, and physical generators. Those stages do **not** mean the agent has
learned reading, inverse rendering, computer use, robotics, or their composition.
The reference trainer and generator contracts support experiments in those domains;
training and evaluating a useful unified model remains research work.

The implementation guide records approximation boundaries: exact but
nondifferentiable set transformations; bounded action encodings; conservative
occlusion; simple mesh rendering; terminal computer affordances; CPU-tested
training; per-event computer reconstruction; and platform-local numerical replay.
No natural data has been used for the initial training experiments.

Generated experiment artifacts are intentionally gitignored. Run the documented
commands to recreate them. Source fingerprints pin replay/checkpoints to their
code revision, so artifacts made before subsequent code changes should not be
silently resumed as though generated by the current implementation.
