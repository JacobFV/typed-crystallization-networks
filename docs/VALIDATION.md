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
accepted every node after deferring three attempts that would have disconnected
remaining gradients. Those deferrals are not evidence that the scheduler produced
the result: an argmax of the trained soft graph, with no crystallizer and no
trials, reaches the same 4/4 and the same frozen program
(`research/crystallization-ablation/RESULTS.md`, arm B0;
`research/perturbation-selection/RESULTS.md`). The exact frozen agent
subsequently achieved **4/4** mean return on another 16 test episodes. The CLI also replayed a frozen-agent episode.
Reports, checkpoint, frozen JSON, standalone `.pyz`, and agent configuration are
under `artifacts/validated-system/joint_prediction_policy/`.

This is a small fixed-structure task with held-out episode addresses. It does not
test unseen relation families, natural language transfer, visual control, or
long-horizon embodiment. No such capability is claimed.

## Unsuccessful and unestablished results

An earlier generic arithmetic/sine scaffold trained for 640 episodes stayed near
chance: mean return **2.03125/4**, with final prediction loss around **0.2806**.
Its report remains at `artifacts/joint-long/report.json`. Replacing that scaffold
with the typed logic experiment established a working joint-learning path.

That 640-episode result does not show what the scaffold can learn. It was
reproduced and audited in `research/scaffold-autopsy/`: the run was stopped before
its learning transition. The same scaffold, same trainer, same learning rate and
same task, run to 5120 episodes, reaches **4/4** deterministic return on held-out
episodes with held-out prediction loss **4.4e-4** on both seeds tested; the
transition occurs near episode 750, 800 and 2600 for seeds 0, 2 and 1. By 2048
episodes the relation itself is learned on all three seeds (prediction loss
1.1e-4 to 9.4e-4); the policy readout is the slower half. Under plain
full-batch supervision the identical program fits the same targets to MSE `1e-5`
at 100% accuracy in about 120 optimizer steps, across twelve width/rate/seed
settings, so neither capacity nor operator search was the obstacle. The generic
scaffold is roughly an order of magnitude less sample efficient than the typed
logic graph, which reached 4/4 in 160 episodes: its per-episode score-function
gradient has a measured signal-to-noise ratio of 0.22, its policy readout must be
found by that noisy term rather than being wired to the latent, and its prediction
and policy gradients transiently conflict (cosine down to −0.68) while it sits on
the plateau. Scaffold choice buys sample efficiency and a shorter search; the
recorded episode budget, not the scaffold, produced the near-chance number.

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

## Corrections from the measurement pass

`research/FINDINGS.md` records an eight-track measurement pass against this
implementation and supersedes claims above it where the two disagree. The
material corrections: progressive crystallization is inert at these budgets and
an argmax of the trained soft graph reaches the same frozen program; the 4/4
joint result holds out episode addresses on a single fixed Boolean function
rather than generating structures, and measures at chance on unseen truth
tables; the serialized description size measures JSON verbosity rather than
learned content; the batch-one latency is dominated by interpreter overhead
rather than the four-operation cost proxy; and exhaustive enumeration recovers
the discrete content of both learned programs in milliseconds.

Fixes applied after that pass are listed in section 10 of the same document.
The numbers in this file were reproduced exactly on this host before and after
those changes: the mixed fixture still crystallizes with exact conformance, and
the joint fixture still reports 0.24884 to 0.00223, 4/4 deterministic return and
4/4 from the exact frozen agent.

`research/perturbation-selection/RESULTS.md` records two further core changes:
the crystallizer now selects by measured perturbation rather than by choice
entropy, and its connectivity guard probes the task objective with the
architecture regularizers removed. The joint numbers above are unchanged by both;
the disconnection deferrals go from four to three, and the corrected guard
catches severed interior regions on this fixture that the previous guard passed
silently.

Generated experiment artifacts are intentionally gitignored. Run the documented
commands to recreate them. Source fingerprints pin replay/checkpoints to their
code revision, so artifacts made before subsequent code changes should not be
silently resumed as though generated by the current implementation.
