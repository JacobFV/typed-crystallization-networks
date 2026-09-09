# Implementation guide

The four carriers and universal operators are the only learning substrate.
Python dataclasses, simulation objects, lesson records, and kernel events are
private implementation structures; they do not become extra learned primitives.

## Types, operators, and graphs

`Type` composes `bool`, `int[n]`, bounded `set[T]`, and `tuple[...]`. `Encoding`
declares integer, fixed, or IEEE floating interpretation, signedness, scale, and
overflow. `Value` validates exact carriers and provides serialization and
relaxation round trips. Role, unit, frame, and bounds are explicit refinements.
Conversions retain refinements and require declared output representations.
Numeric operators currently require matching input schemas; derived units are
explicit output signatures, not a general symbolic dimensional-analysis solver.

Scalar numeric lifts use decoded floats; category/symbol lifts use bit
probabilities. Sets use canonical sorted, padded `(present, element)` slots. This
is permutation invariant but sorting and membership changes are nondifferentiable.
Set operators expose exact gradient boundaries; choice logits can still learn.
Numeric float32 relaxations cannot represent all large integer values exactly;
use exact execution or categorical bit encodings when identity must be preserved.

A `Registry` resolves complete signatures before a candidate enters a node.
A `Candidate` includes an operator and predecessor bindings, so learning selects
wiring as well as operations. `legal_candidates` enumerates bounded legal choices.
A `Program` supplies typed ports, depths, regions, constants, nodes, outputs, and
explicit `(state_name, initial_value, update_source)` recurrence. Within a tick,
execution is acyclic; recurrence commits after evaluating the whole graph.

A `Signal` binds a trace source to a target, permitted regions, type, loss, horizon,
and weight. Supervised callers supply time-aligned examples; the joint trainer
aligns target horizons against episode records. The arithmetic scaffold emits
ordinary mul/add/sin/identity nodes, including explicit conversions. It refuses
implicit categorical, set, or semantic normalization. Construct those encodings
as graph programs rather than hiding them in preprocessing.

`SoftProgram` supports soft mixtures, STE hard-forward trials, exact frozen
boundaries, and precision pressure toward the node's declared encoding. Invalid
numeric domains raise errors. It does not silently clamp logarithm arguments or
coerce incompatible types. Choose scaffolds whose candidates have valid domains.
A precision change requires explicit conversion nodes; pressure alone cannot
change the declared interface.

`Crystallizer` measures entropy/stability, prefers input/output boundaries, trials
hard nodes, retrains the residual graph, checks degradation and gradient paths,
and rolls back failed transactions. A caller can add exact conformance checks.
An immutable registered module is one candidate with stopped internal gradients;
recurrent modules lift state into explicit input/output ports. Definitions are
content addressed and counted transitively once in description size.
`Registry.register_module` and `runtime.save_program` apply `Program.pruned`
first, so a definition charges its live nodes rather than the scaffold it was
learned in; the content address is of the pruned program.

`SoftProgram.export()` materializes an exact candidate program. That operation
alone is not evidence of successful crystallization; check scheduler acceptance
and exact task behavior. The shipped synthesis and joint experiments do both.

`SoftProgram.complexity()` weights `operator.cost` by the choice distribution and
measures execution. `SoftProgram.description_cost()` measures
`L_program_description` instead: expected serialized size in bits of the pruned
hardened program, node liveness and module use both taken as expectations under
the factorized choice distribution, exact at any one-hot point. `synthesis.fit`
exposes it as `mdl_weight` and `TrainConfig` as `description_weight`; both default
to zero. `search.enumerate_fit` takes `rank` in `order`, `description` or `cost`,
returning the first conforming program or the cheapest one, and reports how many
conforming programs it found.

## One generator contract

Each `generators/<name>/manifest.json` names one `Generator` implementation. The
host calls `configure`, `initialize`, `advance`, and `observe`, then constructs
`StepRecord`s. Every generator is loaded through this path.

The actor receives `ActorView` only: visible observations, an observable verb
library, time, and objective configuration. Observations may be public or
actor-owned and carry occurrence/availability times. Latents, probes, transition
records, rewards, and audit metadata are separate. The model itself receives only
configured observation ports: expose objectives as typed observations when they
must condition a policy. Action verb menus do not reveal hidden object preconditions.

Typed actions contain a verb, target, argument values, and actor. The shared host
checks argument signatures and conflicting commands; the generator evaluates
world preconditions and effects. `composition` namespaces child ports and actions,
preserves visibility/time, and reads all links from old records, giving cycles an
explicit delay. Privileged links are explicit generator configuration, not actor
preprocessing. Exogenous links must not conflict with an actor's simultaneous
command to the same child effector.

Addresses separate seed, episode index, split, and random stream. Records and
checkpoints include the local source fingerprint. Restoration rejects a changed
source revision. Replay is validated on the installed platform; MuJoCo and
floating arithmetic do not promise cross-platform bit identity. Source hashes do
not replace dependency pinning: use `uv.lock` and the computer engine's npm lock.

Language constructs symbolic problems before grammatical realization. Its copied
private engine has 179 executable lessons plus one explicitly unimplemented legacy
research-capstone specification; that specification is not an executable generator
or claimed capability. The default language task is unification. Set `lesson`,
`language`, and sufficient prompt `capacity` explicitly for other distributions.
The public record remains the same as every other generator's record.

Mesh images expose exact geometry, camera, depth, IDs, and normals as optional
privileged targets. The CPU rasterizer uses triangle depth testing and supports
text surfaces. It omits near-plane triangle clipping and advanced illumination.
World physics uses MuJoCo rigid contacts, articulated links, motors, hinged doors,
and spring-force grasping. The 2D generator constrains one translational axis.
Interaction occlusion uses conservative oriented bounding boxes; this is a
specified approximation, not a photorealistic visibility engine. Body shapes and
renderer meshes should use matching dimensions (spheres/cylinders use equal radial
scales). Computer and paper focus images share a stable 128×48 raster interface.

The computer bridge replays typed events into an isolated local kernel with an
episode clock and seeded IDs. It supports its implemented synthetic OS commands,
files, applications, and network mechanisms. It does not run arbitrary native OS
binaries. Keyboard access is currently a terminal interface: text, Enter, Backspace.
The copied richer UI state mechanisms are available in the engine, but a general
mouse-driven desktop task is not established by terminal tests. Reconstructing the
kernel per event favors replay correctness over simulation throughput.

## Training, actions, and deployment

`TrainConfig` names any generator, observation ports, targets, objective
configurations, action templates, and optional `ActionBinding`s. A binding maps
a typed argument to an output head containing its distribution parameters:
Boolean Bernoulli logits; category/symbol bit logits; numeric mean/log-standard
deviation; recursive tuples; bounded sets with presence and element parameters.
Numerics use bounded tanh-Normal latent draws and explicit exact encoding.
The policy gradient scores the latent draw, including the transformation density;
it does not claim differentiability through quantized environment actions.

Each binding requires an input port named `action.<template-index>.<argument>` of
that argument's type. It carries the executed argument to prediction. The optional
`action` port carries the template one-hot vector and `dt` carries elapsed time.
Inactive arguments use the deterministic zero-parameter distribution output.
Policies first read previous actions, then prediction/state updates use the action
just selected. The exact `Agent` follows the same ordering. Dynamic targets use
an explicit typed text `__target__` binding, rather than an oracle entity selector.

Prediction heads concatenate targets in configuration order. MSE and Bernoulli
logit targets use one coordinate per lifted value; Gaussian targets use mean then
log-standard-deviation vectors. Optional `probe` outputs contain deterministic
current-target predictions. Dense future supervision and score-function policy
reinforcement are optimized in the same update, along with value, entropy,
complexity, and discreteness terms. Logged gradient cosine reports conflict on
shared parameters; a weighted sum does not guarantee objective alignment.

Checkpoint files store program/library, trainable/frozen flags, weights, optimizer,
temperature, precision pressure, trial choices, configuration, history, and RNG.
The scheduler's acceptance events are experiment reports, not resume state for a
partially executed transaction. Checkpoints are taken between training episodes.
The reference trainer is CPU tested. GPU scalability and distributed training are
not performance claims of this release.

Curriculum `sample`, `synthesize`, and `joint` operations run acceptance fixtures.
The generic `train` operation takes serialized `program`, `training`, optional
frozen `modules`, and an evaluation count in its stage configuration. Each stage
has prerequisites and metric min/max gates. Process workers require a picklable,
module-level runner; ready stages execute in isolation. The curriculum specification
and source must match to resume a journal. Changing either requires a new run path.

Exact artifacts include transitive frozen libraries. The stdlib `.pyz` preserves
runtime recurrence without training dependencies. `PackedBoolean` lowers existing
Boolean operations to Python integer bit lanes; it does not add a primitive or
claim acceleration for mixed-type graphs. Its fast boundary takes prepacked bits;
`batch()` additionally pays typed packing/unpacking costs. The general benchmark
labels its measured boundary explicitly; simulation and raster acquisition are
outside model-only latency.
