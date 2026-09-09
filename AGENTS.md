# Implementation rules

> No domain may receive a computational primitive unavailable in principle to
> every other domain. Domain specificity belongs in generated data, not the
> learning substrate.

> Anything that can plausibly be learned as a reusable transformation should
> not be silently implemented as preprocessing.

Read [ARCHITECTURE.md](ARCHITECTURE.md) before changing the implementation. It is
the single constitutional specification; supporting notes do not define another
architecture. Follow the user's latest instructions when they refine it.

- Keep the core data algebra `bool`, `int[n]`, `set[T]`, `tuple[T...]`. Separate
  semantic schemas, numeric/storage representations, and operators. Domain
  labels and schemas may compose these types without becoming new primitives.
- Make conversions explicit. Node candidate legality includes arity, input and
  output types, representations, and refinements. Never use a loss penalty or
  an `Any` escape hatch to admit illegal computation.
- New convenience helpers must expose compositions of existing operations.
  Operator additions need a stated semantic necessity and a corresponding spec
  update; domain-specific shortcuts are not justified by implementation ease.
- A frozen module is an immutable, versioned callable operator with no internal
  gradients. Charge its internals to description size and its execution to cost.
  Check that crystallization leaves remaining trainable regions learning signal.
- Put every data generator directly under `generators/<name>/` using the same
  lifecycle and record contract. The trainer/runtime cannot branch on domain
  names to create special learning paths. Compositions remain ordinary generators.
- Copy and refactor useful implementation into this repository. Own its types,
  interfaces, and behavior here. Do not introduce sibling-repo imports, path
  manipulation, symlinks, or a dependency on an installed legacy project. Copy
  mechanisms and relevant behavioral checks; reshape them to this architecture.
- Keep the admissible synthesis library separate from generator/evaluator
  implementation. Rich simulation labels do not authorize hidden agent
  preprocessing. A tokenizer, detector, parser, object model, or oracle action
  menu cannot silently become a model input or built-in reasoning mechanism.
- Keep agent observations, privileged targets, and audit metadata physically
  separate at the actor boundary. Check time availability, IDs, seeds, ordering,
  and action masks as well as explicit labels. Unknown is not false.
- Use explicit logical time and randomness, versioned semantic configuration,
  and complete replay state. Check copied code for host-clock and live-data
  dependencies. All initial training/development data is synthetic.
- Put region/input/probe/loss bindings and curriculum prerequisites in explicit
  configuration. Do not equate layer index with semantic abstraction.
- Integrated training must jointly optimize future-latent prediction and policy
  reinforcement over a declared objective distribution. Report objective
  conflicts and actual return; auxiliary loss improvement is not task success.
- Validate meaningful invariants: exact/relaxed/exported semantics, gradients
  before freezing, set/order invariance, replay, visibility, composition, and
  closed-loop behavior. Report tests run and limitations. Distinguish a working
  framework from a trained capability; price the complete inference path.

Hand-initialize freely, and prove the initialization is a prior rather than the
answer. Supplying known structure is the intended mode: the coarse scaffold,
predecessor pools, regions, depths, and which frozen module feeds which are all
declared, and biasing a choice distribution toward a plausible candidate is a
prior like any other. What is not a result is an initialization the run cannot
move off, because then the initialization is the answer and the experiment
measured nothing. Three obligations make the difference checkable rather than
arguable:

- **State it.** Every hand-initialization is named in the scaffold's docstring,
  with what it biases and why. An undisclosed one already invalidated a headline
  result here: the policy decoder of `examples/joint.py` was initialized to the
  exact solution, and the reported return was not measuring what it claimed.
- **Ablate it.** Run the same experiment from a neutral initialization and report
  both numbers side by side. If the neutral arm reaches the same program more
  slowly, the initialization was a prior and it did its job. If only the
  initialized arm works, report that as scaffold sensitivity, not capability.
- **Certify it where you can.** This substrate has a stronger test than an
  ablation: enumerate the same candidate space. If exhaustive search finds the
  same program without any initialization, the answer provably lies in the space
  and the initialization only shortened the path to it. That is a certificate,
  not an argument, and `tcn/search.py` produces it whenever the space is small
  enough to exhaust.

Ordinary private implementation helpers are allowed. They must not create new
public semantics, competing schemas, or domain privileges for convenience.
