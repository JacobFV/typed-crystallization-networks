# A byte is uncommitted, not un-numeric

`role="byte"` excluded a value from `Type.numeric`, so on a pixel or a text octet
the only legal operations were `eq`, `index`, `pack` and the structural family.
Every arithmetic and ordering operator was signature-illegal and there was **no
differentiable conversion out of `byte` at all**. A Euclidean convolution — a
weighted sum over a neighbourhood — was inexpressible, on images and on text
alike, and the computer track's relaxed arm returned exact error 52.0 at held-out
accuracy 0.0 because `unpack`, its single declared exit, is `gradient="none"`.

This branch adds one operator, `interpret`, and one sentence of semantics. The
convolution is now expressible, learned to the exact reference kernel in 4 of 4
seeds, and applied at every position of an image by three caller nodes at max
error 0.0. The cost is a 216x wider candidate space on the fixture that pays for
it, and it is paid only where a program declares the commitment.

Everything below is measured in this worktree. Reproduce with:

```
python research/byte-numeric/legality.py
python research/byte-numeric/surrogates.py
python research/byte-numeric/convolution.py
python research/byte-numeric/enum_projection.py
python research/byte-numeric/cost.py
```

---

## 1. The semantic rule

**A byte is not a category and not a magnitude. It is a carrier whose
interpretation has not been declared, and declaring it is a graph operation.**

The exclusion existed for a real reason and that reason survives intact.
ARCHITECTURE section 1 forbids the *implicit* reinterpretation of a category ID
as a scalar measurement, because averaging two labels denotes nothing. But a
pixel channel is not a category. Neither is it an intensity. It is a raw octet
off a channel, and the mistake was reading its silence as a third semantic class
with its own restricted algebra, when what it actually is is the *absence* of a
declaration. Absence of a declaration is correctly restricted to what holds under
either reading — equality, indexing, structure — and correctly admits a
declaration.

So the role field carries three classes, and the class, not the width, decides
which operators apply:

| class | roles | ordering / arithmetic / mixtures | relaxation lift | `Type.numeric` |
|---|---|---|---|---|
| **uncommitted** | `byte` | no | one scalar | `False` |
| **nominal** | `category`, `symbol` | no | bit decomposition, width `bits` | `False` |
| **magnitude** | `intensity`, and the plain `""` | yes | one scalar | `True` |

and one operator moves between them:

```
interpret : int[n] role=uncommitted -> int[n] role=committed
```

**In one direction only.** From uncommitted to magnitude or to nominal. Never
from nominal to magnitude — that is exactly the reinterpretation section 1
forbids, and it remains illegal at every width and encoding. Never from magnitude
back to uncommitted. Never uncommitted to uncommitted. A generator that knows its
values are labels declares them `role="category"` at the source, and nothing can
undo that; a generator that emits raw octets declares `role="byte"` and a program
must pay a visible node to read them either way.

**The error contract is zero error.** `interpret` preserves the carrier
bit-for-bit: same width, same encoding, same unit, frame and bounds, and only
`role` moves. It is exact and injective on the raw value; there is no operator
back not because information is lost but because permission is not returned.

**The gradient is derived, not chosen.** It follows from the relaxation each
committed class already declares in `Type.flat`:

* to a **magnitude**, the lift is the same single scalar the uncommitted carrier
  had, so the relaxation is the identity, the derivative is exactly 1, and a
  gradient crosses the commitment. `gradient="exact"`.
* to a **nominal** ID, the lift is a bit decomposition of a different width, and
  no derivative from a magnitude into unordered bits is valid. `gradient="none"`
  — an explicit boundary, exactly as ARCHITECTURE section 2 requires of a
  candidate without a valid relaxation.

This is domain-neutral by construction. Nothing above mentions a pixel. The same
two commitments are available to a text octet, a file octet and an audio sample,
and the same prohibition protects a category ID in every one of them.

### Why this is a resolution and not a bypass

The tension was: a byte may be an intensity, where arithmetic is meaningful, or a
category, where averaging denotes nothing. The resolution is that the byte is
neither until a program says which, and the saying is what section 1 already
prescribes — "Representation changes are explicit graph operations... preserves
its role only through a declared conversion/error contract."

Three properties make the declaration checkable rather than a loophole:

1. **It is visible.** `interpret` is a node in the exported program with a
   declared output type. A program that treats a pixel as an intensity says so in
   its serialized text, where it can be read and disputed.
2. **It is falsifiable.** If the octet really is a label, an intensity program
   will not fit the data. The commitment is a hypothesis the search pays for and
   the evidence rejects, not an assumption the substrate grants.
3. **It does not launder a category.** The nominal roles are untouched: still
   excluded from `numeric`, still bit-decomposed, still with no exit. Measured in
   §2, the `category` column of the legality table is byte-for-byte identical to
   the `byte` column.

The previously available crossing — `pack` on a one-field tuple, which
`research/object-identity` used to reach collinearity — remains legal and remains
what it always was: a bit-level reinterpretation that declares nothing and, being
`gradient="none"`, cannot be learned through. §3 measures exactly what that
difference is worth.

### Amendment text

**ARCHITECTURE section 1**, appended after "There is no implicit reinterpretation
of a category ID as a scalar measurement.":

> A carrier's semantic role is therefore one of three classes, and the class, not
> the width, decides which operators apply. A **magnitude** is a scalar
> measurement: ordering, arithmetic, aggregation and mixtures denote. A
> **nominal** identifier is unordered: only equality denotes, averaging two of
> them denotes nothing, and a relaxation gives it a categorical rather than a
> scalar lift. An **uncommitted** carrier is a raw unit off a channel — a pixel
> channel, a text octet, a file octet, an audio sample — whose interpretation has
> not been declared. It is not a third kind of number; it is the absence of a
> declaration, and it is restricted to exactly what holds for either reading,
> which is equality, indexing and structure. A generator that knows its values
> are labels declares them nominal at the source, and no later operation may undo
> that.
>
> Committing an uncommitted carrier is an explicit graph operation with a
> declared conversion/error contract, in one direction only. The commitment
> preserves the carrier bit-for-bit — same width, encoding, unit, frame and
> bounds — and changes only the declared meaning, so its error is exactly zero
> and it is invertible in value though not in permission. What is forbidden is
> the *implicit* reinterpretation and the *reverse* one: there is no conversion
> from a nominal identifier to a magnitude at any width, and none from a
> magnitude back to an uncommitted carrier. A program that treats a pixel as an
> intensity therefore carries that claim visibly in its exported text, where it
> can be read off and falsified by the data, instead of smuggling it through a
> bit-level reinterpretation.

**ARCHITECTURE section 2**, Representation row becomes
`encode/decode, quantize/dequantize, pack/unpack, interpret`, and this is
inserted before "This is a small candidate inventory":

> `interpret` is the section 1 commitment as an operator: one input, one declared
> output type, the same carrier and the same encoding, and only the `role` moves,
> from uncommitted to magnitude or to nominal. It is the sole exit from an
> uncommitted role and it has no inverse. Its gradient is not a free choice but a
> consequence of the relaxation each committed class already declares. Committing
> to a magnitude keeps the same single-scalar lift the uncommitted carrier had,
> so the relaxation is the identity, the derivative is one, and a gradient
> crosses the commitment. Committing to a nominal identifier moves to a
> categorical lift of a different width, and no derivative from a magnitude into
> unordered bits is valid, so that commitment is exact at an explicit gradient
> boundary. Both commitments are exact in the forward direction; only one of them
> is differentiable, and which one is decided by the semantics rather than by
> convenience.

---

## 2. The implementation, and what each operator's case is

Four files, all in core.

* **`tcn/types.py`** — the role vocabulary (`UNCOMMITTED_ROLES`,
  `NOMINAL_ROLES`, `MAGNITUDE_ROLES`) and `interpretable(source, target)`.
  `Type.numeric`, `Type.width` and `Type.flat` now read those sets instead of
  three separately maintained string literals, so the three classes have one
  definition. **No behaviour changed here**: `numeric` still excludes exactly
  `{category, symbol, byte}` and `flat` still bit-decomposes exactly
  `{category, symbol}`.
* **`tcn/operators.py`** — `interpret` joins `CONVERSIONS`, with the resolve
  branch above. The gradient is written `"exact" if output.numeric else "none"`,
  so it is derived from the class rather than hardcoded per role.
* **`tcn/learning.py`** — the relaxation, which is the identity. It is only ever
  reached for the magnitude commitment; the nominal one is `gradient="none"` and
  is returned by `exact_tensor` before the dispatch.
* **`tests/test_byte_commitment.py`** — 8 checks, listed at the end of §6.

### Operator-by-operator, on the committed weighted-sum path

`research/byte-numeric/surrogates.py`, measured:

| operator | gradient | what the relaxation is |
|---|---|---|
| `add` (address arithmetic) | `exact` | true derivative |
| `index` (the gather) | `surrogate` | `softmax(-(addr-k)^2/tau)`, measured in §3 |
| **`interpret`** (byte -> intensity) | **`exact`** | **identity, derivative 1** |
| `decode` (int[8] -> float32) | `surrogate` | identity in value and derivative |
| `mul` (weight) | `exact` | true derivative |
| `sum` (accumulate) | `exact` | true derivative |
| `interpret` (byte -> category) | `none` | exact, explicit boundary |

Four of the six operators on the path have an exact derivative, and **neither
`eq` nor any ordering comparison appears on it at all**. That matters, because
the two dead-surrogate faults on record are properties of exactly those two.
Measured here at tau=1 (`surrogates.py`), confirming FINDINGS 16 and 19
independently:

| operand distance | `eq` value | `eq` d/da | `le` value | `le` d/da |
|---:|---:|---:|---:|---:|
| 1 | 3.679e-01 | 7.358e-01 | 7.311e-01 | -1.966e-01 |
| 5 | 1.389e-11 | 1.389e-10 | 9.933e-01 | -6.648e-03 |
| 11 | **0** | **0** | 1 | -1.669e-05 |
| 16 | 0 | 0 | 1 | -1.192e-07 |
| 17 | 0 | 0 | 1 | **-0** |
| 48 | 0 | 0 | 1 | -0 |
| 192 | 0 | 0 | 1 | -0 |

`eq` is exactly zero in value and derivative from distance 11 and `le` from 17,
as recorded. The convolution's operating distances are byte-scale (the Sobel-x
target spans [-572, 522] on the training images), and the numeric route is
immune to both because it contains neither operator. The temperature fix landing
on another branch is orthogonal to this result rather than a prerequisite for it.

---

## 3. The demonstration: a Euclidean spatial operator over raw pixels

`research/byte-numeric/convolution.py`. Data is the shipped `geometry`
generator's `pixels` observation, read through `record.actor_view()`; nothing
outside the observation reaches a program input. R=8, so 192 raw bytes per image
and 36 fully interior 3x3 blocks. Six training images (216 rows) and six unseen
held-out images from the `test` split (216 rows).

The neighbourhood needs no new operator, exactly as expected: `add(pos, off)`
computes each tap's address and `index` gathers it. The convolution is

```
conv(pos) = sum_k  w_k * decode(interpret(index(obs, add(pos, off_k))))
```

### The relaxed gather, measured before it was used

`index`'s relaxation is a softmax over all 192 addresses. Max error of the
relaxed gathered byte against the exact one, over the 3x3 stencil:

| index temperature | max byte error |
|---:|---:|
| 1.0 (shipped) | 28.07 |
| 0.25 | 2.244 |
| 0.05 | **0** |
| 0.01 | 0 |

At the shipped temperature the gather returns a blur of the neighbourhood, so
arm A runs at tau=0.02 where it is exact. These nodes have **one candidate each**,
so the D2 coupling (one temperature for both the surrogate and the candidate
softmax) cannot bite: a one-way softmax is temperature-invariant. §3.3 shows what
happens when the node *does* have a choice, which is a different story.

### 3.1 Arm A — learn a 3x3 Sobel-x kernel from raw bytes

50 nodes, **zero discrete choices**, nine trainable `float32 role=intensity`
weights initialised at 0 with `constant_noise=0.05` per seed (`SoftProgram`
zero-initialises trainable constants as well as choice logits, so without the
noise four seeds would be one outcome repeated four times). 600 Adam steps at
lr 0.25, index tau 0.02.

| seed | held-out MSE | held-out max abs error | weights rounded to integers | seconds |
|---:|---:|---:|---|---:|
| 0 | 1.870e-01 | 0.990 | = reference | 5.1 |
| 1 | 2.606e+02 | 36.98 | = reference | 8.4 |
| 2 | **3.108e-06** | **0.0040** | = reference | 8.2 |
| 3 | 1.114e-03 | 0.0765 | = reference | 5.9 |

**4 of 4 seeds recover the exact reference kernel** `[[-1,0,1],[-2,0,2],[-1,0,1]]`
after rounding, and the rounded program's held-out max error is **exactly 0.0**
in all four. Seed 2's raw learned weights are within 2.9e-06 of the reference in
every one of the nine taps. Rounding is not a rescue here: it is the precision
pressure ARCHITECTURE section 5 already prescribes, applied to a crystallized
constant.

Baselines on the same held-out rows:

| predictor | held-out MSE |
|---|---:|
| predict 0 | 2.181e+04 |
| predict the training mean | 2.168e+04 |
| learned kernel (best seed) | 3.108e-06 |
| learned kernel, rounded (all seeds) | **0** |

That is a factor of 7.0e9 against the better constant baseline before rounding,
and exact after.

**Against enumeration** (`enum_projection.py`). Enumeration cannot address a
continuous part at all — `SearchResult.continuous` reports the trainable
constants and declines. The comparable discrete question is the same nine taps
with each weight drawn from a 5-value pool:

| | |
|---|---|
| discrete analogue space | 1,953,125 programs |
| measured rate | 1,613 programs/s on 108 rows |
| projected exhaustive wall clock | **1,211 s** |
| conforming found in the first 40,000 | 0 |
| gradient descent, strictly larger continuous class | **5.1–8.4 s per seed, 4/4 exact** |

So on this target relaxation is 150–240x faster than exhausting a *coarser*
hypothesis class, which is the first time gradient descent has clearly won on a
pixel task in this tree since FINDINGS section 14.

### 3.2 Arm B — the same family, small enough to certify

Two taps, each choosing its address from the 3x3 stencil (9 candidates) and its
weight from `{-1, 0, +1}` (3 candidates): 9x3x9x3 = **729 programs**. Target is
the horizontal derivative `g(y, x+1) - g(y, x-1)`.

Enumeration exhausts all 729 in **0.276 s** and returns **2 conforming**, which
are exactly the two orderings of one program:

```
{addr_a: tap (1,0), w_a: -1,  addr_b: tap (1,2), w_b: +1}
{addr_a: tap (1,2), w_a: +1,  addr_b: tap (1,0), w_b: -1}
```

Held-out max error **0.0** on six unseen images. The non-uniqueness is the
commutativity of a sum, not a supervision gap.

### 3.3 Arm B' — the `pack` control, and the boundary this branch removes

The identical scaffold with the byte leaving `role="byte"` the old way —
`tuple(byte)` then `pack` to a plain `int[8]` — has the identical 729-program
space, the identical 2 conforming programs and the identical held-out 0.0 under
enumeration. The two arms differ only in whether a gradient crosses the exit.

Four seeds, `init_noise=0.5`, 600 steps at lr 0.15, swept over the index
temperature. "addr logit grad" is the max-abs gradient at the two address-choice
nodes' logits at the first backward pass:

| index tau | `interpret` exact | `interpret` addr logit grad | `pack` exact | `pack` addr logit grad |
|---:|---:|---|---:|---|
| 1.0 | **3/4** | 5.24e-01 … 1.349e+03 | 0/4 | **0.0 … 0.0** |
| 0.25 | 2/4 | 1.134e+00 … 3.482e+03 | 0/4 | 0.0 … 0.0 |
| 0.05 | 1/4 | 6.51e-05 … 6.309e+03 | 0/4 | 0.0 … 0.0 |
| 0.02 | 0/4 | 1.51e-17 … 9.972e+03 | 0/4 | 0.0 … 0.0 |

The address-choice logits behind `pack` receive **exactly 0.0** at every
temperature — not `None`, because other nodes keep the graph alive, but exactly
zero, which is the hard boundary FINDINGS section 23 identified as "distinct
from the three numerical mechanisms in section 16" and "no temperature fix
reaches". Behind `interpret` the same logits receive up to 1.3e+03 and the arm
conforms in 3 of 4 seeds. **That is the whole change, isolated to one node.**

**A trade this measured, and it is not free.** The index temperature that makes
the gather exact (§3, tau <= 0.05) is the one that kills the address gradient
(3/4 -> 0/4). One number at one node is doing two jobs — sharpening a surrogate
and deciding whether a mixture is differentiable — which is the D2 coupling
appearing at `index` rather than at `eq`. Arm A escapes it only because its
addresses are computed constants; any scaffold that both gathers relaxedly and
*chooses* an address must pick a compromise. The best setting found here is the
shipped tau=1.0, where the gather is blurred by 28 byte units and the arm still
conforms 3 of 4 times. A fix separating a node's surrogate temperature from its
choice temperature would remove this trade too, and this is a fourth instance of
the same defect.

### 3.4 Applied at every position

The crystallized kernel is registered as a module and called by the shipped
three-node `insert`/`pair`/`map` caller (`tcn.scaffold.positional_scaffold`),
with no change to the caller and no new operator:

| resolution | positions | caller nodes | max error | median seconds |
|---:|---:|---:|---:|---:|
| 8 | 36 | 3 | **0.0** | 0.027 |
| 16 | 196 | 3 | **0.0** | 0.621 |

Three caller nodes at both widths, against the reference convolution recomputed
on unseen `test`-split episodes. The module's description is 3,086,264 bits and
its execution cost 50.0; the description is dominated by the 192-field byte tuple
type repeated in every node's serialized signature, which is a property of the
record interface rather than of this change.

**Verdict: a Euclidean convolution over raw pixels is expressible, learnable and
crystallizable.** It was none of the three before.

---

## 4. What it costs

`research/byte-numeric/cost.py`.

### 4.1 Candidate width, per declared node output

`tcn.graph.legal_candidates` over the full operator inventory (63 names), on a
port set of raw bytes, and then on the same set with two committed magnitude
ports added:

| declared node output | bytes only | + committed magnitude |
|---|---:|---:|
| `bool` | 13 | 33 |
| `byte` | **10** | **10** |
| magnitude `int[8]` | 2 | 52 |
| index `int[16]` | 16 | 16 |

The `byte` and index rows are unchanged, which is the point: **the widening is
not automatic**. A byte is still not numeric, and node output types are declared,
so the arithmetic family opens only at a node whose declared output is a
magnitude, downstream of a node a program paid for. The whole-inventory legality
count is 3 of 30 on an uncommitted byte, 3 of 30 on a nominal ID (identical, as
required), and 21 of 30 on a committed magnitude (`legality.py`).

### 4.2 The existing fixture, before and after

`research/discrete-perception`'s rung-3 foreground module —
`fg(px) = truth_t(truth_s(cmp(r), cmp(g)), cmp(b))` over the 5-value byte pool —
is the tree's canonical byte task, exhausted at 32,000 programs. The "after" arm
inserts one `interpret` node per channel and lets each channel predicate also be
`lt`/`le`/`gt`/`ge` against a magnitude threshold. 8 training images at 48 pixels
each (384 rows), 8 unseen held-out images at every pixel (512 rows), foreground
fraction 0.516.

| | shipped (byte only) | with commitment | ratio |
|---|---:|---:|---:|
| space size | 32,000 | 6,912,000 | **216x** |
| exhausted within a 32,000-program budget | yes (25.6 s) | no | — |
| projected exhaustive wall clock | 25.6 s | 3,900 s | **152x** |
| uniform random conforming density (20,000 draws) | 0.08165 | 0.02400 | **0.29x** |
| projected conforming programs | 2,613 | 165,888 | **63x** |
| first conforming, held-out max error | 0.0 | 0.0 | — |
| held-out-exact fraction of conforming programs | 0.84 | 0.91 | +0.07 |
| gradient descent, train-exact of 4 seeds | 4/4 | 4/4 | 0 |
| gradient descent, held-out-exact of 4 seeds | 4/4 | 4/4 | 0 |

(The random-draw estimator is validated by the shipped arm, where it projects
2,613 conforming against the exhaustive count of 2,608.)

**Read honestly, the cost is this.** Enumeration on this fixture goes from
affordable to not: 25.6 s becomes a projected 3,900 s, because the space is 216x
wider and 3.4x more dilute. And the conforming set grows 63x, from ~2,600
programs to ~166,000, so `enumerate_fit`'s tie-break has far more ways to be
wrong — which matters, since FINDINGS section 14 already measured that tie-break
being wrong on fresh episodes at this exact fixture.

**What it does not cost.** Nothing previously solvable became unsolvable. The
first conforming program still has held-out max error 0.0. The *quality* of the
conforming set improved rather than degraded (0.84 -> 0.91 held-out-exact),
because an ordering predicate on an intensity generalizes across shades where an
equality against a specific byte does not. And gradient descent is completely
unaffected: 4/4 train-exact and 4/4 held-out-exact in both arms.

So the trade is **enumeration pays, relaxation does not** — and the whole point
of the change is that relaxation now works on bytes at all. That is a coherent
bargain rather than a wash, but it should be stated plainly: a track that relies
on exhausting a byte space must not add `interpret` to its pool without checking
its budget, and the 216x is a per-channel 6x compounding, so it grows as
`(1 + |comparisons| )^{channels}`.

---

## 5. Reproduction of the shipped fixtures

| fixture | before | after |
|---|---|---|
| `python -m tcn train --episodes 160` | 0.248835613951087 -> 0.0022308224288281053, mean return 4.0/4, `fully_frozen: true` | **byte-identical output** |
| `python -m tcn synthesize` (the mixed fixture) | — | `exact_conformance: true`, `fully_frozen: true`, loss 1.0066399909192114e-06, `agrees_with_enumeration: true`, unique among 96 |
| `pytest tests` | 179 passed | **187 passed** (8 new) |

The `tcn train` JSON is byte-for-byte identical before and after the change,
which is the strongest available statement that no existing path moved.

---

## 6. Limitations, and what this does not claim

* **The target is a reference filter, not a task label.** Arm A learns Sobel-x
  against an arithmetic reference computed outside the substrate. That
  establishes expressiveness and learnability of a weighted sum over raw pixels;
  it does *not* establish that a convolution is the right feature for any
  generator's probe. Wiring the crystallized kernel into a downstream perception
  target is the obvious next rung and is not attempted here.
* **The relaxed address trade is unresolved.** §3.3 measures it and does not fix
  it. A scaffold that both relaxes a gather and chooses an address must pick a
  compromise temperature until the surrogate/choice temperatures are separated.
* **`interpret` cannot change width in the same node.** By design — the
  commitment is a role declaration and nothing else — but it means a magnitude
  needs a following `encode` or `decode` before it can be summed without
  overflow, which is two nodes where a fused operator would be one. That is the
  right trade for auditability and it is a real cost in description bits.
* **Whether a commitment is *correct* is not checked by the type system**, and
  cannot be. It is checked by the data. A search may declare a genuinely
  categorical octet an intensity; it will simply fail to fit, and the declaration
  will be visible in whatever it exports.
* **`pack` was not removed.** The one-field-tuple crossing that
  `research/object-identity` used still works and still declares nothing.
  Deprecating it in favour of `interpret` is a separate decision with a live
  dependent result; this branch only makes the principled route available and
  measures the difference.
* **Measurement hygiene.** Every gradient number here reports its `init_noise`
  and, for arm A, its `constant_noise`, because `SoftProgram` zero-initialises
  both choice logits and trainable constants and `torch.manual_seed` alone varies
  nothing. Every one-candidate node in this track's scaffolds is built with
  `selected=None`, because `SoftProgram` treats any `selected` node as frozen and
  detaches it — the severing bug confirmed three times in FINDINGS.

### Checks added to the core suite (`tests/test_byte_commitment.py`)

1. an uncommitted byte and a nominal ID stay illegal for `add`, `sub`, `mul`,
   `lt`, `le`, `neg`, `sum`, `mean`, and legal for `eq`;
2. `interpret` resolves byte -> intensity as `exact` and byte -> category as
   `none`, and rejects all seven reverse and same-class directions;
3. `interpret` rejects any change of width, encoding, signedness, overflow
   contract, unit, frame or bounds, and requires an explicit output;
4. `interpret` is exact and bit-preserving at 0, 1, 127, 128 and 255, and the
   magnitude keeps the scalar lift while the nominal ID takes the 8-wide one;
5. the magnitude relaxation is the identity with derivative exactly 1, and the
   nominal one is a non-differentiable 8-wide tensor;
6. the weighted sum over raw bytes executes exactly;
7. the gradient reaches the kernel weights *through* the committed bytes, with
   the analytically correct values;
8. a committed program serializes to an identical digest and crystallizes into a
   callable `gradient="none"` module.
