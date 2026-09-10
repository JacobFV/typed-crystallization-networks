# The complete inference path, priced

Research track `inference-cost`.  Produced with the repository `.venv`
(Python 3.13.15) and, for the deployment rows, the *system* interpreter
`/usr/bin/python3` 3.12.3 with `-I` — no repository on the path, no virtualenv,
no torch, no numpy.  **Nothing under `tcn/` or `generators/` is modified.**
Every table below is regenerated from `out/*.json` by the scripts in this
directory; nothing is transcribed from another document except where a row is
explicitly labelled as inherited from `research/baselines/RESULTS.md`.

> **The host is shared.**  Every wall clock here is reported beside a
> load-independent count — **operator applications**, taken by wrapping
> `Registry.exact` for the duration of the call — so a reader on a different
> machine can reprice the row.  Where a single number is quoted it is the median
> of repeated samples with the garbage collector disabled; minima are in the
> JSON.

---

## 0. Verdict, stated plainly

ARCHITECTURE §9 asks for "complete-path cost/memory/batch-one latency".  It has
now been measured, end to end, on four artifacts.  Three findings, in order of
how much they matter to the project's pitch.

**1. The complete path is dominated by neither the operators nor the model —
it is dominated by the typed value layer re-encoding the observation at every
graph edge.**  On the screenshot parse, **97.7%** of execution time is
`tcn/types.py`'s `decode`, `encode` and `validate_raw`; the operator semantics
and the graph walk that do the actual work are **0.6%**.  Cost per operator application is not a
constant: it is about **3.5 µs + 0.07 µs per element of the widest value on that
edge**, which is why the same interpreter costs 3.4 µs/op on the mixed fixture
and 239 µs/op on a 3,072-byte raster.  `execution_cost` counts operators and is
therefore blind to the term that dominates.

**2. It is a caching bug, not a fundamental limit — but the fix is not enough on
its own.**  Memoizing `Value.decoded` and skipping revalidation of carriers the
interpreter itself produced (monkeypatched, `tcn/` untouched, outputs checked
**bit-identical**) takes the parse from **15.4 s to 4.5 s**, a 3.4× win for two
mechanical changes.  The remaining 4.5 s is still 26,000× a plain-Python parse
that returns the identical 19 rectangles in **0.17 ms**.

**3. The exported artifact's size is a serialization artifact, decisively.**
`visual.pyz` is **117.7 MB** — large enough that a push of it was rejected — and
the program inside it is **611 typed nodes, 24 operators, 41 KB of node and
operator declarations, 4.6 KB of constants and 21.3 bits of learned selection**.
**99.68% of its minified JSON is repeated type declarations**: a 3,072-position
raster is a `tuple` whose 3,072 item types are each written out as a separate
JSON object, at every node and every operator signature that touches it.  A
further 71% of the file on top of that is `json.dumps(indent=2)` whitespace,
which `zipapp`'s default `compressed=False` then stores verbatim.  The file
`gzip -9`s to **0.573 MiB** as shipped and to **149 KB** if minified first —
206× and 830×.  Compression ratio rises monotonically with declared observation
width across the four artifacts (3.6× → 48× → 170× → 206×), which is what
repeated type declarations look like and is not something distinct learned
content could do.  This is FINDINGS §3's (track 6's)
`description_bits`-measures-JSON-verbosity fault, now in the shipped artifact.

The honest one-line claim is at the bottom of §9.

---

## 1. What was measured, and on what

| artifact | what it does | evidence it works | rebuilt from |
|---|---|---|---|
| **visual** | a 32×32 RGB screenshot → a set of widget rectangles with parent keys | 215/215 rectangles on 12 held-out screens, 173/215 parent links vs a 0.140 baseline (`research/visual-ladder/RESULTS.md`) | `out/rung3.json` selections, as `parse_report.py` rebuilds them |
| **computer** | a typed agent acting in a live OS: read the file, write the successor digit | 10/10 held-out documents, mean return 2.0/2 (`research/computer-capability/RESULTS.md`) | `out/search.json` enumerated selections, as `closed_loop.py` builds them |
| **language** | grammaticality of a Dyck word, from raw prompt bytes | 1.000 at lengths 8–14 never trained on, majority constant 0.548 (`research/language-capability/RESULTS.md`) | `stage_b.json` selections, as `final_eval.py` rebuilds them |
| **mixed** | `answer = sin(xor(a,b) + x)` — the **historical reference** | held-out max abs error 2.6e-8 (`research/baselines/RESULTS.md` §3) | `artifacts/demo/synthesis/program.json` |

Nothing was searched, fitted, tuned or re-selected in this track.  Every program
is the one its own track froze.

---

## 2. The itemised complete path, in process

Batch one.  `acquire` is the generator producing the raw observation — for
**computer** that is a real round-trip through the OS bridge, not a synthetic
render.  `encode` is `Value.of` building the typed carrier.  `execute` is
`program.run`.  `decode` is `.decoded` on the outputs.

| artifact | acquire | encode | execute | decode | **total** | operator applications | µs / operator |
|---|---|---|---|---|---|---|---|
| mixed | — (inputs given) | 0.0022 ms | **0.0138 ms** | 0.0004 ms | **0.0163 ms** | 4 | 3.4 |
| language | 1.312 ms | 0.087 ms | **2.666 ms** | 0.0001 ms | **4.061 ms** | 164 | 16.3 |
| visual | 4.92 ms | 1.78 ms | **15,364 ms** | 0.023 ms | **15,371 ms** | 64,346 | 238.8 |
| computer *(per agent step)* | **967.85 ms** *(live kernel)* | — | **13.23 ms** *(encode + two full passes + decode + action binding, not separable without editing `tcn/`)* | — | **981.1 ms** | 46 | 287.6 |

`tcn/agent.py:Agent.act` runs the program **twice** per environment step — once
to choose the action, once to commit the executed action to state — so the
computer row's 13.23 ms and 46 operator applications are both for two full
passes.  Its encode/execute/decode cannot be split further without editing
`tcn/`, so they are reported as one cell.

**What §9's recorded 0.0487 ms was.**  It is the `execute` cell of the mixed row
and nothing else.  The complete path on the same machine is **0.0163 ms** in
process (this measurement is *faster* than the record, on a pruned program with
gc disabled) and **0.035 ms** through the shipped `.pyz` including JSON
transport — so for the mixed fixture the omitted stages happen to be small.
They are not small anywhere else: for **computer** the omitted acquisition stage
is **73×** the program, and for **language** acquisition and encoding together
are 34% of the path.

### 2.1 Cost per operator application is a function of value width

The load-independent unit is not constant, and this is the whole story:

| artifact | widest value on an edge | µs / operator |
|---|---|---|
| mixed | 1 float | 3.4 |
| language | 129-element prompt tuple | 16.3 |
| visual | 3,072-byte raster | 238.8 |
| computer | 4,097-element terminal (+1,022-float action tail) | 287.6 |

A least-squares reading of those four points is **≈ 3.5 µs + 0.069 µs per
element**.  `Program.execution_cost` sums a per-operator constant and cannot see
the second term, which is why it is ~2.5 orders of magnitude off on the mixed
fixture (`research/baselines/RESULTS.md` §7) and **four** orders off here:
`execution_cost` for the parse is 1,429,970 "cost units" against 64,346 real
operator applications and 15.4 s.

---

## 3. Where the execution time actually goes

`cProfile` over one screenshot parse, `tottime` grouped by role.  The profiler
inflates the absolute seconds about 3.5×; the shares are the measurement.

| role | profiled seconds | share |
|---|---|---|
| Type.decode (types.py:171,175) | 35.05 | **61.6%** |
| Type.encode (types.py:134,142) | 9.80 | **17.2%** |
| validate_raw (types.py:185) | 5.30 | **9.3%** |
| numeric/type guards called by both | 3.18 | **5.6%** |
| Type/Value dict round-trip | 2.22 | **3.9%** |
| operator semantics + graph walk | 0.32 | **0.6%** |
| other | 1.00 | **1.8%** |
| **typed value layer, total** | **55.54** | **97.7%** |
| total profiled | 56.87 | 100% |

**This is the honest headline, and it is fixable engineering rather than a
fundamental limit.**  `Registry.exact` decodes *every argument in full* on every
call (`xs=[v.decoded for v in args]`), `Value.decoded` is an uncached property
that walks the whole recursive carrier, and `Value.__post_init__` re-validates
every carrier the interpreter itself just constructed.  A 3,072-element raster
therefore pays a full 3,072-element decode at each of the 64,346 operator
applications that can see it.  FINDINGS §26 found 73% of one apply path in
`Value.of` re-encoding; on this artifact the same family of faults is **97.7%**,
and `Value.of` re-encoding is 17.2% of it — decoding is the larger half.

### 3.1 What two mechanical fixes buy, with outputs checked identical

Monkeypatched in the measurement script — `tcn/` is untouched — and the parsed
rectangle set is verified **bit-identical to the unpatched program** in both
arms:

| arm | parse latency | speed-up | still slower than plain Python by |
|---|---|---|---|
| as shipped | 15,364 ms | 1.0× | 90,400× |
| memoize `Value.decoded` | 6,711 ms | 2.3× | 39,500× |
| + skip revalidating interpreter-produced carriers | **4,476 ms** | **3.4×** | 26,300× |

So roughly two thirds of the interpreter's cost is a missing cache.  The
residual third is the structural cost of passing whole typed values across every
edge, which needs a representation change (a view or a slice type), not a cache.

---

## 4. The three references

### 4.1 Plain Python — the same computation, hand-written

Every reference here was checked to produce the **identical output** on the
identical input, not merely a similar one.

| artifact | TCN complete path | plain Python | ratio | agreement checked |
|---|---|---|---|---|
| mixed | 0.0163 ms | 0.000112 ms | **146×** | **CORRECTED: not 0.0.** The float32 round-trip makes bit-identity impossible; the true discrepancy against `sin(float(a!=b)+x)` is 1.5e-9 to 2.6e-8 on 4 cases. |
| language | 4.061 ms | 0.000496 ms | **8,187×** | **CORRECTED: 7/12, not 12/12** (this line first said 9/12; five episodes disagree — §62). This track's own `out/inproc.json` records `language.all_agree: false`; episode 2 disagrees. See FINDINGS section 39 for the cause — the frozen program is being evaluated on a stream the lesson audit re-drew. The *cost* comparison is unaffected: both arms ran the same program on the same inputs. |
| visual | 15,371 ms | 0.170 ms | **90,400×** | identical 19- and 16-rectangle sets on 3 held-out screens |
| computer *(program only)* | 13.23 ms | 0.000096 ms | **137,800×** | same verb and same written digit |
| computer *(complete step)* | 981.1 ms | 967.85 ms + 0.0001 ms | **1.014×** | — |

The last row is the one a funder should read twice.  For the agent, the program
is **1.3%** of the step; the other 98.7% is the operating system the agent is
acting on, and it is a cost a hand-written Python agent and a 100-billion-
parameter model would both pay in full.

### 4.2 A matched neural network — where one exists

Track 6 built matched-information baselines for the two fixtures only.  These
rows are **inherited** from `research/baselines/RESULTS.md` §3–§5 and were not
re-measured here:

| task | TCN | matched MLP | lookup table | what the MLP costs in quality |
|---|---|---|---|---|
| mixed | 0.0588 ms, error 2.6e-8 | 625-param MLP: 0.0514 ms torch / 0.0034 ms numpy | — | interpolation 5.2e-3, extrapolation RMSE 0.232 — five orders worse, and it fails the repo's own 1e-6 assertion |
| joint | 0.247 ms, 68,768 bits | 153-param MLP: 0.0378 ms, 4,896 bits, 4.00/4 | 32 bits, 0.00045 ms, 4.00/4 | indistinguishable — finite input domain |

**No matched neural baseline exists for visual, computer or language, and none
was built here.**  That is a real gap and it is the most important missing
reference in this report: the efficiency claim for the three capabilities that
actually work is currently TCN-versus-plain-Python only.  A fair one would have
to see exactly what the program sees and nothing more — for visual, the same
3,072-byte raster with `hierarchy` as the only supervision; for language, the
same 128-byte prompt with the `answer` probe; for computer, the same terminal
observation with `reference_byte` and `reference_action` — and be given the same
budget.  Until those exist, no claim in this repository about TCN being cheaper
*than a model* is supported for these three capabilities.

### 4.3 The exported `.pyz` under `python3 -I` — what a deployment ships

No torch, no numpy, no repository, no virtualenv.  `cold_start` is measured with
an **empty stdin** so no inference runs at all: it is process spawn + zipapp
import + artifact load + digest verification.  `inference` is amortized over a
batch on one warm process and **includes** parsing the JSON input line and
serializing the JSON reply, i.e. the transport §9 omitted.

| artifact | `.pyz` size | cold start | inference (amortized) | transport in / out | peak RSS |
|---|---|---|---|---|---|
| bare `python3 -I -c pass` | — | 17 ms | — | — | **9.4 MB** |
| mixed | 45 KB | **50.7 ms** | **0.035 ms** (n=2000) | 250 B / 177 B | **18.9 MB** |
| language | 1.49 MB | 90.6 ms | 7.64 ms (n=20) | 17.3 KB / 78 B | 23.8 MB |
| computer | 11.6 MB | 409.7 ms | 48.6 ms (n=20) | 609 KB / 118 KB | 60.4 MB |
| visual | 117.7 MB | **5,179 ms** | **66,557 ms** (n=1) | 410 KB / 1.4 KB | **377.5 MB** |

The `.pyz` is **slower than the in-process path everywhere**, and the gap grows
with declared type width: mixed 0.035 ms against 0.0163 ms (2.1×), language
7.64 ms against 4.06 ms (1.9×), computer 48.6 ms against 13.2 ms (3.7×), visual
66.6 s against 15.4 s (4.3×).  Two causes, both measured: 609 KB of JSON per
input has to be parsed and re-validated before computer's 46 operators run, and
§4.3.1's type-identity effect.  Transport is a first-class
cost of this deployment shape, and it scales with declared type width for
exactly the reason §5.1 gives.

### 4.3.1 The exported artifact is 4× slower than the same program in process

Visual's `.pyz` inference is **66.6 s** against **15.4 s** in process.  That gap
is not the interpreter version.  Rebuilding the artifact through `load_program`
and running it **under the repository's own Python 3.13**, on the identical
input, with the parsed rectangle set checked identical, reproduces it:

| arm | parse latency | rectangles |
|---|---|---|
| live objects | 16,219 ms | 19 |
| reloaded via `load_program` (same interpreter) | **66,127 ms** | 19, identical |
| slowdown | **4.08×** | |

`Type.from_dict` allocates a fresh object per occurrence, so two structurally
equal types are no longer the same object.  `Registry.exact` opens with
`tuple(v.type for v in args) != op.inputs`, which is an identity hit on the live
graph and a **deep structural compare of a 3,072-element tuple type** on the
reloaded one.  Interning types at load — one dictionary keyed by the type's own
canonical JSON — recovers the 4× and is the same fix as §5.1's shared type
table, seen from the runtime side.  `load_program` itself costs a further
3,285 ms on this artifact.

---

## 5. Size, honestly

`description_bits` is `8 × len(json.dumps(program.to_dict(), sort_keys=True))`.
Four numbers per artifact, so no single one can hide the others:

* **learned content** — log2 of the search space the frozen selection was drawn
  from, taken from each track's own recorded enumeration.  This is the
  information the search actually chose.
* **node / operator declarations** — the minified JSON of the program's node
  names, operator names, parameters and wiring: the program as structure.
* **`description_bits`** — what the repository currently reports.
* **gzip** — the same artifact deflated: what a deployment would actually carry.

| artifact | nodes (caller + module) | distinct operators | learned content | node+operator decl. | constants | `description_bits` | **gzip** | gzip ÷ `.pyz` |
|---|---|---|---|---|---|---|---|---|
| mixed | 4 | 4 | **6.6 bits** (96 programs) | 275 B | 0 B | 17,728 b = 2.2 KB | **451 B** | 100× |
| language | 84 + 5 | 9 | **28.8 bits** (4.76e8 programs) | 6.4 KB | 1.0 KB | 4,043,552 b = 505 KB | **6.3 KB** | 248× |
| computer | 23 | 14 | **21.7 bits** (7,480 × 448) | 1.7 KB | 2.2 KB | 33,971,936 b = 4.2 MB | **15.2 KB** | 802× |
| visual | 4 + 607 | 24 | **21.3 bits** (256 × 400 × 25) | 41.2 KB | 4.6 KB | 319,889,560 b = 40 MB | **149 KB** | 831× |

### 5.1 The 117.7 MB `.pyz`, decomposed

Asked directly: is `visual.pyz` big because the program contains that much
structure, or because the serialization is wasteful?  **It is serialization,
and the evidence is not close.**

| component of `visual.pyz` | bytes | share |
|---|---|---|
| total file | 123,424,313 | 100% |
| the tcn interpreter itself (`types.py`, `operators.py`, `graph.py`, loader) | 39,417 | 0.03% |
| `program.json`, stored **uncompressed** (`zipapp` default `compressed=False`) | 123,384,155 | 99.97% |
| — of which `json.dumps(indent=2)` whitespace | 88,045,763 | **71.3%** |
| — minified JSON | 35,338,392 | 28.6% |
| —— **repeated type declarations** | 35,223,872 | **99.68% of minified** |
| —— node + operator declarations (the program) | 41,150 | 0.12% of minified |
| —— constant payloads | 4,595 | 0.01% of minified |
| the same artifact, gzipped | **148,602** | **0.12%** |

### 5.1.1 The control that settles it: compression ratio rises with declared width

`gzip -9` over each whole `.pyz` (measured independently by the supervising
session; not re-measured here), beside this track's byte decomposition:

| artifact | declared observation width | `.pyz` | `gzip -9` of the `.pyz` | ratio | interpreter source inside the `.pyz` | minified-then-gzipped artifact JSON |
|---|---|---|---|---|---|---|
| mixed | 1 float | 0.043 MiB | 0.012 MiB | **3.6×** | 39,417 B | **451 B** |
| language | 129 elements | 1.49 MiB | 0.031 MiB | **48×** | 39,417 B | **6,289 B** |
| computer | 4,097 elements | 11.63 MiB | 0.069 MiB | **170×** | 39,417 B | **15,207 B** |
| visual | 3,072 elements | 117.71 MiB | 0.573 MiB | **206×** | 39,417 B | **148,602 B** |

**Mixed is the control and it behaves like ordinary JSON**: 3.6×, because its
45 KB file is 87% the tcn interpreter's own Python source (39,417 B, identical
in all four artifacts) and only 2 KB of program.  The ratio then rises with how
much *type declaration* the file contains — 48×, 170×, 206× — which is what
repeated type declarations look like and is not something distinct learned
content could do.  (The driver is width × number of mentions, not width alone:
computer declares the wider type, 4,097 against 3,072, but visual has 611 nodes
to computer's 23 and so names its wide type far more often.)  Combined with
§5.1's direct decomposition
(99.68% of visual's minified JSON is type declarations, 0.12% is the program),
hypothesis (2), *genuine program size*, is excluded.

The last column shows the two fixes compose: dropping `indent=2` **before**
compressing takes visual from 0.573 MiB to **149 KB**, another 4×.

The mechanism: `bytes_type(32,32)` is a `tuple` with 3,072 item types, and
`Type.to_dict` writes every one of them as its own JSON object — at each node's
`output`, at each candidate operator's `inputs` and `output`, and at each
constant's `type`.  The 585-node rect module names that type hundreds of times.
Nothing about the *program* grew; the type's *spelling* did.

The same pattern, weaker, everywhere else: type declarations are **99.8%** of
the computer artifact's minified JSON, **93.6%** of language's, and **52%** of
mixed's — the last only because a scalar type is short.

**What a deployment would actually carry: 149 KB** (gzip), or ~190 KB with the
interpreter, for a 611-node parse over 24 distinct operators.  Two independent
one-line fixes reach most of that without any format change — `save_program`
dropping `indent=2` (−71%) and `export_executable` passing
`zipapp.create_archive(..., compressed=True)` (−99.6%).  A shared type table
indexed by node would collapse the rest.

**Neither "117.7 MB" nor "a few KB" is an honest summary on its own.**  The
honest summary is: *611 typed nodes and 24 operators — 41 KB of program, 4.6 KB
of constants, 21.3 bits of learned selection — inside a 117.7 MB JSON envelope
that is 99.7% repeated type declarations and gzips to 149 KB.*

### 5.2 What `description_bits` is worth as a measure

It disagrees with every other measure of the same artifact by two to seven
orders of magnitude, in a direction set by how often the program's declared
types are spelled out rather than by anything the search found:

| artifact | `description_bits` ÷ gzip bits | `description_bits` ÷ learned bits |
|---|---|---|
| mixed | 4.9× | 2,700× |
| language | 80× | 140,000× |
| computer | 279× | 1,570,000× |
| visual | 269× | 15,000,000× |

FINDINGS §3 records this as refuted for the joint fixture at 2,150×.  On the
capabilities that actually work it is **1.5e6×** and **1.5e7×**.  The number
should not be quoted as a size at all.

---

## 6. Memory and startup

Measured on the child process with `/usr/bin/time -v`, under `python3 -I`.

| | bare interpreter | mixed | language | computer | visual |
|---|---|---|---|---|---|
| peak RSS | 9.4 MB | 18.9 MB | 23.8 MB | 60.4 MB | 377.5 MB |
| cold start | 17 ms | 50.7 ms | 90.6 ms | 409.7 ms | 5,179 ms |
| accelerator | none | none | none | none | none |
| runtime dependencies | — | **none** | **none** | **none** | **none** |

**The dependency-free claim survives, and it is the strongest thing in this
report.**  All four artifacts run to correct output on a stock `python3 -I`
with no torch, no numpy and no repository present.  The mixed program is a
**45 KB file that answers in 0.035 ms inside 19 MB of RSS** — 9.5 MB of which is
the interpreter itself, so the artifact's own footprint is under 10 MB.

Memory and startup track the artifact's *serialized* size, so §5.1's fixes
apply here directly: the computer artifact's 60 MB RSS and 410 ms cold start are
the cost of parsing 12 MB of JSON that gzips to 15 KB.

The in-process (repo `.venv`, torch imported) peak RSS for the parse is 430 MB;
essentially all of that is torch and the search-time scaffolding, and none of it
is on the deployment path.

---

## 7. What these numbers do NOT show

Stated at the same volume as the results, because the credibility of the
efficiency claim depends entirely on its scope.

1. **These are small, specialised programs.**  The parse is 611 typed nodes over
   24 operators on a **32×32** synthetic screen from `generators/gui`, with a
   32-colour palette, axis-aligned rectangles, flat fills and no anti-aliasing.
   It is **not** a vision system.  It has never seen a photograph, a real
   screenshot, a font, a gradient, a shadow, or an occluded widget.  Nothing here
   licenses a claim about screenshots of real software.
2. **The parse is not even complete on its own benchmark.**  Rectangles are
   215/215, but parent links are **173/215**, the root is *supplied*, and the
   comparison is rectangle-not-id.  All 42 wrong links are colour-key collisions
   (`research/visual-ladder/RESULTS.md` §0).  A good latency figure for a parse
   whose links are 80% right is not a good latency figure for a parse.
3. **A 46-operator agent step is not computer use.**  The computer artifact
   decides between three action templates on a one-line file.  It reaches 10/10
   on held-out documents of that exact shape.  It is not a general
   computer-using agent and nothing here measures one.
4. **The efficiency comparison is against plain Python, not against a model.**
   For visual, computer and language there is no matched neural baseline in this
   repository, so "cheaper than a billion-parameter model" is **not measured
   here for any of them**.  What is measured is that all three are *far more
   expensive than the few lines of Python that compute the same function*.
5. **Latency and quality are separate claims.**  Every artifact was measured on
   the inputs its own track validated it on.  A fast wrong answer is not
   measured anywhere in this document.
6. **The `.pyz` path was exercised, not stress-tested.**  Batch sizes are 20–2000
   for the small artifacts and 1 for visual; there is no concurrency, no long-
   running-process measurement, and no adversarial input.
7. **The host is shared.**  Absolute milliseconds will differ on other hardware.
   Operator-application counts, node counts, byte counts and ratios will not.

---

## 8. Reproduce

```bash
.venv/bin/python research/inference-cost/inproc.py        # mixed + language, in process
.venv/bin/python research/inference-cost/visual.py        # parse path + plain-Python ref + the caching fix (~4 min)
.venv/bin/python research/inference-cost/computer.py      # live-kernel agent step (~1 min)
.venv/bin/python research/inference-cost/refs.py          # plain-Python agent reference
.venv/bin/python research/inference-cost/pyz.py           # export + python3 -I deployment rows
.venv/bin/python research/inference-cost/sizes.py         # .pyz size decomposition
.venv/bin/python research/inference-cost/profile_path.py  # where execution time goes (~2 min)
.venv/bin/python research/inference-cost/reload_cost.py   # live objects vs load_program (~3 min)
```

`out/*.json` holds every figure above (the parse's are in `out/parse.json`).
`out/*.pyz` (131 MB together) and `out/visual_reload.json` (118 MB) are
gitignored; regenerate them with `pyz.py` and `reload_cost.py` before running
`sizes.py`.

| file | what it measures |
|---|---|
| `harness.py` | operator-application counter, timing, size report |
| `inproc.py` | mixed and language complete paths, plain-Python references |
| `visual.py` | parse complete path, plain-Python reference, the two-cache experiment |
| `computer.py` | live-kernel agent step, split into environment and program |
| `refs.py` | plain-Python computer-agent reference |
| `pyz.py` | export, cold start, transport, amortized inference, peak RSS under `python3 -I` |
| `sizes.py` | exact byte decomposition of each `.pyz` by role |
| `profile_path.py` | `cProfile` of one parse, grouped by role |
| `reload_cost.py` | the same program as live objects vs reloaded through `load_program` |

---

## 9. The claim, scoped so it can be defended

### What is measured

A crystallized TCN program is **ordinary software**.  All four artifacts run to
correct output on a stock `/usr/bin/python3 -I` — no torch, no numpy, no
repository, no virtualenv, no accelerator, no network:

* **mixed** — 4 typed nodes, 4 operators, **6.6 bits** of learned selection, a
  **45 KB** file, **0.035 ms** per answer including JSON transport, **50.7 ms**
  cold start, **18.9 MB** peak RSS (9.4 MB of which is CPython itself).
* **language** — 89 typed nodes, 9 operators, **28.8 bits** learned, **6.3 KB**
  of compressed content, **7.6 ms** per answer, **90.6 ms** cold start,
  **23.8 MB** peak RSS, 1.000 accuracy at string lengths never trained on.
* **computer** — 23 typed nodes, 14 operators, **21.7 bits** learned, **15 KB**
  compressed, **13.2 ms** of program per agent step against **968 ms** of
  operating system, **60.4 MB** peak RSS, 10/10 held out.
* **visual** — 611 typed nodes, 24 operators, **21.3 bits** learned, **149 KB**
  compressed, **15.4 s** per 32×32 screen, **377 MB** peak RSS from a 117.7 MB
  JSON envelope that is 99.7% repeated type declarations.

### What is not measured

That any of this is cheaper than a neural network at the same task.  Track 6
built matched baselines for the two toy fixtures only; for the parse, the agent
and the language program **no matched neural baseline exists in this repository
and none was built here**.  Against the reference that *was* measured — the same
function hand-written in plain Python, checked to produce identical output —
every artifact is **146× to 90,400× slower**, and 97.7% of that is the value
layer re-encoding whole observations at every graph edge.

### The sentence that can be defended in front of a funder

> *After crystallization the artifact is ordinary software: a few hundred typed
> nodes and tens of kilobytes of program, running on a stock Python interpreter
> with no framework, no accelerator and tens of megabytes of RAM.  It is not yet
> fast — today it costs two to five orders of magnitude more than the same
> function written directly in Python — and the measured causes are an uncached
> value decode, a re-validated carrier and a JSON type spelling, not anything
> about the method.*

Anything shorter than that overclaims.  In particular, **"tasks people assume
require a billion-parameter model run on a light desktop" is not established by
this report**: the deployment footprint claim is measured and survives, the
*efficiency* claim is measured and currently fails against plain Python, and the
*capability* claim is scoped by §7 to four small, synthetic, specialised tasks.

### What would have to change, in cost order

| fix | measured effect | where |
|---|---|---|
| memoize `Value.decoded` | 2.3× faster execution, outputs bit-identical | §3.1 |
| skip revalidating interpreter-produced carriers | 3.4× cumulative | §3.1 |
| intern `Type` objects at `load_program` | 4.1× on the exported path | §4.3.1 |
| `save_program` without `indent=2` | −71% of artifact bytes | §5.1 |
| `zipapp.create_archive(compressed=True)` | −99.6% of `.pyz` bytes | §5.1 |
| a shared type table keyed by canonical type | collapses the remaining 99.7% | §5.1 |
| build a matched neural baseline for visual / computer / language | closes the one reference this report is missing | §4.2 |

None of these is research.  All of them are engineering, and the first three are
measured here to work.
