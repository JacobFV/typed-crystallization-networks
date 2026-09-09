# Naming crystallized programs

There will be many of these. The goal is that a name never has to be extended
with another suffix — anything that varies goes in a path or in metadata, and
the name identifies a *capability family*, not one artifact.

## The one fact that forces the scheme

A module hardened at one observation width **cannot** be registered against
another: its input type names the observation, so `map` raises a signature
mismatch (FINDINGS section 30). Width is part of type identity, not a
decoration. The same is true of capacity, palette, and any declared refinement.

If those become suffixes you get `...-r8-p32-c64-v2-unique-pruned` and nobody can
tell which axis matters. So they become **paths and fields**, and the name stays
short.

## Three names, three scopes

**1. Library reference (inside the repo).** Already implemented in
`tcn/library.py` and unchanged:

```
<domain>.<capability>@<version>
logic.xor@1        gui.corner@2        geometry.foreground@1
```

Version increments on relearning, dependents are marked stale, and the digest is
the true identity. Never encode shape here.

**2. Hub repository — one per capability family.**

```
jacob-valdez/tcn-<domain>-<capability>

tcn-gui-parse          tcn-geometry-segment      tcn-logic-emit
tcn-gui-corner         tcn-geometry-edge         tcn-logic-interpret
tcn-computer-panel     tcn-language-balanced     tcn-control-pendulum
```

`<domain>` is the generator. `<capability>` is the verb the program performs.
Both are lowercase, single words, no shape, no version, no certificate.

**3. Path inside the repository — one directory per typed shape.**

```
r32/            resolution 32 raster
w3/             width-3 input
d1-8/           trained depths 1-2, valid 1-8
cap64/          declared set capacity 64
```

A shape directory holds `program.json`, `program.pyz`, `fixture.json` and
`card.json`. Two shapes of the same capability live side by side and are
genuinely different operators — which is exactly what the type algebra says.

## What goes in metadata, never in the name

`card.json` per shape, and the README table across shapes:

| field | why it is not a suffix |
|---|---|
| `certificate` — `unique` / `complete` / `none` | the strongest property here, and it changes when supervision changes |
| `conforming` | how many programs fit; 1 with `unique` is the claim |
| `space_size`, `evaluated`, `exhausted` | the search that produced it |
| `nodes`, `operator_cost`, `description_bits` | size; note `description_bits` is JSON length, not learned content |
| `held_out`, `baseline` | a number without its baseline is uninterpretable |
| `source_fingerprint` | pins replay to a code revision; a mismatch must fail loudly |
| `requires` | transitive module dependencies, charged once |
| `trained_on` / `valid_for` | e.g. trained at depths 1-2, valid 1-8 |

## Rules

1. **No shape, version, or certificate in a repo name.** They vary; the family
   does not.
2. **One directory per typed shape.** If two artifacts have different input
   types they are different operators and get different directories, never a
   suffix.
3. **A certificate is a field, not a badge.** `unique` is a property of the
   search that produced the program, and re-running with different supervision
   can change it.
4. **Every card carries its baseline.** A held-out number alone is not a result.
5. **`latest` is the highest library version that verifies**, i.e. whose fixture
   re-executes exactly. Never a moving pointer to an unverified artifact.
