# Session handoffs

One file per working session, named `YYYY-MM-DD-session-N-<topic>.md`.

**These are append-only.** A session writes its own file and does not edit
earlier ones — an archived handoff is a point-in-time record of what was believed
*then*, and its errors are part of the record. Where a later session overturned
something, the earlier file carries a note at the top pointing forward; its body
is left as written.

The living entry point is [`../../HANDOFF.md`](../../HANDOFF.md), which is an
index and is updated in place. The authoritative measurement record is
[`../../research/FINDINGS.md`](../../research/FINDINGS.md).

| session | date | what it covered |
|---|---|---|
| [1 — overnight](2026-09-09-session-1-overnight.md) | 2026-09-08/09 | Substrate fixes, composition, thirteen capabilities, four instrumentation faults |
| [2 — efficiency and baselines](2026-09-09-session-2-efficiency-and-baselines.md) | 2026-09-09 | Compiled runtime (§42), program length (§41), matched neural baselines (§43), earned abstraction (§44), two shipped corrections |
