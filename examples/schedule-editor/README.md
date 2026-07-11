# Schedule editor prototype

This is the islands-stratum prototype for torchlette's live model schedule
editor. It renders a detector-derived partition, applies only structural
`merge` and `split` moves, recomputes the production-compatible boundary hash,
and exports a future engine request in full-partition and move-list form.

## Run

```sh
pnpm install
pnpm test
pnpm check
pnpm build
pnpm preview
```

Regenerate the bundled ground truth from the worktree root:

```sh
TORCHLETTE_CPU_ONLY=1 pnpm exec tsx tools/t-schedule-dump.ts \
  examples/schedule-editor/public/data/gpt2-tiny-forward.json
```

The dump is a genuine lazy forward from torchlette's `GPT2`: two transformer
blocks, 64-wide embeddings, two attention heads, batch 1, sequence 32. It is a
**cold forward**, so deterministic parameter-initialization nodes are part of
the plan. This avoids model downloads and a backend execution dependency while
still exercising the real model, `buildMergedPlan`,
`segmentPlanForExecution`, `reifyPartition`, and `computePlanFingerprint`.

## Legality policy

The client enforces the parts of `docs/islands-design.md` §2 that can be proven
from the dump:

- exactly two emission-order-adjacent islands are required for merge;
- reduction islands are treated as opaque atoms and cannot merge;
- a sequential island may merge only when every member op is marked fusible by
  the production op registry copied into the client gate;
- all merged member output shapes must be identical;
- every interior member boundary (`1 <= cut < members.length`) may split;
- split preserves the source kind on both halves; inverse moves retain exact
  pre-move kinds, so undo/redo restores structure rather than a snapshot.

The shipped partition has `sequential | fused | reduction` kinds, while the
design text's earlier legality discussion uses `elementwise | authored`.
Conservatively, this prototype treats `reduction` as authored/opaque, treats a
`fused` island as elementwise-capable, and proves a `sequential` island
elementwise-capable from its node ops before merging it.

The dump intentionally has no dependency edges, device binding limit, storage
sizes, checkpoint/WAR annotations, or authored-epilogue capabilities. The
client therefore does not claim to prove convexity, buffer-count legality,
device-keyed binding budgets, chunking, or atom epilogues. Emission adjacency is
used as the available convexity witness; unsupported authored/device-specific
merges are not offered. A live engine must revalidate every request against the
full graph and device, even if the client enabled the gesture.

## Boundary-hash proof

`src/lib/partition.ts` ports `partitionBoundaryHash` byte-for-byte: FNV-1a over
little-endian 32-bit words for island count, kind code, member count, and member
positions. `src/lib/partition.test.ts` checks a vector generated directly by
the production implementation:

```sh
pnpm exec tsx ../../tools/t-schedule-dump.ts --hash-vector
```

The production result is decimal `2962669663` / hex `0xb096c05f`.

## Sequence UI

The app follows Sequence UI's 12.5px instrument density, sharp corners, flat
surfaces, token colors, semantic type roles, container-owned stacks, and
light/dark theme wiring. `AppBar`, `ThemeProvider`, and `ThemeToggle` were
installed verbatim from their registry JSON items (including the transitive
`theme-provider` dependency). Purple is not an arbitrary new choice: it matches
the existing Sequence-style torchlette examples in this worktree.

See [contract.md](./contract.md) for the proposed engine message.

## Intra-island workbench

Selecting an island opens the P3 consumer-side ScheduleState workbench. The two
proposal fixtures and schema live under `public/data/schedule-states/` and
`public/data/schedule-state.schema.json`. The tiled matmul exposes a full loop,
staging, and role skeleton. The fused attention forward state is deliberately
authored/opaque, so macro structure is locked while its real BR=64, BC=32,
WG=64, vec4-oriented decorations remain editable.

The static performance tier updates from the state plus adapter limits and
documented fallback architecture/roofline constants. The measured tier declares
its RPC in [workbench-contract.md](./workbench-contract.md) and remains visibly
awaiting the engine. Consumer-side schema findings are recorded in
[workbench-findings.md](./workbench-findings.md).

## Editable NCD spike

The `NCD diagram` mode reframes the property workbench as a bidirectional
Neural Circuit Diagram term. Its semantic boxes, wires, axes, and columns stay
fixed while residency colors, group/stream partitions, and divisibility
superscripts are editable labels. Static assets live in `public/data/ncd/`.

The attention walkthrough replays the admitted online-softmax lemma, two legal
fusion recolorings, `g_q=64`, and `s_x=32`. H/M are read directly from the wire
residency and partition labels, and the loop/pseudocode pane is mechanically
projected from the same term. See [ncd-findings.md](./ncd-findings.md) for the
consumer-side failures and ambiguities exposed by making the notation editable.
