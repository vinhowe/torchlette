# Remote Training Protocol: Design Exploration

**Status:** design doc, no implementation
**Goal:** offload heavy training compute to a remote GPU server (thin client, fat
server) while keeping the developer experience close to local Torchlette.
**Use case:** browser or laptop runs `api.*` / `nn.*` code; an A100 box runs the
kernels. The user writes code that feels like local training.

This document picks *where* to draw the client/server line (§3, §5), and
sketches a protocol for the chosen boundary (§6). Transport is a secondary
question; §6.1 argues for WebSocket.

---

## 1. What "remote" already means in this repo

There is already distributed infrastructure, but it does something different:

- `src/distributed/diloco.ts`, `gossip.ts`, `e3m0.ts`, `outer-optimizer.ts`
  (1018 LoC) implement **Streaming DiLoCo** — peer-to-peer, each peer runs its
  own full training loop, peers gossip 4-bit compressed pseudo-gradients via
  WebRTC/PeerJS. This is **horizontal** scaling: N equal peers, no server.
- The transport (`gossip.ts`) assumes WebRTC data channels between browsers,
  E3M0-compressed payloads, and no coordinator.

What this design targets is the **orthogonal axis**: **vertical** offload. One
rich server, many thin clients, *unequal* roles. A client might not have a GPU
at all; the server does all the work.

Both models can coexist. A DiLoCo peer could itself be a thin client driving a
remote GPU, for example.

---

## 2. Torchlette's layers (the candidate seams)

Torchlette is layered — any layer could in principle be the remoting boundary.
The data that crosses a layer determines how chatty the protocol is and where
state lives.

```
┌───────────────────────────────────────────────────────────────┐
│  User code:  api.matmul(x, w); loss.backward(); opt.step()    │
├───────────────────────────────────────────────────────────────┤
│  Frontend Tensor + autograd  (src/frontend/)                  │  <- boundary C
│    - Tensor { gradNode, gradValue, requiresGrad }             │
│    - AutogradNode graph (built during forward)                │
├───────────────────────────────────────────────────────────────┤
│  RuntimeEngine  (src/runtime/engine.ts)                       │  <- boundary B
│    - lazy: every op creates LazyIRNode, nothing executes      │
│    - force() / forceAllPending() flushes graph → Plan → GPU   │
├───────────────────────────────────────────────────────────────┤
│  Executor + Compiler  (src/executor/, src/compiler/)          │
│    - fusion-detect, matmul-epilogue, graph-rewrites           │
│    - ExecutionPlan = { nodes: LazyIRNode[] }                  │
├───────────────────────────────────────────────────────────────┤
│  Backend ops  (src/backend/webgpu/)                           │  <- boundary A
│    - dispatchMatmul, fusedAttentionForward, adamStep, ...     │
│    - ~60 ops over BackendTensor handles + GPUBuffer pool      │
└───────────────────────────────────────────────────────────────┘
```

Boundaries **D** (training step) and **E** (job) wrap everything above.

### Key data types

- `LazyIRNode` (`src/graph/types.ts:74`): `{id, op, inputs: LazyRef[], shape,
  dtype, device, result?, payload?, isCheckpointBoundary?}` — JSON-safe minus
  `result` (a runtime `StorageHandle`).
- `LazyRef` (`src/graph/types.ts:97`): `{kind: "pending", node}` |
  `{kind: "materialized", storage}` | `{kind: "scalar", value, dtype}`.
- `ExecutionPlan` (`src/graph/types.ts:132`): just `{nodes: LazyIRNode[]}`.
- `StorageHandle` (`src/graph/types.ts:66`): `{id, device, backendTensor,
  baseStorageId?}` — `id` is process-local and *not* stable across restarts.
- Frontend `Tensor` (`src/frontend/tensor.ts:24`) wraps `RuntimeTensor` and adds
  autograd state (`gradNode`, `gradValue`, `requiresGrad`, `retainGrad`).

---

## 3. The five candidate boundaries

For each: what crosses, who holds state, and an honest assessment.

### A. Op-level (per-dispatch forwarding)

**Client:** full engine, full autograd, full compiler. Only the WebGPU backend
calls are forwarded.

**Wire format (per op):**
```ts
{ op: "matmul", inputs: [handle_7, handle_8], outShape: [256,768], outDtype: "f16" }
```

**Pros:**
- Trivial to implement. Swap the `Backend` interface (`src/backend/types.ts`)
  for a remote stub. ~20 op dispatch sites to wrap.
- Reuses every optimization above it (fusion, epilogue, checkpointing).
- Backend already has a clean async hook (`force()` is async).

**Cons:**
- **Fatally chatty.** GPT-2 Medium does ~741 fused nodes/step (see CLAUDE.md
  Baseline C), and each node is one dispatch. At ~1ms round-trip that's 741ms
  of serial latency per step — before kernel time.
- Defeats the shared-encoder optimization: the server can't pipeline passes if
  the client re-issues one-at-a-time.
- Pooling is broken across a network: `releaseParamsBuffer`, `sharedEncoderWriteSet`,
  `endSharedEncoder` are all coupled to a single process's lifecycle.

**Verdict:** reject. Only viable for debugging / correctness bridges.

### B. Runtime / Lazy-IR boundary (plan forwarding)

**Client:** autograd + tensor API, but the `RuntimeEngine` flushes *plans* to
the remote. Everything below `force()` runs there.

**Wire format (per force):**
```ts
{
  type: "execute_plan",
  plan: { nodes: [
    { id: 1, op: "matmul", inputs: [{kind:"materialized",id:42},
                                     {kind:"pending",node:0}],
      shape:[256,768], dtype:"f16" },
    ...
  ]},
  externalInputs: { 42: <server-side handle> }
}
→ { outputs: { nodeId → storageHandleId } }
```

**State:** server holds `StorageHandle` registry (id → GPU buffer). Client
holds lazy `LazyIRNode` graph with pending refs keyed by remote handle ids.

**Pros:**
- **One RPC per force / markStep**, not per op. Typical: one per training step
  (backward + optimizer are flushed together).
- Plans are already the unit of compiled work. `ExecutionPlan` is already
  `{nodes: LazyIRNode[]}`.
- Server keeps the shared encoder, buffer pool, fusion — client doesn't lose
  optimizations.
- The async model fits: `force()` is already `Promise<void>`.

**Cons:**
- **Payload is the whole IR graph every step.** 500-1000 nodes × a few hundred
  bytes = 100-500 KB per step. Fine on LAN, visible on WAN.
- Payload types (`SumOptions`, `GeluOptions`, `StridedScatterOptions`, ...)
  must be JSON-encodable. They already are today, but must be *kept* that way.
- Handle lifetime: the server must `destroy` what the client abandons. Needs a
  keep-alive or explicit `release_handles` RPC.
- Saved-for-backward tensors persist across steps — the client has to track
  which remote handles the autograd graph still needs.

**Verdict:** strong candidate. Cleanest architectural seam in the codebase.

### C. Frontend Tensor boundary (api.* forwarding)

**Client:** stub `Tensor` class forwarding every method call.

**Wire format:**
```ts
{ method: "matmul", args: [handle_7, handle_8], requiresGrad: true }
→ { result: handle_9, shape:[256,768], dtype:"f16", gradNode: <opaque id> }
```

**State:** server holds both storage *and* autograd graph. Client holds opaque
handles plus some shape/dtype metadata for error messages.

**Pros:**
- API-compatible for users. A `RemoteTorchlette` drop-in replacement.
- Autograd runs server-side, so saved-for-backward tensors don't need
  cross-network lifetime tracking.
- Backward is one call (`backward(loss_handle)`) — no graph traversal chatter.

**Cons:**
- **Every user op is an RPC.** A transformer forward is ~100-200 `api.*` calls.
  Better than (A) but much worse than (B).
- Duplicates work done by the lazy engine: the client serializes each op
  individually when it could batch.
- Control flow leaks: `if (loss.item() < threshold) ...` still needs readback
  but inline in the user loop.
- Introspection APIs (`tensor.shape`, `tensor.toArray()`, `tensor.dtype`) each
  need synchronous-looking semantics, forcing eager RPCs or a shadow metadata
  cache.

**Verdict:** viable but strictly dominated by (B) in the Torchlette design.
The lazy engine *is* the batching layer; (C) throws it away.

### D. Training-step boundary

**Client:** defines model architecture, provides batches, receives loss.
Server owns model weights, optimizer state, forward/backward/step.

**Wire format (per step):**
```ts
{ type: "train_step",
  batch: { input: Uint32Array, target: Uint32Array, shape:[bs,seq] },
  accumulate: false }
→ { loss: number, gradNormPreClip?: number, stepIdx: number }
```

**State:** server holds *everything*: model, optimizer, RNG, autograd. Client
holds only a session id.

**Pros:**
- **Minimum bandwidth.** Batch in, scalar out. For GPT-2 finetuning at
  seq=256: batch ≈ 1 KB, response ≈ 16 B.
- Matches Tinker exactly (`forward_backward`, `optim_step`).
- Session-scoped server state is easy to reason about — no handle book-keeping.
- Server can do anything internally (gradient accumulation, ZeRO, LoRA
  sharing, checkpointing) without client coordination.

**Cons:**
- Client does not define training dynamics — it picks from a *menu* of
  server-supported ops. New loss functions, custom schedulers, debugging
  hooks, all become server API changes.
- Losing "write PyTorch-y code locally, it runs on the server" feel. Can't
  just import `api` and `nn` and have it work.
- Sampling, interactive inspection (`tensor.toArray()` on an intermediate)
  become awkward RPCs.

**Verdict:** viable as a **product**. Limiting as a **protocol** — it forces
the client into a fixed API shape. See §5 on combining with (B).

### E. Job-level boundary

Client submits `{model: "gpt2", dataset: ..., config: {...}, maxSteps: 1000}`,
server runs everything, client polls for loss / checkpoints.

**Pros:** simplest client, easy to multi-tenant, easy to spot-resume.
**Cons:** zero interactivity. Not really a "remote training protocol" — it's a
batch job submission service. Can always be *built on top of* (B) or (D).

**Verdict:** out of scope for this exploration; belongs a layer up.

---

## 4. Cross-cutting concerns

These decisions affect every boundary choice:

### 4.1 Where do weights live?

**Server, by default.** The offload premise already commits to this — if the
client had room for the weights, it would have room for the activations too.

In (B), "server-side weights" is not a separate concept — it falls out of the
handle model. When the client calls `engine.randn([768, 768], "gpu")`, that
emits a plan node; the resulting storage is allocated on the server; the
client only ever sees the handle id. The client's `Module` (e.g. `nn.Linear`)
holds those handles as parameters, which **pins them** — they survive
`markStep()` cleanup because the client holds a reference.

No `download_weights` is required for training. It is one RPC away when the
user wants it (inspection, export, transfer to another service).

**Persistence is a different question** from location. See §6.3.

**Multi-client / shared base models**: a common base (`gpt2-base`) can be
loaded once server-side and exposed as **named read-only handles**
(`engine.mountModel("gpt2-base")` → `{wte, wpe, h[], lnf}`). Each client
owns its own LoRA adapters but shares the read-only base. This is the
Tinker LoRA-sharing story, in (B)-native terms, and costs no new protocol
primitives beyond "named server handles."

### 4.2 Async model

Torchlette is already lazy — nothing blocks until you call `force()`, `item()`,
or `toArray()`. This is **load-bearing for remote:** every op call locally is
cheap; network round-trips cluster at `force()` boundaries.

Any boundary choice must preserve this. Boundary (C) risks breaking it by
making every op potentially await.

### 4.3 Handle stability

`StorageHandle.id` is a monotonic integer assigned by the process. **Not**
stable across restarts, processes, or machines. A remote protocol must either:

- Use opaque server-issued handle strings (uuid or session-scoped integer),
  and translate client-side `id` → server-side handle at the boundary. Client
  keeps a `Map<localId, serverHandle>`.
- OR use content-addressed handles (hash of producing plan + input handles).
  Enables caching but complicates destroy semantics.

Recommend the simple option: **server issues handle ids, client stores them
in the `StorageHandle.id` field.** Backward compatible; nothing downstream of
the runtime cares where ids come from.

### 4.4 Autograd: client-side or server-side?

Boundary (B) puts autograd on the *client* — the client builds `AutogradNode`
graphs on the frontend, then when `backward()` runs it emits a backward plan
which flushes to the server. This works because autograd in Torchlette is
entirely frontend (`src/frontend/autograd.ts`, `src/frontend/custom-backward.ts`)
— it only sees `Tensor`s, never touches the backend.

Boundary (D) puts autograd on the *server* — client never touches it.

**Implication for (B):** saved-for-backward tensors are *remote handles*, held
alive by the client's autograd graph until `backward()` is called. Need an
explicit "keep these alive across markStep" protocol — Torchlette already has
this concept (see CLAUDE.md on step-scoped storage cleanup).

### 4.5 Bandwidth sanity check

For GPT-2 finetuning, seq=256, bs=8, f16:
- **Batch up**: 8 × 256 × 4B (int32) = 8 KB
- **Loss down**: 8 B
- **Plan (boundary B)**: ~500 nodes × 150B = 75 KB / step
- **Weights download** (on demand): 124M × 2B = ~250 MB
- **Gradients up** (if held client-side): same as weights, ~250 MB

Verdict: **training-loop bandwidth is trivial**. Weight/gradient sync is where
the bytes are. Handle it as a rare checkpoint, not a per-step concern.

### 4.6 Latency sanity check

For step-time ≈ 200ms (GPT-2 Medium, CLAUDE.md Baseline C):
- LAN round-trip: ~0.5ms → irrelevant
- WAN round-trip: ~20-100ms → **one** round-trip per step is acceptable; one
  per op (boundary A/C) is not
- Browser → WAN: +TLS, +fetch overhead, ~100-200ms best-case

---

## 5. Recommendation

**Boundary (B), single layer. No Tinker-style server API.**

Connecting to a remote engine returns a `RuntimeEngine`-shaped object. The
client's training code is **just Torchlette code** — `api.*`, `nn.*`, `optim.*`
— running against a remote backend. Nothing is pre-defined server-side
beyond the plan executor, handle registry, and persistence.

```ts
const engine = await connectRemote({ url: "wss://gpu.example.com/v1" });

const model = new GPT2({ engine });            // weights allocated remotely
const optimizer = new AdamW(model.parameters(), { lr: 3e-4 });

for (const batch of dataset) {
  await engine.beginStep();

  const logits = model(batch.input);
  const loss = api.crossEntropy(logits, batch.target);
  await loss.backward();
  await optimizer.step();
  optimizer.zeroGrad();

  console.log(await loss.item());              // one readback RPC
  await engine.markStep();                     // flush + cleanup
}

await engine.snapshot("run-100", model.parameters());   // server-local
```

**Why single-layer, no (D):**

- You already get `forwardBackward` / `optimStep` / `sample` — they're
  called `loss.backward()` / `optimizer.step()` / the sampling loop you
  already wrote. Building a Tinker-like surface adds a menu the user must
  order from, for no added capability.
- New losses, custom schedulers, debug hooks, gradient introspection, mixed
  LoRA + full-FT — all work the same way they do locally. No server version
  bumps required.
- The protocol is *one concept*: plans travel, handles persist. Easier to
  specify, version, and reason about.
- Torchlette's lazy IR *is* the IPC payload. The cleanest seam wins.

**Trade-off accepted:** the client needs the Torchlette JS bundle (~100KB
minified-gzipped, rough guess). If that's a concern for embedded clients,
a thin-client Tinker-style RPC can be added **on top of** (B) later
without changing the core protocol. Don't build it preemptively.

**Sampling latency is the one honest wart.** Autoregressive generation does
~1 RPC per token (one `forward` + one `item()` for the argmax). On a 50ms
WAN link, that's 50 tok/s ceiling from latency alone. Two mitigations, both
deferrable: (a) server-side control flow in plans (a `while` node; big
spec change, not first-release), or (b) a narrow `engine.generate(...)`
helper that pushes the loop into a single server-side task (small spec
addition, re-introduces one D-flavored endpoint). Note and move on.

---

## 6. Protocol sketch (recommended path)

The protocol is **RPC over a persistent WebSocket connection**. One socket
per session; the session lives exactly as long as the socket. Closing the
socket frees all non-snapshotted server state.

### 6.1 Transport: WebSocket

**Why not REST / HTTP fetch:**
- Plans are opaque payloads, not resources — no URL hierarchy maps
- Session affinity is automatic with a persistent connection (no sticky-
  session tricks, no cookie management)
- Binary frames carry blobs without base64's 33% overhead
- Server-push for streaming (progress stats, generate-token streams)
- Works in every browser, Node, Bun, Deno without extra deps

**Why not WebRTC (as `src/distributed/gossip.ts` uses):** peer-to-peer
makes no sense with an unequal client/server relationship, and needs a
signaling server anyway.

**If you want typed schemas / codegen:** [Connect-RPC](https://connectrpc.com)
over HTTP/2 is the alternative. Same transport tradeoffs, but it forces you
into a protobuf/protovalidate workflow up front. Worth it for stable public
APIs; probably overkill for v0.

**Framing (first cut):** length-prefixed frames. Each frame is either JSON
(control) or binary (blob data), tagged by the first byte. Request/response
correlation via request id in the JSON header.

### 6.2 RPC methods

All methods are bidirectional: client→server requests and server→client
notifications (for streaming) use the same framing.

**Lifecycle**
```
hello                           → { sessionId, version, capabilities }
bye
```

**Plan execution (the hot path)**
```
execute { plan: SerializedPlan, externalInputs?: Record<NodeId, HandleRef> }
  → { outputs: Record<NodeId, HandleRef>, stats?: {...} }

release { handles: HandleRef[] }
  → { releasedCount: number }
```

`SerializedPlan` is `ExecutionPlan` with `StorageHandle` replaced by
`HandleRef` (session-scoped string). The client tracks which node outputs
it keeps references to, releases the rest at `markStep()`.

**Data I/O**
```
upload   { shape, dtype } + binary frame   → { handle: HandleRef }
download { handle: HandleRef }             → binary frame
```

Upload/download move **actual tensor bytes** between client and server.
These are the expensive ops — avoid them in the training loop.

**Server-local persistence (snapshot/restore)**
```
snapshot { name: string, handles: Record<string, HandleRef> }
  → { snapshotId, byteSize }

restore { name: string }
  → { handles: Record<string, HandleRef> }

listSnapshots { prefix?: string }
  → { snapshots: [{ name, createdAt, byteSize }] }

deleteSnapshot { name: string }
```

Snapshots **do not travel** across the network. They write handle storage
to the server's disk (or S3, etc.) under a name. Restore reads them back
and returns fresh session-scoped handles pointing at the rehydrated data.

The training loop checkpoints by calling `snapshot("run-100", {...})` —
no 1.4GB upload. Only `download` moves bytes client-ward, and only when
the user explicitly wants the bytes (e.g. exporting trained LoRA weights
to commit to git).

**Introspection**
```
stats                                   → { gpuMem, activeHandles, ... }
readScalar { handle: HandleRef }        → { value: number }
```

`readScalar` is what `tensor.item()` compiles to — avoids the overhead of
the full `download` path for 1-float reads.

### 6.3 Session & handle semantics

- **Handles are session-scoped strings.** Server issues them; client stores
  them as the `id` field of `StorageHandle` (widen from `number` to
  `number | string`, or use string always).
- **Handles are reference-counted on the server**, but the count is
  *client-driven*: the client is the authority on what's still referenced.
  The server frees a handle when `release` is called OR the session ends.
- **`markStep` cleanup:** the client's existing step-scoped cleanup (see
  CLAUDE.md) produces the set of handles to release. This becomes one
  `release` RPC per `markStep`, batched.
- **Reconnection:** socket drop = session death by default. Resumable
  sessions are an open question (§8.9).

### 6.4 Serialization schema gotchas

- `LazyIRNode.payload: unknown` is the biggest risk. Every op's payload
  type needs an explicit schema. Today they are typed interfaces in
  `src/graph/types.ts` and op-option files, but nothing enforces
  JSON-safety. Proposed: a `SerializablePayload` marker type, checked in
  CI via a type-level test that forbids `Function`, `Symbol`, class
  instances in payload unions.
- `LazyRef { kind: "materialized", storage: StorageHandle }` — only the
  `id` field travels. The server materializes the rest from its registry.
- `scalar` refs travel as-is.
- `isCheckpointBoundary` and `module` (profiling) travel as-is.
- `result?: StorageHandle` **does not travel** — it's the executor's output,
  allocated server-side during `execute`. Strip it before serializing.
- Plans may contain `externalInputs` only by **handle reference**, never by
  inline tensor data. Upload first, then reference. This keeps the `execute`
  payload small and makes big uploads explicit.

---

## 7. Comparison: Tinker vs this design vs DiLoCo

| | **Tinker** | **This (B)** | **DiLoCo (existing)** |
|---|---|---|---|
| Topology | 1 server, N clients | 1 server, N clients | N peers, no server |
| Weights location | server | server (handle-pinned) | each peer |
| Client compute | none (API client) | autograd + plan build | full training |
| Primary primitives | `forward_backward`, `sample`, `optim_step` | `execute(plan)`, `snapshot`, `release` | gossip grads |
| Granularity | training step | lazy plan | outer step |
| Extensibility | cookbook (above API) | just write more Torchlette code | change the algorithm |
| Transport | HTTPS (assumed) | WebSocket | WebRTC |

**Key delta vs Tinker:** Tinker exposes a *menu* of primitives
(`forward_backward`, `optim_step`, `sample`, ...). This design exposes a
*substrate*: any Torchlette op sequence is a valid RPC payload. You don't
add an API method to support a new loss — you just write the loss in the
client and it becomes plan nodes. The cost: client bundle size (needs the
Torchlette autograd + plan builder). The benefit: zero server-API churn as
the framework evolves.

**Orthogonality to DiLoCo:** a DiLoCo peer could be a thin remote-training
client (thin DiLoCo peer → fat compute server → peer-to-peer gradient
gossip between *servers*). The two can compose.

---

## 8. Risks & open questions

1. **Payload JSON-safety** is a landmine. A typed `SerializedPlan` schema and
   a `toJSON`/`fromJSON` discipline per payload type is needed before this
   protocol can be stable. Low cost up front, annoying to retrofit.
2. **Autograd saved-tensor lifetime across the wire.** Client-side autograd
   holds references to server-side handles; forward-saved tensors must
   survive `markStep` until `backward()` runs. Torchlette already has the
   step-scoped storage concept (see CLAUDE.md §"Step-Scoped Storage
   Cleanup") — needs to extend to the remote case without changes.
3. **Debugging when compute is remote.** No more local `mul_` RuntimeTensor
   investigation (CLAUDE.md §"Open Performance Targets") — good. But a
   hang becomes "what is the server doing?" — bad. Needs a first-class
   `stats` stream and server-side profiling hooks exposed via RPC.
4. **Multi-tenant isolation.** If the server hosts multiple sessions on one
   GPU, the buffer pool / shared encoder invariants (CLAUDE.md) must hold
   *per session*. The current shared encoder is process-global. Options:
   (a) one worker process per session (simple, wastes VRAM on model sharing),
   (b) make shared encoder session-aware (harder, violates current
   invariants), or (c) model-sharing via mounted read-only handles + one
   session per LoRA adapter (Tinker's answer; compatible with (a)).
5. **Snapshot storage format.** Needs a binary format. Options: safetensors
   (already used for GPT-2 loading via `src/frontend/weights.ts`), raw tensor
   + JSON manifest, or the snapshot is a single binary blob with an
   in-band index. Safetensors is the obvious default.
6. **Sampling latency** (per §5) — 1 RPC per token. Acceptable for
   evaluation; unacceptable for interactive generation. Either add a
   server-side `generate` helper (narrow D-style escape hatch) or
   eventually add control-flow nodes to plans.
7. **Failure recovery mid-step.** Server crash between plan execution and
   client seeing the result: client's autograd graph references handles
   the server no longer has. Plausible mitigation: `execute` is
   **idempotent** on `(sessionId, planHash)` — retry-safe — but this
   requires content-addressing the plan, which costs a hash per call.
   Alternative: no retry, session is dead, resume from last snapshot.
8. **Session resumability.** Socket drop = session death is brutal for
   long training runs over flaky networks. Alternative: sessions survive
   disconnect for N minutes, handles stay pinned, client reconnects with
   `hello { resumeSessionId: ... }`. Adds a GC timer and one concept.
   Probably worth it in v1.
9. **Client bundle size.** Client needs Torchlette's frontend + autograd +
   plan builder (no backend). Ballpark 60-100KB min+gz. Fine for browser
   ML apps; possibly heavy for embedded / non-JS clients. Not a v1 concern.
10. **Non-JS clients.** A Python client would need to either (a) re-implement
    plan building in Python, (b) embed the JS client via a JS runtime, or
    (c) expose a narrower server-side API that Python speaks directly (this
    is when a Tinker-style surface re-enters the picture — as the
    *language-boundary* layer, not the default).

---

## 9. What to build first (if this path is picked)

None of this is code yet. Order of operations:

1. **Define `SerializedPlan`** — a discriminated union over every
   `LazyOpCode`, with a payload type per op, in (say) `src/remote/wire.ts`.
   Enforce JSON-safety with a type-level test. This is the load-bearing
   decision: get the wire format right and everything else is transport.
2. **In-process round-trip test.** Build a plan locally, serialize → JSON
   → deserialize → execute against a *fresh* `RuntimeEngine`, compare
   outputs bit-exact to local execution. Validates format before touching
   any network. If this works, the rest is plumbing.
3. **Handle registry + `execute`/`release`** on a same-process server
   stub. Still no network. Verifies handle semantics, saved-for-backward
   lifetime, and `markStep` cleanup integration.
4. **`upload` / `download` / `readScalar`** — move bytes through the
   stub. Now a full training step works end-to-end in-process.
5. **`snapshot` / `restore`** — server-local persistence. Safetensors
   format. Verify resume-from-checkpoint produces identical next-step
   losses.
6. **WebSocket transport layer.** Wrap the stub in a WS server and
   client. At this point the public API is just a URL change.
7. **`RemoteRuntimeEngine`** — client-side class implementing the same
   surface as `RuntimeEngine` but routing `force()` through the socket.
   Ideally shares code with the local engine via an internal interface.

Step 2 is the load-bearing one. If plan round-tripping works locally,
the network is just transport.

**Out of scope for v0:** multi-tenant isolation (use one process per
session), session resumability (socket drop = session death), server-side
`generate` helper, non-JS clients.
