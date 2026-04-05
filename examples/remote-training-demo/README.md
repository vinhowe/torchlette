# Remote Training Demo

Trains a tiny char-level transformer from the browser, with every
forward/backward/optimizer plan shipped as JSON over WebSocket to a Node
server that runs the computation on its CPU backend.

## What this proves

The browser client never executes a single kernel. It just builds Torchlette
autograd graphs. Every `force()`/`loss.item()`/`loss.backward()` turns into
a plan serialized via `src/remote/serialize.ts` and shipped over WS. The
server runs `executePlanSequential` and returns opaque handles the client
can reference in subsequent plans.

Total client-side compute: none. Total server-side code: pure Torchlette,
no modifications.

## Architecture

```
┌──────────── browser (client.ts) ────────────┐
│  Torchlette + autograd                       │
│  ↓ api.matmul(), loss.backward(), etc.       │
│  ↓ builds LazyIRNode plans                   │
│  RemoteRuntimeEngine                         │
│    ↓ intercepts force/force*/cpu/item        │
│    ↓ serializePlan → JSON                    │
│    ↓ WebSocket                               │
├──────────── server.ts (Node) ────────────────┤
│  WebSocketServer + handle registry           │
│    ↓ deserializePlan                         │
│  executePlanSequential(plan, cpuBackend)     │
│    ↓ returns HandleRef for each output       │
└──────────────────────────────────────────────┘
```

Per session, the server keeps a `Map<HandleRef, StorageHandle>`. Client
maintains the inverse `Map<localStorageId, HandleRef>` so it can translate
materialized refs when serializing subsequent plans. At the end of each
training step, the client calls `markStep(params)` which releases every
handle not bound to a parameter tensor.

## Files

```
examples/remote-training-demo/
├── server.ts            Node WebSocket server + static HTTP
├── smoke-test.ts        Minimal end-to-end (upload → execute → download)
├── integration-test.ts  Toy XOR MLP training via remote
├── transformer-test.ts  Tiny transformer training via remote
├── README.md            (this file)
└── client/
    ├── index.html       UI: buttons, loss chart, sample box, log
    ├── client.js        (bundled by esbuild; do not edit)
    ├── client.ts        Browser entry point
    ├── engine.ts        RemoteRuntimeEngine (the patch)
    ├── transport.ts     WebSocket RPC client
    ├── model.ts         Tiny transformer: 2L, D=32, 4 heads
    └── train.ts         Training step + data pipeline
```

## Run

```sh
# 1. Build the browser bundle
npx esbuild examples/remote-training-demo/client/client.ts \
  --bundle --format=esm --platform=browser \
  --outfile=examples/remote-training-demo/client/client.js \
  --external:webgpu --external:@roamhq/wrtc \
  --external:node-datachannel --external:peerjs --external:peer \
  --target=es2022

# 2. Start the server (pick a free port)
npx tsx examples/remote-training-demo/server.ts --port 9882
# Uses WebGPU by default (via Dawn in Node). Pass --cpu to force CPU.

# 3. Open http://localhost:9882 in a browser (or over SSH port-forward).
#    Click "Start training" and watch the loss curve.
```

## Sanity tests (headless, Node)

All should pass with the server auto-launched in a subprocess:

```sh
npx tsx examples/remote-training-demo/smoke-test.ts
npx tsx examples/remote-training-demo/integration-test.ts
npx tsx examples/remote-training-demo/transformer-test.ts
```

The `transformer-test` builds the exact same transformer the browser does
and runs 10 steps of training. Verifies loss drops and handles are cleaned
up by `markStep`.

## Protocol

Wire format: `src/remote/wire.ts` (types), `src/remote/serialize.ts`
(functions). RPC envelope: `src/remote/rpc.ts`.

Messages (request/response over single WebSocket):

| method | request | response |
|---|---|---|
| `execute` | text JSON | text JSON |
| `upload` | binary frame | text JSON |
| `download` | text JSON | binary frame |
| `readScalar` | text JSON | text JSON |
| `release` | text JSON | text JSON |

Binary frame layout (little-endian):

```
[4B id][1B dtype][1B rank][2B pad][rank × 4B shape][raw bytes]
```

dtype enum: 0=f32, 1=f16, 2=i32, 3=u32, 4=bool. See
`src/remote/binary-frame.ts`.

The `handlesReleased` counter on `RemoteEngine.stats` tracks how many have
been explicitly released back to the server across the run. Expect
~(plan size − param count) per training step.

## Known limitations of this demo

- **Tiny transformer.** 2 layers × D=32 × 4 heads × 30K params. On this
  model size, WebGPU kernel-dispatch overhead dominates matmul cost so
  CPU and WebGPU run at roughly the same ~100-140ms/step. The wire
  format is backend-agnostic; bigger models show GPU winning. The tiny
  model is intentional — it keeps the per-step latency snappy for the
  demo and validates the protocol end-to-end.
- **Single session per connection.** No auth, no multi-tenancy.
- **No auth / no multi-tenancy.** One session per WebSocket connection.
- **No snapshot/restore yet.** Checkpointing would be `snapshot(name, [handles])`
  that dumps to server disk without touching the wire, plus `restore(name)`.
- **No fused kernels used.** The CPU backend doesn't have fused attention,
  layernorm, etc. On WebGPU server-side, fused kernels work transparently —
  they show up as single nodes in the plan.
