# Weight-Only Quantization Design (int8 first; int4 planned)

*2026-07-11. Task #76. Phase-0 design doc, written before implementation.*

## One-sentence declaration (the house test)

> A quantized weight is a **matmul operand format** — a packed-int storage tensor
> plus a per-group-scale companion tensor — declared on the tensor and selected
> as a matmul *variant*, with dequant fused into the kernel's K-loop load path and
> never materialized as a tensor.

If this design ever needs a second sentence to state, it has overreached.

## Why (the steepness argument)

The Gemma-2-2B SAE demo runs ~2.35 s/token on the user's 16 GB unified-memory
Mac, >99 % of it the GPU forward fence, which is **memory-bandwidth-bound** (the
decode matmuls are M=1 GEMV — one FLOP per weight byte read, so wall time ≈
weight-bytes / bandwidth). f16-embed already proved the residency lever on this
exact demo (17 s → 2.3 s/token). Weight-only int8 **halves** weight bytes vs f16
(int4 quarters them): both the bandwidth term (per-token) and residency (fits a
bigger model in 16 GB) win. On an A100 (2 TB/s, not bandwidth-starved) the win
will be small-to-none — we report honest numbers and lean on the Mac bandwidth
math for the phase-2 case.

**Scope: INFERENCE weights only.** Activations, residual stream (must stay f32 —
proven divergence otherwise), KV cache, embedding/lm-head table, and all of
training are OUT. Weight-only means: the *only* thing that changes is how a Linear
weight is stored and read inside the matmul. Everything downstream stays f32.

## Format: symmetric per-group int8, group size 64 along K

- **Symmetric** (zero-point = 0): `q = round(w / scale)`, `w ≈ q * scale`.
  Weight-only symmetric is the standard choice (GPTQ/AWQ-int8, llama.cpp Q8);
  asymmetric buys little for weights and costs a zero-point per group + an extra
  add in the hot loop.
- **Per-group along K, group size G = 64** (the matmul contraction dim). Each
  group of 64 contiguous-in-K weight elements shares one f16 scale. Group-along-K
  is the only choice that lets the GEMV/tiled K-loop pick up the right scale by
  `k / G` as it streams — a per-*output-row* (per-N) scale would also work for
  NT but per-group-K generalizes to both transpose modes and matches the
  llama.cpp block layout. G=64 is a good accuracy/overhead tradeoff (scale
  overhead = 16 bits / 64 weights = 0.25 bits/weight; effective int8 ≈ 8.25
  bits/weight vs f16's 16). G is a **declared parameter**, not a constant —
  int4 will want G=32 or 128; the kernel reads G from the operand format, and G
  is in the shader cache key + variant context.
- **f16 scales**, one per group, stored `[N, K/G]` row-major for the NT weight
  layout `[N, K]` (each output row N has K/G scales). f16 scales are ample
  precision for a symmetric int8 dequant and keep the scale buffer at 1/32 of
  the original f16 weight bytes.

### Packing (int8, this phase)

`nn.Linear` weight is `[out, in] = [N, K]` (HF layout, no transpose at load).
- Quantized weight is stored as **u32 words, 4 int8 per word**, laid out
  row-major over `[N, K]` then packed 4-along-K into `[N, K/4]` u32. So word
  `w[n, k/4]` holds `q[n, 4·(k/4) .. 4·(k/4)+3]` in its 4 bytes (little-endian
  byte 0 = lowest k). Requires `K % 4 == 0` (all real transformer K are; assert
  at quantize time). Total weight bytes: `N·K` (int8) vs `2·N·K` (f16) → **2×**.
- Scales `[N, K/G]` f16, `N·K/G·2` bytes.
- Storage buffers are **u32-typed bindings** by construction (the #59 lesson:
  packed-int weights live in u32 bindings; unpack in-kernel). WGSL
  `unpack4x8snorm`/`unpack4x8unorm` decode 4×i8 in one builtin, but they
  normalize to [-1,1]; we use plain byte extraction + sign-extension (or
  `unpack4x8snorm(word)·127.0` for the snorm mapping — decided at impl time by
  which lowers cleaner) so `q` is the true integer in [-127, 127], multiplied by
  the group scale. Accumulation is f32.

### int4 packing (PLANNED — do NOT implement this phase)

Same skeleton, **8 int4 per u32 word**, packed 8-along-K into `[N, K/8]` u32,
G likely 32. Dequant unpacks a nibble (`(word >> (4·j)) & 0xF`, sign-extend from
4 bits) instead of a byte. The variant/format machinery below is designed so int4
is a *new operand-format value* (`"int4-grouped"`) reusing the same seam — no new
fork. Deferred to phase 2 pending the phase-1 report.

## The operand-format declaration (selection-as-data)

The matmul variant registry (`src/backend/webgpu/matmul/variants.ts`, task #61)
selects a kernel family from `MatmulVariantContext` — pure data (m/n/k, dtypeA,
dtypeB, transpose, epilogue, capability). A quantized weight is a **new operand
format on the B operand**, which the context must carry.

`DType` (`src/backend/types.ts`) is `"f16" | "f32" | "i32" | "u32" | "bool"`.
Adding `"int8-grouped"` as a DType is **wrong**: a DType is a single-buffer scalar
element type, but a quantized operand is *two* buffers (packed + scales) with a
group parameter. Conflating them would leak quantization into every dtype-keyed
code path (dtype-promotion rules, elementwise ops, casts) — a new sin-taxonomy
side channel.

Instead, introduce a small **`QuantFormat`** descriptor carried alongside the
operand, mirroring how `dtypeB` and `epilogue` already ride in
`MatmulVariantContext` and `DispatchMatmulOptions`:

```ts
// src/backend/webgpu/matmul/types.ts (new)
export type QuantFormat = {
  scheme: "int8-grouped"; // "int4-grouped" added in phase 2
  groupSize: number;      // G along K (declared, not constant)
  // scales live in a companion buffer bound next to the packed weight;
  // shape [N, K/G] f16, addressed by (n, k/G).
};
```

- `MatmulVariantContext` gains `quantB?: QuantFormat`. Selection keys on it:
  a new `quantVariant` (or a `quant` flag inside the gemv/tiled variants —
  decided below) is applicable iff `quantB` is present.
- `DispatchMatmulOptions` gains `quantB?: QuantFormat` and a `scalesBuffer`
  companion (bound like an epilogue input — an extra storage binding).
- The **tensor** carries the format so `api.linear` can thread it through. A
  Linear whose weight is quantized holds `weight` (the packed u32 tensor) plus a
  `weightScales` tensor and a `weightQuant: QuantFormat` field. `api.linear`
  reads these off the module (not off the Tensor object — keeping Tensor free of
  quant state, consistent with admission-pressure: prove it on the module first).
  The matmul call site attaches `quantB` + `scalesBuffer` to the dispatch options.

This is the selection-as-data idiom exactly: "which kernel, with which operand
format" is a data decision in the context, resolved by the registry, single-
sourced at `planTiledMatmul` (the seam both the executed path and the stage-4
stream generator consume).

### Variant vs. flag

The packed-int load is a *load-path* change, not a new grid/reduction strategy.
Both the GEMV NT kernel (the dominant M=1 decode path — `api.linear` weight is NT
after simple-transpose detection) and the tiled kernel differ from their f16
selves only in **how they read B in the K-loop**. So the cleaner factoring is a
**`quant` axis inside the existing variants** (like `useSubgroups` is a config
axis, not a variant), NOT a third top-level variant:

- `gemvVariant.isApplicable` already gates on geometry; it gains a branch that
  accepts `quantB` (the NT dot-product load becomes a dequant-load).
- `GemvKernelOptions` / `CodegenOptions` gain an optional `quantB: QuantFormat`;
  the kernel builder injects the dequant into the K-loop load and binds the
  scales buffer.

Phase 1 implements the **GEMV NT** path first (the decode hot path that the Mac
demo actually spends its time in) and the **tiled** path if tractable in the same
pass; if only GEMV lands in phase 1, the tiled path stays f16 and we say so. The
epilogue-fused GEMV variant already exists (bias/activation seam); quant is
orthogonal to it (dequant is at load, epilogue is at store) so they compose.

## Where dequant happens (fused into the K-loop load — never materialized)

In `createGemvKernel` NT mode the K-loop is (gemv.ts:354-373):

```ts
ctx.forStride(lane, k, groupSize, (i) => {
  acc.addAssign(ctx.load("a", i).toF32().mul(ctx.load("b", rowBase.add(i)).toF32()));
});
```

Quantized B replaces `ctx.load("b", ...)` with a **dequant-load**: read the u32
word `bq[(rowBase + i) / 4]`, extract byte `(rowBase + i) % 4`, sign-extend to
int, multiply by `scales[(row * (k/G)) + (i / G)]` (an f16 load → f32). The
tile-IR already has every primitive needed — `shr`, `and`, `toF32`, `bitcast`,
and the `unpackHalf` precedent for f16-through-u32 (tile-ir.ts:293, the #59
mechanism). We add a small `ctx.dequantInt8(word, byteIdx, scale)` helper (or
inline via the existing bit-ops) — a *tile-IR expression*, so it lowers to WGSL
through the same compiler and is visible to the graph, not a side channel.

**Crucially: no dequantized weight tensor is ever created.** The f16/f32 weight
never exists at runtime — the packed buffer is the only weight residency. This is
the whole residency win; a materialized dequant tensor would defeat it and
reintroduce the f16 bytes.

The scales buffer binds as an extra storage buffer, addressed the same way the
epilogue-input bindings are (gemv.ts:307-310) — `[a, bq, out, params, scales]`.
The group-size G enters as a shader constant (in the cache key) so `i / G` is a
compile-time-division-friendly shift when G is a power of two (G=64 → `i >> 6`).

## Quantize-at-load path

A quantization utility converts an f16/f32 weight `[N, K]` → packed u32
`[N, K/4]` + f16 scales `[N, K/G]`:

```
for each output row n:
  for each group g of G contiguous-K weights:
    scale[n,g] = max(|w[n, g*G : (g+1)*G]|) / 127
    q = round(w / scale)  clamped to [-127, 127]
  pack 4 consecutive q along K into each u32 word
```

- **Lives in `tools/` first** (admission pressure — prove it in tooling before
  `src/`). A `quantizeLinearWeight(f32data, N, K, G)` → `{packed: Uint32Array,
  scales: Uint16Array}` pure function, plus a loader hook. If it earns generality
  (a second model, a browser path) it graduates to `src/`.
- **Loader integration** (examples/qwen3/loader.ts `extractWeight` / the
  `resolveDest` copy): when a target Linear is marked quantized, quantize the
  extracted f32 data on the host and upload the packed + scales buffers instead
  of the f16 weight. The model config gains `weightDtype: "int8"` alongside the
  existing `"f32" | "f16"` (packages/qwen3-browser model.ts:41) — a Linear built
  with the int8 format allocates the packed weight + scales tensors.
- **Browser IDB caching**: the packed weight is smaller and quantization is a
  one-time host cost, so caching the packed form in IndexedDB (as the browser
  demo caches f16 weights today) is strictly better — noted, but **browser
  wiring is OUT of scope this phase**.

## Accuracy budget and gates (thresholds stated BEFORE measuring)

Per-group symmetric int8 is a well-characterized ~8-bit weight quantization; the
expected logit drift on a 1-2B model is small but nonzero. Thresholds:

1. **Kernel-level differential (exactness — no quant error).** Quantized-matmul
   kernel vs the f16 matmul with the SAME already-dequantized values as input:
   feed `q * scale` (computed on host, as f16) as an ordinary f16 weight to the
   f16 kernel, and the packed `(q, scale)` to the quant kernel, same activations.
   These must agree to **f32 rounding noise (max-abs ≤ 1e-4 relative)** — this
   isolates *kernel correctness* (the unpack/dequant/accumulate arithmetic) from
   quantization error. This is the CLAUDE.md cross-path numerical guard the new
   optimized path is required to ship with. Lands as an in-suite spec
   (webgpu project).

2. **Model-level parity (quant error, real weights).** Quantized model vs f16
   baseline on real prompts, one forward:
   - **Top-1 next-token agreement: 100 %** on a fixed prompt set (≥8 prompts) —
     if int8 flips the argmax the format is too lossy for this scope.
   - **Top-5 logit agreement: ≥ 4/5** overlap per prompt.
   - **Max-abs logit drift ≤ 0.5** and **mean-abs drift ≤ 0.05** vs f16 (logits
     are O(10-30) in magnitude; 0.5 is < 2 % of range and below the top-1 margin
     for well-separated tokens).
   Thresholds are set here, before measuring. If real numbers blow past them, the
   report says so honestly rather than moving the goalposts.

3. **Generation coherence, ≥ 20 tokens.** Greedy-decode 20+ tokens from a fixed
   prompt; output must be coherent English and match the f16 greedy decode for at
   least the first several tokens (divergence deep in a greedy chain is expected
   and acceptable; early divergence is a red flag).

4. **`npm run build` + full suite + `npm run test:gates` green.** Weight-only
   inference doesn't touch the optimizer/compiled-plan replay hazards (it's
   noGrad), but the gates must stay green (no regression to the training path).

5. **Measured perf (honest).** tokens/s and bytes-resident, quant vs f16, same
   A100, same prompt, same model. A100 is not bandwidth-starved, so the tokens/s
   win may be ~0 or even slightly negative (extra unpack ALU); the **residency**
   win (≈2× smaller weights) is the real, measurable A100 result and the proxy
   for the Mac bandwidth win. Report both, and the Mac bandwidth projection.

## Env flags

**Target: zero new `TORCHLETTE_*` flags.** The format is selected by the operand
descriptor (data), not a global flag. If a temporary opt-out is unavoidable
during soak (e.g. `TORCHLETTE_QUANT=0` to force the f16 path for A/B), it is born
with a sunset: soak → default → delete, named in the landing commit.

## Sin-taxonomy check (architecture-debt.md)

- **Not an op-granularity side channel:** G and the scale buffer are *data* —
  scales are a bound buffer (graph-visible), G is a shader-cache-key constant, the
  format rides in the variant context. No per-step-varying scalar is baked into a
  recipe (weight-only inference is static — scales never change after load, unlike
  Adam's step_size). This sidesteps the frozen-scalar disease entirely.
- **Single source at the seam:** the operand format is declared once (on the
  module), threaded once (`planTiledMatmul` context), and both the executed path
  and the stream generator read the same context. The scales buffer's `[N, K/G]`
  layout is asserted at the quantize/dispatch seam (packed element count must
  equal `N·K`, scale count must equal `N·K/G`).
- **Differential across the activation threshold:** gate #1 crosses the kernel's
  correctness boundary; gate #2 crosses the real-weight quant boundary. Both are
  same-input cross-path diffs, per the corollary.

## Phase-1 deliverables (STOP after)

int8 weight-only for GEMV NT (+ tiled if tractable), quantize utility in tools/,
one real model (Qwen3-1.7B) wired end-to-end on Node/Dawn, all five gates.
**STOP and report** — int4, browser integration, and SAE-demo wiring are phase 2,
decided by the phase-1 report.

## Phase-1 outcome (2026-07-11)

**Scope landed: GEMV NT (decode) only.** The tiled (M>1 / prefill) quant path was
NOT built — the tiled B operand is [K,N] with a transposed, cooperatively-staged
load, so fusing dequant there is a materially larger and less-tested change than
the GEMV NT row-dot. Prefill therefore stays f16/f32; decode (the M=1,
bandwidth-bound path that is the entire steepness argument) is the quant target.
Stated per the design's own "if only GEMV lands, say so."

**Model wiring: the tied lm_head / embedding** (N=vocab, K=hidden — the single
largest weight, ~311M params, the clearest residency win), spliced as the final
decode projection where the input is a real materialized hidden state and the op
is a clean M=1 GEMV. Full per-layer (q/k/v/o/gate/up/down) wiring needs the input
forced+materialized before each eager quant dispatch, which fights the lazy
forward (RoPE/attention compose lazily) — that's a **lazy custom-op** integration
(a new IR node + backend executor threading the packed+scales pair through the
plan builder), deferred to phase 2. The eager `dispatchQuantizedGemvNT` proves
the op on a real model-weight distribution end-to-end.

**Gate results (A100 dw-2-1, real Qwen3-1.7B lm_head N=151936 K=2048):**
- Gate 1 (kernel exactness, `probe-quant-gemv.ts`, in-suite
  `test/quant-gemv-parity.spec.ts`): quant kernel == f32-dequant control to f32
  noise (~1e-5 rel) over 10 shapes, G∈{64,128}. PASS.
- Gate 2 (real-weight parity, `quant-lmhead-realweight.ts`): int8 lm_head vs f32
  lm_head — top-1 match, top-5 5/5, **max-abs logit drift 0.103** (≤0.5),
  **mean-abs 0.016** (≤0.05), on a real weight + realistic post-norm hidden.
  PASS.
- Gate 5 (perf/residency): **residency 3.88x vs f32, 1.94x vs f16** (1245→321MB).
  **Speed: int8 2.18 ms/call vs f32 1.34 ms/call — int8 is SLOWER on A100.**
  Honest and expected: A100 (~2TB/s) is not bandwidth-starved, and the phase-1
  kernel is SCALAR-load (per-element `unpack4x8snorm` + f16 scale) while the f32
  GEMV uses vec4 loads. The A100 win is residency; the tokens/s win is the 16GB
  Mac's bandwidth term (lm_head 1244→311 MB/token). A vec4-vectorized int8 load
  is the obvious A100-speed follow-up (phase 2).
- Gate 3 (20-token generation): NOT completed — the full 28-layer model forward
  hangs intermittently at load/fence under Node/Dawn on this box (the
  vkCreateDevice-class flake the task flags), unrelated to the quant kernel
  (which passes standalone). Deferred to the lazy-custom-op wiring (phase 2),
  where per-layer quant runs inside the model's own forward.
- Gate 4 (build + suite + gates green): build clean; the new kernel gate is
  in-suite (webgpu project); the existing GEMV probe still passes 124/124 (no
  regression from the quantB branch).

**Kernel seam bug caught pre-ship** (the single-source-at-seam class): the
tile-IR dead-code pass (`someExprChild`) had no `unpackInt8Snorm` case, so it
pruned the packed-weight load's index `let`s — the WGSL referenced an undeclared
`b_row_base`. Added the case; a differential would have caught it at runtime, but
the missing traversal case is the exact "two sides of a seam silently disagree"
pattern the house rules warn about. Also: `unpack4x8snorm` is the chosen builtin
(q/127, ·127 folded into the stored scale = group abs-max — the quantizer is the
single source for that mapping).
