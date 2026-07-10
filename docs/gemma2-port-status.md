# Gemma-2-2B port — status & handoff

Branch `gemma2-port` (from main 0d2accd4). Ports Gemma-2-2B onto the browser
inference stack, the last prerequisite of the SAE-steering demo (Gemma Scope
SAEs target this model). Template: the Qwen3 stack.

## Status: M1 + M2 GREEN

Two commits: M1 (node logit parity), M2 (generation). Both gated.

## Config (verified from unsloth/gemma-2-2b config.json)
vocab 256000, hidden 2304, 26 layers, 8 heads / 4 KV heads, head_dim 256,
intermediate 9216, rope_theta 10000, rms_norm_eps 1e-6, attn_logit_softcapping
50, final_logit_softcapping 30, sliding_window 4096, query_pre_attn_scalar 256.
layer_types alternate `sliding_attention`/`full_attention` starting SLIDING at
layer 0 → **EVEN layers = local/windowed, ODD = global** (opposite of the task's
parenthetical guess; verified against the config + transformers Gemma2 source).

## Weight source
`unsloth/gemma-2-2b` — ungated base mirror (gated=False), single 5.2GB
safetensors, snapshot 25319945f7fd83b8b903e12081777b7eef2ba993. No HF_TOKEN in
the environment; this mirror needs none. Reference dumped with conda torch
2.5.1+cu121 (uv installed a CPU-only torch; conda has CUDA) + transformers 5.3.0,
`attn_implementation="eager"`, fp32, `<bos>` prepended.

## Architecture deltas vs Qwen3 (all reuse/declaration per #64, EXCEPT 2 engine gaps)
- Attention soft-cap (cap=50, ALL layers) + sliding window (4096, LOCAL layers)
  via #64 modifiers: `{scoreMod:{softcap,50}, maskMods:[{causal}(,{slidingWindow,4096})]}`.
- Attn scale = `query_pre_attn_scalar**-0.5` (= 1/16), NOT 1/sqrt(headDim).
- RMSNorm sandwich: 4 norms/layer. post-attn/post-ffn norms apply to the
  SUBLAYER OUTPUT before the residual add (differs from Qwen3's postAttnNorm→MLP).
- `(1+weight)` zero-centered RMSNorm: baked +1 at load.
- GeGLU: `gelu(gate, tanh) * up`.
- Embedding × sqrt(hidden) (2304→×48).
- Final-logit soft-cap (cap=30): elementwise `30*tanh(logits/30)`, model-level.
- Tied lm_head, RoPE base 10000 (half-split, matches the fused kernel), GQA.
- residualHook steering seam preserved (layer-indexed) for the SAE demo.
- `attnModKey = gemma2.sc50.w4096` folds into the decode capture bucketKey.

## Files
- `packages/gemma2-browser/src/{model,index}.ts` + `package.json` — the model.
- `examples/gemma2/{loader,parity,dump-reference,kv-differential,
  kv-differential-nocapture,window-discrimination,generate-smoke}.{ts,py}`.

## Loader specifics (the Gemma-2-specific load path)
- f16 linears via a bf16→f16-raw-bits fast upload (`bf16BufferToF16Bits` +
  `tensorFromArray(Uint16Array, f16)`), skipping the slow f32 `Array.from` in the
  creation route — load 5min → ~45s. Norms stay f32.
- **The embedding table is now f16 when the model is f16 (#59).** Gather became
  dtype-parametrized (`src/backend/webgpu/ops/gather-scatter.ts`), so an f16
  table produces a correct f16 lookup. Resident memory: **2.36GB f32 → 1.18GB
  f16** (the single biggest lever on a 16GB Mac). The model then upcasts the
  small [seq,hidden] lookup to f32 so the RESIDUAL STREAM stays f32 — an f16
  residual entering layer 0 diverges catastrophically (hidden[1] maxAbs 1e-6 →
  60, top-5 scrambled); activations inside the linears stay f16 (mixed-dtype
  matmul). f16-embedding M1 parity: top-5 EXACT, logits maxAbs 1.05e-4 / 3.86e-4
  (vs f32-embed 9.35e-5 / 1.39e-4 — the honest f16 rounding; ref is HF f32 eager,
  deployment gates on top-5).
- The f32 embedding (2.36GB) still exceeds the 2GB storage-buffer BINDING limit:
  - Construction randn-init (binds whole) is replaced with a `zeros` (clearBuffer,
    no binding) BEFORE the randn node forces (f32 only; f16 Embedding is zeros).
  - It's uploaded via chunked `writeBuffer` (no binding) + re-registered; the
    embedding-forward gather auto-chunks its READ.
  - The tied lm_head is pre-split into 3 sub-2GB vocab chunks (`Gemma2.lmHeadChunks`)
    matmul'd separately + concat. The **f16 table (1.18GB) is UNDER the 2GB
    binding limit**, so the tied lm_head reads `embedTokens.weight` directly —
    no chunks (`lmHeadChunks` stays null).

## Engine gaps found (the #64 audit claimed "essentially none" — TWO were missed)
Both are the same failure class: a Gemma-2 dimension the audit didn't model.

1. **`maxComputeWorkgroupStorageSize`** (`src/backend/webgpu/gpu-context.ts`,
   both requestDevice paths). The device was requested with the DEFAULT 16KB;
   head_dim=256 attention tiles need 32KB → the attention pipeline was invalid
   and EVERY attention submit was silently DROPPED (block-0-onward hidden states
   read zero). Fix = request the adapter's supported max (V100/A100 report 48KB).
   Harmless for headDim≤128 (Qwen3 fits 16KB). **Required for M1.**

2. **Chunked gather → compiled-plan recorder** (`src/backend/webgpu/ops/
   gather-scatter.ts`). The chunked gather path allocated its output via
   `createTrackedBuffer` (raw device.createBuffer), invisible to the recorder →
   "storage buffer not registered" under ANY compiled-plan/capture recording.
   The 256k-vocab embedding forces the chunked path, and repeated decode builds a
   compiled plan → this blocked ALL of decode. Fix = route through
   `resolveOutputBuffer` (one-line mirror of the direct gather path). **Required
   for M2.** (scatterAdd's chunked path at L388 still uses createTrackedBuffer;
   the KV scatter never chunks so it isn't hit, but it has the same latent gap.)

Everything else in the port is genuinely reuse/declaration as the audit said.

## Gates (verbatim)
M1 (f16 weights vs HF f32 eager reference, V100 device 10):
```
prompt0 logits maxAbs=9.35e-5  top5 exact [476,573,974,1170,2078]
prompt1 logits maxAbs=1.39e-4  top5 exact [141,145,235248,1,140]
ALL PROMPTS PASS
hidden[0] maxAbs=0 (scaled-embedding match); hidden 1..13 ~1e-4
```
Window discrimination (device 0): window>=S ≡ causal (0.0), window<S differs
(0.33) — the slidingWindow maskMod is wired + active exactly when it should be.

M2 (f16, V100 device 10/11):
```
kv-differential: no-cache == cat == static == TAPED identical;
  tape replays=9 hits=9 (100%), readyTapes=1, 0 misses/invalidations.
generate-smoke: 30 tok @ 17.9 tok/s (V100), tape hits=26/30, ready=true.
  "The capital of France is" → " a city of contrasts. It is a city of art,
  culture, and history. It is also a city of fashion, food, and fun." (greedy).
```

## Notes / caveats
- Ran on **sivri** (16× V100-32GB), NOT the A100 dw-2-1 box. f32 Gemma-2-2B
  (~17.5GB + 2.36GB embed + 2.36GB lm-head chunks) is too tight for 32GB; the
  DEPLOYMENT config is f16 (the browser demo runs f16), so M1 gates f16-vs-f32
  in the ~1e-4 qwen3-parity band + exact top-5. On an A100 an f32-vs-f32 run
  would tighten further, but f16 is what ships.
- Perf: 17.9 tok/s on V100. Browser (the real target) + A100 will differ.

## Remaining for the SAE-steering demo
- **Browser wiring**: a gemma2-browser chat/steering app (mirror
  examples/qwen3-chat / qwen3-steering). The model + generateChat-shape decode
  are ready; needs the SvelteKit shell + browser weight loader (the qwen3
  browser-loader/idb-cache pattern). Resident memory (f16 deployment, #59):
  f16 weights ~6.4GB + **f16 embed 1.18GB** (was 2.36GB f32) + norms — the tied
  lm_head reuses the f16 embedding (no separate chunks). Total resident ~7.6GB,
  down ~1.18GB from the pre-#59 f16-lm-head-chunks config; on a 16GB Mac this
  drops gemma-2's ~8.7GB residency toward fully-resident (~6.3GB embed-side),
  which is what unblocks sub-second decode (the owner's Mac re-measure is the
  real acceptance). The residualHook seam is in place for SAE steering.
- **Gemma Scope SAEs**: load + wire an SAE at a chosen layer via residualHook
  (encode activations → steer → decode). Not started.
- **#83 consolidation**: gemma2-browser mirrors qwen3-browser's shape rather than
  sharing code (generate.ts, sampling, KV types, loader scaffold are near-dupes).
  A shared `llm-browser` core (model-agnostic decode loop + static-KV + tape
  wiring, parametrised by a model interface) is the clean consolidation. Deferred
  to avoid a risky qwen3 refactor mid-port.
- **scatterAdd chunked-recorder gap** (gap #2's sibling at gather-scatter.ts:388)
  should get the same resolveOutputBuffer fix before any >2GB scatter path is
  exercised under compiled plans.
