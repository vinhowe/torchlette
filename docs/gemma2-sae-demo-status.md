# Gemma-2-2B SAE-steering demo — status & handoff

Branch `sae-demo` (from main 062ae36f). SAE-feature steering on Gemma-2-2B in the
browser, using the Gemma Scope layer-20 residual-stream SAE. Template: the
qwen3-steering CAA demo; model substrate: the gemma2-browser stack + the layer-
indexed `residualHook` seam.

## Convention verifications (all confirmed against the official release)
- **Hookpoint**: Gemma Scope `layer_20` res SAE = `blocks.20.hook_resid_post` —
  the residual stream AFTER the full block 20 (attn + MLP), before the final
  norm (arXiv 2408.05147 §3.1 "post MLP residual stream"). In this codebase that
  is `hidden[21]` from `Gemma2.forward({collectHidden:true})`, i.e. the value
  `residualHook(x, 20)` sees.
- **JumpReLU encode** (App. A + tutorial): `pre = x @ W_enc + b_enc`;
  `acts = relu(pre) * (pre > threshold)`. **No b_dec pre-centering at inference.**
- **Decode**: `x_hat = acts @ W_dec + b_dec`.
- **Steering direction** = `W_dec[feature]` (a row). **W_dec rows are unit-norm**
  (verified: min=max=1.0000), so α is directly in residual-norm units; no
  renormalization applied.
- **Canonical variant**: `layer_20/width_16k/average_l0_71` (16384 features,
  dModel 2304). Neuronpedia's `20-gemmascope-res-16k` maps to this.
- **Neuronpedia URL**: `https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/{feature}`.
- **Weights** load as **f32** (~300MB, 151MB per W matrix — comfortable on 16GB).

## Milestones
- **M1 (commit d208bba4) — SAE pipeline + parity gate. GREEN.**
  `packages/gemma-scope-sae`: framework-agnostic `GemmaScopeSAE` (encode/decode/
  featureDirection). `examples/gemma2-sae/{dump-sae-reference.py,sae-parity.ts}`:
  dump the layer-20 residual (Python/transformers) + a numpy JumpReLU reference on
  the SAME residual; the TS encode over that dump matches. RESULT (device 10, 3
  prompts): acts maxAbs=2.4e-3 (gate 5e-2), top-20 agg+last indices EXACT,
  n_active_last exact (87/52/45, L0≈71). `convert-npz.py` flattens params.npz →
  raw f32 .bin + manifest.
- **M2 (commit bf5c3a2a) — browser stack + app. GREEN (compiles/serves).**
  `packages/gemma2-browser` gains browser-loader / generate / weights-map /
  idb-cache (mirrors qwen3-browser, Gemma-2-adapted). `examples/gemma2-sae-demo`
  (SvelteKit static, dev :5176): worker-based load, FEATURE INSPECTOR (top-K
  features + Neuronpedia links), STEERING panel (indices/clicks + bipolar α,
  multiple features), baseline-vs-steered compare, tape on + tok/s. Verified:
  vite serves the app + worker + SAE assets; svelte-check clean in app src (the
  residual errors are the pre-existing Module.forward-override class in model.ts).
- **M3 (commit TBD) — presets. GREEN.** `examples/gemma2-sae/preset-{sweep,
  refine,confirm}.ts` discover themed features (max-agg over CONTENT positions —
  skipping position 0 = `<bos>`, whose attention-sink features fire identically on
  every prompt) + sweep α; picked → `src/lib/presets.ts`. Verified greedy
  baseline-vs-steered (feature direction = W_dec[f], hook adds α·dir at layer 20):
  - **#12082 "dogs" α=120** — the Golden Gate moment. "…my favorite thing to do on
    the weekend is to go to the **farmers market**…" → "…to go to the **DOG PARK**.
    I love watching the **dogs** play and run around…". Fully coherent.
  - **#8993 "banking/finance" α=100** — "…a new way to think about the world…the
    new **physics**…" → "…a new kind of **bank account**…a 'digital wallet'…".
  - **#3124 "San Francisco / Bay Area" α=150** — the literal Golden-Gate analog;
    subtler pull (nudge α up to strengthen).
  The same position-0 fix is applied to the app's `inspectFeatures` (else the
  inspector shows identical generic features for every prompt).
  **α is in residual-norm units** (W_dec unit-norm); coherence holds to ~150,
  degenerates into repetition above.

## Running
```bash
# Weights (once): unsloth/gemma-2-2b + the SAE npz → ckpts/ (symlinked into worktree)
#   hf download unsloth/gemma-2-2b --local-dir ckpts/gemma-2-2b
#   hf download google/gemma-scope-2b-pt-res layer_20/width_16k/average_l0_71/params.npz --local-dir ckpts/gemma-scope-2b-pt-res
#   uv run --with numpy python3 packages/gemma-scope-sae/src/convert-npz.py \
#     ckpts/gemma-scope-2b-pt-res/layer_20/width_16k/average_l0_71/params.npz \
#     ckpts/gemma-scope-2b-pt-res/sae-layer20-16k --layer 20 --width 16384 --l0 71

# SAE reference (CUDA torch — conda has it; uv's torch is CPU-only):
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python3 examples/gemma2-sae/dump-sae-reference.py

# SAE parity gate (Dawn — pin a clean GPU via VULKAN_DEVICE_INDEX + vk-shim):
VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim npx tsx examples/gemma2-sae/sae-parity.ts

# Preset sweep:
VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=1 \
  npx tsx examples/gemma2-sae/preset-sweep.ts

# The app:
pnpm --filter gemma2-sae-demo dev   # → http://localhost:5176
# (static/sae/*.bin are symlinked to ckpts; gitignored. For a prod build, copy
#  the real .bin files into static/sae/ — adapter-static won't follow symlinks.)
```

## Notes / gotchas
- Dawn IGNORES CUDA_VISIBLE_DEVICES — pin with `VULKAN_DEVICE_INDEX=N` +
  `LD_LIBRARY_PATH=tools/vk-shim`. Pick a clean GPU (nvidia-smi memory.used <100MB).
- The reference dump needs CUDA torch: use `/opt/conda/bin/python3` (torch
  2.5.1+cu121 + transformers 5.3.0), NOT `uv run` (CPU-only torch).
- Any standalone Dawn script ends with `process.exit(0)`.
- The app worker's `tape-flag` import must stay FIRST (sets TORCHLETTE_STEP_TAPE
  before torchlette module eval).
- Did NOT touch `src/` (weight-norm srcSLOC unchanged); all work is in
  packages/ + examples/.
