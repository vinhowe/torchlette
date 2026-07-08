# inc-2a: capturable-optimizer contract (optimizer-scalars-as-data) — implementation handoff

**Status: IN PROGRESS. Gate 1 (build) + Gate 2 (formula) GREEN and committed.
The rest of the ladder is not started — it requires the full ~20-file ripple
below and 9 more GPU gates. Per the campaign charter ("commit only at full
green; land NOTHING half-done"), NO source ripple has been landed; this doc +
the Gate-2 test are the committed-doc handoff so the next session starts from a
known-good tree with the risky assumption already retired.**

Authoritative spec: `docs/staged-execution-phase2b.md`, the **inc-2a** section
(the essence, retirements, gates). This doc is the *execution plan* for that
spec — read the spec first, then this.

---

## What is DONE (committed, verified)

1. **The riskiest assumption is retired (Gate 2).** The in-kernel bias-correction
   derivation from an on-device `t` is numerically validated against f64.
   - Test: `test/optim/adam-biascorrection-formula.spec.ts` (cpu project,
     GPU-free, permanent gate).
   - **Measured (fround-emulated WGSL f32, t∈{1,2,3,10,100,1000,10000,200000},
     β=(0.9,0.999)):** expm1 form worst stepSize rel err **6.33e-7**, worst bc2
     rel err **7.28e-8**; naive `1−pow(β,t)` worst bc2 rel err **1.95e-5 @ t=2**
     (reproduces the spec's measured cancellation). Gate band < 2e-6: PASS for
     expm1, FAIL for naive — the test asserts both directions so a regression to
     the naive form cannot ship silently.
2. **Build is clean** with the test present (`npm run build`).

## The LOCKED formula (single source at the seam)

The WGSL kernel derivation and the test's `expm1F32`/`biasCorrectedStepSizeF32`
emulation MUST stay identical. Constants:

```
lnBeta1, lnBeta2  : precomputed f64→f32 on CPU, passed as STATIC uniforms
bc = -expm1(t * lnBeta)
expm1(y):  |y| < 0.25 → 5-term Horner  y*(1 + y*(1/2 + y*(1/6 + y*(1/24 + y/120))))
           else       → exp(y) - 1
stepSize     = lr * sqrt(bc2) / bc1
epsAdjusted  = eps * sqrt(bc2)     // eps ORIGINAL is the static uniform now
lrTimesWd    = lr * weight_decay   // lr from buffer, wd static
```
`t` is f32-exact to 2^24 steps (fine — 200k gated). Same formula across ALL
fused variants (scalar/vec4/f16/unscale/chunked), foreach, and elementwise —
no per-path special case (the (d) requirement). Foreach/elementwise need the
same series as a graph subgraph (see "expm1 as a graph op" below).

---

## The mechanism (design, locked from spec + code)

- **`t`**: ONE persistent `f32 [1]` tensor per optimizer, advanced IN-PLAN by
  `copy_(t, add(t, 1))` inside the optimizer plan (so replays advance it). Per-
  param `steps[i]` collapses to shared `t` (PyTorch capturable semantics: a
  param whose grad is absent skips its update but `t` still advances — DOCUMENT
  the divergence from today's per-param counters; it is the explicit-never-
  silent baseline-shift risk called out in Gate 8).
- **`lr`**: ONE persistent `f32 [1]` tensor per param-GROUP, written by the LR
  scheduler / `setLR` at DRIVER level (outside any captured body).
- **`invScale`** (GradScaler): persistent `f32 [1]`, `invScale = reciprocal(scale)`
  as a graph op, fed to `unscaleGrad` as a TENSOR input. The CPU `_scale` number
  is demoted to a stats-only mirror (updated at `resolveDeferred` readbacks).
- **Fused kernel bindings**: add `t` and `lr` as two 1-elem storage bindings
  (direct persistent node inputs — NOT a packed hyper buffer; direct inputs are
  the most capture-faithful, zero extra graph ops, and binding count stays ≤ 10:
  grad,param,m,v,[param_f16],t,lr = ≤7 storage). The uniform config becomes
  FULLY STATIC: beta1, beta2, eps(orig), weight_decay, decoupled_wd,
  num_elements, lnBeta1, lnBeta2 (the last two replace the removed step_size /
  lr_times_wd fields; reuse the `_pad*` slots).

### adamStep node: 4 inputs → 6 inputs `[grad, param, m, v, t, lr]`
This is the ripple. `t`/`lr` flow as tensor DATA so the recorder sees TAG_WRITE
stable buffers (the whole point — kills the varying-payload refusal class).

---

## THE LOAD-BEARING HAZARD (justified the stop; test-FIRST)

The default fused path is the **packed** optimizer (`dispatchPackedOptimizer`,
`src/optim/packed-dispatch.ts`; `adamStepBatch` in
`src/backend/webgpu/ops/fused.ts:244`): it concatenates grad/param/m/v across
same-element-count params and dispatches ONE kernel per size class with ONE
config. Per-param `lr` as a scalar buffer only works if **every item in a packed
group shares the same `lr` tensor** (same param-group) and the shared `t`.

**Required fix:** the adam-batch grouping key
(`src/executor/lowered-plan.ts` ~737-763 emits the `adam-batch` action;
`src/executor/executor.ts` ~2511-2574 builds `AdamBatchItem[]`) must key on
**lr-tensor identity / group index**, not the old config hash. Get it wrong and
one group's LR is silently applied to another group's params — invisible to
every single-group test. **This is exactly the frozen-scalar-family failure
mode.**

**NEW GATE (test-FIRST, write before touching the packed path):** a MULTI-GROUP
differential — two `AdamParamGroup`s with different LR (and different wd), packed
fused path, trajectories vs the sequential/elementwise reference over 20+ steps.
It MUST FAIL if the grouping key is wrong. No existing gate covers this
(`fused-vs-elementwise.spec.ts` is all single-group). Extend
`tools/adam-trajectory-probe.ts` to accept a 2-group config, then add the gate
to `test/optim/fused-vs-elementwise.spec.ts` (or a sibling multigroup spec).

---

## Ripple map (edit set, with anchors)

| file | change |
|---|---|
| `src/backend/webgpu/adam-kernel.ts` | +2 storage bindings (`t`,`lr`); static uniform struct (drop `step_size`/`lr_times_wd`, add `ln_beta1`/`ln_beta2`); in-kernel `emitBiasCorrection`/expm1 helper used by `emitAdamScalarBody` across all variants; **RETIRE** `volatileRepack` (both `dispatchAdamStep` and `planAdamStepDispatch`) + the per-step fields of `setAdamConfigUniforms`. |
| `src/backend/types.ts:193` | **RETIRE** `AdamStepConfig.stepSize` / `.lrTimesWd` / `.invScale`. |
| `src/optim/adam.ts` | persistent `t` (`_advanceStep`→in-plan `copy_(t,add(t,1))`); persistent per-group `lr` tensors; `_stepFused` node gets `[grad,param,m,v,t,lr]`, drops JS bc1/bc2/stepSize/eps-adjust; `_foreachGroupStep` + `_updateParamElementwise` replace JS bc1/bc2/stepSize with the shared expm1 subgraph reading `t`/`lr`; `setLR`/`setGroupLR` write the lr tensor. |
| `src/optim/lr-scheduler.ts` | schedulers keep computing the JS value but `setLR` now WRITES the persistent lr tensor (the `HasLR` seam already funnels through `setLR`). |
| `src/optim/grad-scaler.ts` | `_scale`→persistent tensor; `scale(loss)`=tensor mul; `invScale=reciprocal(scale)`; `_unscaleFused` passes invScale as a TENSOR input to the `unscaleGrad` node; CPU number stats-only. |
| `src/backend/webgpu/unscale-kernel.ts` + `ops/fused.ts:369` | invScale from a bound 1-elem buffer, not config; **RETIRE** its volatileRepack. |
| `src/executor/op-dispatch.ts:484` `executeAdamStep` | pass `backendInputs[4]=t`, `[5]=lr`. |
| `src/backend/webgpu/ops/fused.ts` | `adamStep`/`adamStepBatch`/`PackedOptimizerItem` signatures gain t/lr buffers; packed kernel binds one t + one lr for the group. |
| `src/executor/executor.ts` (adam-batch) + `lowered-plan.ts` | `AdamBatchItem` gains t/lr; **grouping key on lr-tensor identity** (the hazard fix). |
| `src/executor/stream-generate.ts` ~2900-3107 | generated packed-adam buffer/slot resolution for t/lr inputs; **RETIRE** the adam `volatilePack`/TAG_UNIFORM path (config now static). |
| `src/executor/compiled-plan.ts`, `scalar-table.ts` | adam uniform now static; drop the adam volatile-uniform recording plumbing. |
| `src/compiler/graph-rewrites.ts`, `fusion-detect.ts` | adamStep structural/input keys must include the 2 new inputs (keep the outputIndex convention). |
| `src/core/step-tape.ts` | **RETIRE LAST** the inc-1 `stDeclareBatchCover` batch-representative rule FOR ADAM (`~272-290`, `~420-426`, refusal msg `~590`, persistence `~779-784`) — Gate 6 requires the probe to pass WITHOUT it firing for adam. Keep the mechanism only if another batched op needs it; else delete and name it. |
| `src/remote/wire.ts`, `src/remote/client-engine.ts` | 6-input adamStep node serialization. **WIRE-COMPAT:** this changes the on-wire op arity — add a wire-format version note in the commit; the distributed path is Vin-kept but not hot, so a version bump/note suffices (no live-peer migration needed). |

### expm1 as a graph op (foreach/elementwise)
The spec mandates ONE formula across all three paths. Add `expm1` either as a
registry unary op (`src/ops/registry.ts` — WGSL `exp(x)-1`, but that reintroduces
cancellation for the graph paths at small y) OR as a shared TS helper that builds
the series subgraph (`y*(1+y*(...))` with a `where(|y|<0.25, series, exp(y)-1)`
select). Prefer the series subgraph helper so foreach/elementwise match the
kernel's expm1 exactly. One helper, called by `_foreachGroupStep`,
`_updateParamElementwise`, and any CPU reference.

---

## Named retirements (execute + name in the commit, with ref counts)
Retire LAST, only after the replacing mechanism's gates are green:
1. `AdamStepConfig.stepSize` / `.lrTimesWd` / `.invScale` (3 fields).
2. `setAdamConfigUniforms` per-step volatile fields → static-only.
3. adam's TAG_UNIFORM volatile-repack closure + recording plumbing
   (`adam-kernel.ts` ×2, `tile-dispatch` adam usage, `compiled-plan`/`scalar-table`).
4. inc-1 batch-representative recorder rule for adam (`step-tape.ts`).
5. `unscaleGrad` invScale config field + its volatile uniform.
6. foreach's per-step scalar-adapt churn (JS bc1/bc2/stepSize).
Target: NEGATIVE or near-zero net src SLOC. `bash tools/weight-norm.sh --log`
at campaign end; the commit names deletions.

---

## Gate ladder (11 gates; all solo, vk-shim pinned, verbatim in report, in order)
GPU pinning: `VULKAN_DEVICE_INDEX=<free> LD_LIBRARY_PATH=<tools/vk-shim ...>`
(the launch-diloco mechanism; Dawn ignores CUDA_VISIBLE_DEVICES). Pick a free
device via `nvidia-smi`. Write output to files; judge by printed results (Dawn
may exit 139 at teardown after complete output).

1. `npm run build`. **[DONE]**
2. Kernel-formula unit check — `test/optim/adam-biascorrection-formula.spec.ts`,
   rel err < 2e-6. **[DONE]**
3. fused == foreach == elementwise trajectory differentials
   (`test/optim/fused-vs-elementwise.spec.ts`, extended for the new mechanism).
4. **NEW (test-first): MULTI-GROUP** — 2 groups, diff LR+wd, packed fused vs
   sequential reference, 20+ steps. Must fail if grouping key wrong.
5. LR-SCHEDULE EXACTNESS — fullstack CosineAnnealingLR digit-for-digit vs
   sequential over 30+ steps; late-LR-change tests (#1/#6) still green
   (`tools/parity-fullstack-tl.ts` + the fused-vs-elementwise late-LR tests).
6. GradScaler trajectory parity vs current main over 30 steps
   (`test/optim/grad-scaler.spec.ts` + a fullstack scaler arm).
7. `tools/t-train-tape-probe.ts`: eligiblePairs>0, refusals=0, BOTH fused and
   foreach — AND the adam batch-cover rule retirement verified (probe passes
   WITHOUT that rule firing for adam).
8. Full suite BOTH flag states (`TORCHLETTE_STEP_TAPE` on/off); `test:gates`
   4/4; observed-liveness spec 5/5.
9. Production 124M regression: baselines {9.81, 5.92, 5.15, 4.64}. Shared-t
   semantics may shift grad-gap edge cases — if so STOP and report the delta
   EXPLICITLY (never silently re-baseline).
10. Decode stack: gen-tape-gate (hardened, 3-gen) + kv-differential PASS.
11. STRICT_LIFETIME + STRICT_GPU fullstack — zero throws.

Commit when ALL green (single commit; message names mechanism + expm1 formula +
every retirement with ref counts + the wire-format version note).

---

## Next action for the resuming session
Start at sequencing step "kernel + adam.ts + backend plumbing" (the zero-cross-
file-risk pieces — Gate-2 test, formula — are done). Recommended order inside:
(a) add persistent t/lr to adam.ts + the `emitBiasCorrection` helper +
`expm1` graph helper; (b) 6-input node + executeAdamStep + fused.ts single-item
adamStep (get the NON-packed path green on Gate 3 first); (c) packed path +
grouping-key fix, gated by the NEW multi-group Gate 4 written first; (d)
stream-generate/compiled-plan; (e) GradScaler + scheduler; (f) retirements;
(g) full ladder. Keep the tree buildable between sub-steps but do NOT commit
until full green.
