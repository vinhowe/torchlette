# Plain-path "FP spike" — characterization (task #85)

**Status:** CHARACTERIZED. Class verdict: **optimization-dynamics transient, not a
framework/fp bug.** No fix required; P0 gates need a spike-aware policy (below).
**Probe:** `tools/t-fp-spike-probe.ts` + batched driver `tools/t-fp-spike-sweep.sh`.
**Measured on:** WebGPU / Dawn-Vulkan, A100 (`VULKAN_DEVICE_INDEX` + `tools/vk-shim`),
distilgpt2, seq 512, the exact plain loop the P0 gates use (autocast + checkpoint +
GradScaler + clipGradNorm + AdamW + CosineAnnealingLR).

## TL;DR

The folklore "~1/50 single-run fp spike" is a **single training step whose loss
jumps far from both neighbours and then fully recovers on the next step** — a
local peak/valley, not a divergence. Resolved into two populations:

| pop | where | magnitude | recovers | cause | verdict |
|---|---|---|---|---|---|
| **A** | **step 1 (2nd optimizer step)** | up to ~8.4 nats (12.5 → 20.9 → 12.6) | YES, next step | Adam early-bias-correction × data-dependent grad-direction flip (grad-norm surges 20→118) | **THE spike — real dynamics, not a bug** |
| **B** | scattered interior steps | 0.15–0.3 nats | mixed | fp16 accumulation jitter (~1e-3) tripping the detector on otherwise-smooth runs | noise floor, not a spike |

Neither population contains NaN/Inf; no run diverges. The spike is **fully
deterministic per data seed** and **invariant across every execution path**
(fusion on/off, compiled-plan on/off, checkpoint on/off) — which rules out the
entire framework bug class the task worried about (buffer-pool WAW, uniform
reuse, fusion aliasing, GC-timed release) and identifies it as plain
optimization dynamics.

## Measured rate (N=152 clean runs, GPUs 0/10/11)

- **Population A (step-1 transient, |resid| > 0.5 nat): 3 / 152 ≈ 2.0% ≈ 1 in 50.**
  This is the folklore figure, confirmed.
- Detector-flagged transients of any size (incl. pop B): 9 / 152 ≈ 1 in 17.
- **NaN/Inf runs: 0.** GPU uncaptured errors on clean GPUs: 0.
- The 200-run raw sweep had 3 all-zero runs + 1 crashed batch — **all on
  `VULKAN_DEVICE_INDEX=1`** (a VkOOM that dropped a submit → downstream reads
  saw zeroed/stale data). That is an ENVIRONMENT artifact of a memory-pressured
  shared GPU, NOT the training loop; excluded from the rate (see caveat below).

## Characterization table (the step-1 spike)

| property | finding | evidence |
|---|---|---|
| **magnitude** | up to +8.4 nats (loss 12.5→20.9); milder seeds -0.6 nats | sweep + determinism arm |
| **recovers?** | YES — always returns to trend at step+1 (never diverges) | every captured spike |
| **which step** | step 1 (the 2nd optimizer step) exclusively for the large class | histogram: large spikes all at step 1 |
| **grad-norm at spike** | surges 20.1 → **118.0** (5.9×) at step 1, then normal | `GRAD_NORM=1` arm, 5/5 repeats |
| **deterministic per seed?** | YES. seed 389265 ×5 same GPU: step1=20.948±0.002, final=8.1955±0.0007. seed 159614 ×5 identical. control 1234 ×5: no spike. | determinism arm |
| **cross-GPU determinism** | YES — seed 9153 spiked byte-close on 4 independent GPUs (final spread 2.4e-3 = fp16 noise floor) | vary sweep |
| **path dependence** | NONE — step1 = 20.948 (default) = 20.948 (planner off) = 20.951 (fusion+planner+ckpt all off) | bisect arm |
| **backend scope** | measured on WebGPU. CPU out of scope (fp32; and blocked by independent pre-existing CPU bugs — see caveats) | — |
| **strict-lifetime / STRICT_GPU** | no `[lifetime]` throw, no GPU uncaptured error on clean GPUs at spike time | probe error-count delta = 0 |

## Class verdict

**Optimization-dynamics transient (numeric-but-correct), NOT a framework bug.**

Mechanism: at the 2nd optimizer step, Adam's bias correction is still extreme
(`v̂` uses `1−β2^2 ≈ 8e-4`, a tiny denominator). For the ~1/50 of random-data
draws where the first two minibatch gradient directions differ sharply, the
update overshoots — grad norm jumps to ~118 and the loss momentarily rises to
~21 — and the *next* step, now with a better-conditioned `v`, snaps back. This
is standard early-Adam behaviour, amplified here because the probe uses fresh
**random** targets each step (max gradient disagreement) and no LR warmup.

Every framework-bug hypothesis from the house checklist is falsified by the
path-invariance + determinism:
- buffer-pool WAW / uniform reuse / fusion aliasing → would differ between
  fusion-on and fusion-off; it does not (20.948 vs 20.951, within fp16 noise).
- GC-timed release / timing race → would differ across same-seed repeats or
  across GPUs; it does not (5/5 identical, 4/4 GPUs identical).
- compiled-plan frozen-scalar class (#84 relatives) → would differ with
  `PLANNER=0`; it does not.

## Recommendation for P0 differential gates

The spike does **not** poison a well-constructed differential gate, because it is
**deterministic and path-invariant**: a compiled-vs-lowered or tl-vs-PT diff on
the *same seed* sees the spike on *both* arms and cancels it. The historical
gates (`parity-fullstack-tl.ts`, `compiled-plan-parity.spec.ts`) already fix the
seed and compare arm-to-arm, so they are unaffected.

Two concrete guards for the P0 runway:

1. **Fix the seed and compare arm-to-arm (already the practice).** Never gate on
   an absolute loss value at an early step; gate on the *difference* between two
   execution paths of the same seed. The step-1 excursion is identical on both
   sides to fp16 noise (~2e-3), so it cancels.
2. **If a gate must read an absolute early-step loss, avoid step 0–2** (or use a
   short LR warmup / non-random repeated data), where Adam's early-bias transient
   lives. The steady-state trajectory (step ≥ 4) has no large-transient class.

No spike-aware *tolerance inflation* or rerun-on-spike policy is needed for the
seed-fixed arm-to-arm gates; the phenomenon is not a nondeterministic noise
source. **This discharges the #85 P0 entry criterion: the noise source is
characterized and shown not to be a framework noise source.**

## Reproduction

```bash
# single-process sample (keep RUNS <= ~8 — see the multi-run VkOOM caveat)
VULKAN_DEVICE_INDEX=<free> LD_LIBRARY_PATH=tools/vk-shim \
  RUNS=8 STEPS=24 SEED_MODE=vary npx tsx tools/t-fp-spike-probe.ts

# the extreme, deterministic step-1 spike (12.5→20.9→12.6), with grad norms:
VULKAN_DEVICE_INDEX=<free> LD_LIBRARY_PATH=tools/vk-shim \
  RUNS=5 STEPS=24 SEED_MODE=fixed SEED_BASE=389265 GRAD_NORM=1 \
  npx tsx tools/t-fp-spike-probe.ts

# path-invariance bisect (spike survives all three): default; PLANNER=0; FUSION=0 PLANNER=0 CKPT=0
# (set the FUSION/PLANNER/CKPT/CKPT_SEG env flags on the same SEED_BASE=389265)

# large batched sweep across GPUs (batched children avoid the VkOOM):
TOTAL=200 BATCH=8 GPUS="0 10 11" OUTDIR=/tmp/sweep bash tools/t-fp-spike-sweep.sh
```

## Caveats (things that are NOT the spike)

- **Multi-run-per-process VkOOM.** Building > ~10 fresh engines+models in ONE
  Node process exhausts GPU memory (Dawn `CheckVkOOMThenSuccess` → a dropped
  submit → downstream "Input not ready" / all-zero reads). This is per-process
  GPU-memory accumulation across engine rebuilds, not a training spike. The
  batched driver caps runs/child at 8 to avoid it. (A latent follow-on: repeated
  in-process engine construction does not fully reclaim GPU memory.)
- **`VULKAN_DEVICE_INDEX=1` produced all-zero runs** in the raw sweep — the same
  VkOOM-dropped-submit signature, triggered early because that GPU was
  memory-pressured by other tenants at launch. An environment condition, not a
  code path; excluded from the clean rate.
- **CPU backend is out of scope and independently broken** for this loop: the
  memory-planner `captureActionLayouts` path crashes on CPU, and an "Engine is
  busy" re-entrancy fires under repeated `item()`. Neither relates to the fp
  spike (which is an fp16/optimization-dynamics phenomenon).
