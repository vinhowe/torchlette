# Performance Optimization Roadmap

Profiled on DistilGPT-2 training, step 4 steady-state. **Baseline: 272ms/step** (64ms fwd + 147ms bwd + 1ms opt + 59ms cleanup).

GPU compute: 63ms. CPU overhead: 209ms (3.3x GPU). **CPU dispatch is the bottleneck.**

## Recommendations (ranked by impact)

### Tier 1: High Impact (each saves 20-90ms/step)

- [ ] **R1. Buffer pool size-class binning** — *Target: 90ms → ~20ms CPU*
  - 981 createBuffer calls/step at 92µs avg = 90.8ms
  - Pool reuse only 59.7% — 134 fresh allocs despite 257 pooled buffers
  - Fix: Round sizes to power-of-2 or fixed size classes to increase pool hit rate

- [ ] **R2. Fused LayerNorm kernel** — *Target: save ~3ms GPU + ~30ms CPU*
  - 106 mean dispatches (9% GPU) from LayerNorm fwd+bwd
  - Backward recomputes mean/rstd instead of saving from forward (unlike PyTorch)
  - Fix: Single-dispatch fused kernel, save mean/rstd for backward

- [ ] **R3. Reduce Adam CPU overhead** — *Target: 40ms → ~10ms CPU*
  - ensureContiguous (28.3ms) + allocBufs (12.9ms) on fixed-shape params every step
  - Fix: Cache contiguous buffers and pre-allocate output buffers once

### Tier 2: Medium Impact (each saves 5-20ms/step)

- [ ] **R4. Batch queue.submit calls** — *Target: 158 submits → ~10-20*
  - 158 queue.submit/step, each triggers processImmediate overhead (34ms total)
  - Fix: Flush less frequently (every N segments or N dispatches)

- [ ] **R5. Fuse the 5.1ms backward `add`** — *Target: 5ms → <0.5ms*
  - Single add on [50257, 768] (38.6M elements) = lm_head weight gradient
  - Fix: In-place grad accumulation or fuse into matmul backward

- [ ] **R6. Pipeline cache warmup** — *Target: step 0 from 3.4s → ~0.5s*
  - Step 0 is 13x slower due to pipeline compilation
  - Fix: Pre-compile all pipeline variants during model load

### Tier 3: Lower Impact (each saves 2-5ms/step)

- [ ] **R7. Fold more AMP casts into matmul epilogues** — 74 standalone cast ops
- [ ] **R8. Object pooling for Tensor metadata** — GC at 3.7% CPU
- [ ] **R9. Reduce backward sum dispatch count** — 114 sum dispatches

## Impact Projection

| Optimization | GPU saved | CPU saved | New total |
|-------------|-----------|-----------|-----------|
| Current baseline | — | — | 272ms |
| R1: Pool size-class | 0ms | ~70ms | ~200ms |
| R2: Fused LayerNorm | ~5ms | ~30ms | ~165ms |
| R3: Adam cache | 0ms | ~30ms | ~135ms |
| R4: Batch submits | 0ms | ~15ms | ~120ms |
| R5: Fuse backward add | ~5ms | ~2ms | ~113ms |
| **Cumulative target** | **~10ms** | **~147ms** | **~115ms** |

## Profiling Baseline (pre-R1)

### GPU Kernel Time (Step 4)
| Kernel | Count | Total(ms) | Avg(µs) |
|--------|-------|-----------|---------|
| matmul | 93 | 27.6 | 296 |
| matmul++cast+bias | 24 | 8.3 | 348 |
| adamStep | 100 | 8.2 | 82 |
| mean | 106 | 5.6 | 53 |
| add | 14 | 5.3 | 377 |
| sum | 120 | 4.3 | 36 |
| fused | 119 | 1.7 | 14 |
| fusedCrossEntropyBwd | 1 | 0.2 | 250 |
| fusedCrossEntropyFwd | 1 | 0.1 | 139 |

### CPU Top Functions
| Self-time | % | Function | File |
|-----------|---|----------|------|
| 1406ms | 21.9% | createBuffer region | index.ts:2123 |
| 994ms | 15.5% | tensorFromArray | index.ts:3422 |
| 795ms | 12.4% | processImmediate | timers |
| 591ms | 9.2% | loadGPT2Weights | loader.ts:276 |
| 364ms | 5.7% | buffer setup | index.ts:1528 |
| 242ms | 3.8% | tensorFromArray | engine.ts:745 |
| 240ms | 3.7% | GC | (native) |

### Wall Clock
| Step | Total(ms) | Fwd | Bwd | Opt | Cleanup |
|------|-----------|-----|-----|-----|---------|
| 2 | 275 | 68 | 147 | 1 | 60 |
| 3 | 275 | 65 | 149 | 1 | 61 |
| 4 | 264 | 60 | 147 | 1 | 56 |
| **Avg** | **272** | **64** | **147** | **1** | **59** |
