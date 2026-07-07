# GPT-2 Finetuning Implementation Plan

This document tracks what's needed to finetune GPT-2 small (124M params) in WebGPU with all optimizations: compiled mode, fusion, memory planning, DCE, CSE, and checkpointing.

## Current Status

**Infrastructure: ~97% complete**
**Neural Network Ops: ~85% complete**

Sprint 3 complete! All ops needed for GPT-2 forward pass now implemented except `dropout` and `cross_entropy`.

---

## Already Implemented

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **Compiled mode** | ✅ | `src/engine/compiled-region.ts` | `compile()` with lazy execution |
| **Fusion** | ✅ | `src/engine/fusion-detect.ts` | `enableFusion: true` in RuntimeEngine |
| **CSE** | ✅ | `src/engine/ir-optimize.ts` | Common subexpression elimination |
| **DCE** | ✅ | `src/engine/ir-optimize.ts` | Dead code elimination |
| **Memory planning** | ✅ | `src/engine/memory-planning.ts` | `enableMemoryPlanning: true` - buffer reuse |
| **Checkpointing** | ✅ | `src/engine/checkpoint.ts` | §10 - Purity fences, replay support |
| **Autograd** | ✅ | `src/frontend.ts` | Forward+backward with saved tensors |
| **Optimizers** | ✅ | `src/optim/` | SGD, Adam |
| **Matmul** | ✅ | `src/backend/webgpu/matmul/` | Tiled with f16 support, ND batching |
| **Embedding lookup** | ✅ | `gather()` | With autograd |
| **Basic ops** | ✅ | Frontend | add, sub, mul, div, sqrt, relu |
| **View ops** | ✅ | Frontend | transpose, permute, reshape, expand |
| **Reductions** | ✅ | Frontend | sum, mean (with dim) |
| **RNG** | ✅ | `src/engine/rng.ts` | Keyed randomness for reproducibility |
| **AMP** | ✅ | `src/engine/amp.ts` | `autocast()` for mixed precision |
| **Dtype casting** | ✅ | Frontend | `toDtype()`, `half()`, `float()`, `int()` |

---

## Missing for GPT-2

### Phase 1: Expose Existing Fusion Ops ✅ COMPLETE

These ops already exist in `src/backend/webgpu/fusion-codegen.ts` and are now exposed:

| Op | Priority | Backend | Frontend | Autograd | Notes |
|----|----------|---------|----------|----------|-------|
| `exp` | CRITICAL | ✅ | ✅ | ✅ | `exp(x)`, grad = exp(x) |
| `log` | CRITICAL | ✅ | ✅ | ✅ | `log(x)`, grad = 1/x |
| `neg` | HIGH | ✅ | ✅ | ✅ | `-x`, grad = -1 |
| `abs` | HIGH | ✅ | ✅ | ✅ | `|x|`, grad = sign(x) |
| `gelu` | HIGH | ✅ | ✅ | ✅ | Approximation formula |
| `tanh` | HIGH | ✅ | ✅ | ✅ | `tanh(x)`, grad = 1 - tanh²(x) |
| `sigmoid` | MEDIUM | ✅ | ✅ | ✅ | `1/(1+exp(-x))`, grad = sig*(1-sig) |
| `silu` | MEDIUM | ✅ | ✅ | ✅ | `x * sigmoid(x)` |

**Test**: `test/unary-ops.spec.ts` - 16 tests passing

### Phase 2: Core Transformer Ops (MEDIUM effort)

| Op | Priority | Effort | Status | Notes |
|----|----------|--------|--------|-------|
| `max(dim)` | CRITICAL | Medium | ✅ | Reduction along dimension |
| `softmax(dim)` | CRITICAL | Medium | ✅ | With autograd support |
| `layernorm` | CRITICAL | Medium | ❌ | `(x - mean) / sqrt(var + eps) * gamma + beta` |

**Test**: `test/reduction-ops.spec.ts` - 11 tests passing

**softmax implementation**:
```typescript
function softmax(x: Tensor, dim: number): Tensor {
  const maxVal = x.max(dim, { keepdim: true });  // numerical stability
  const shifted = x.sub(maxVal);
  const exps = shifted.exp();
  const sumExps = exps.sum(dim, { keepdim: true });
  return exps.div(sumExps);
}
```

**layernorm implementation**:
```typescript
function layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
  const mean = x.mean(-1, { keepdim: true });
  const variance = x.sub(mean).pow(2).mean(-1, { keepdim: true });
  const normalized = x.sub(mean).div(variance.add(eps).sqrt());
  return normalized.mul(weight).add(bias);
}
```

### Phase 3: Comparison + ArgMax/ArgMin + LayerNorm ✅ COMPLETE

| Op | Priority | Backend | Frontend | Autograd | Notes |
|----|----------|---------|----------|----------|-------|
| `gt/lt/ge/le/eq/ne` | HIGH | ✅ | ✅ | N/A | Comparison ops for masks |
| `argmax/argmin` | HIGH | ✅ | ✅ | N/A | Index of max/min (shares reduction core with max) |
| `layernorm` | CRITICAL | ✅ | ✅ | ✅ | `(x - mean) / sqrt(var + eps) * gamma + beta` |

**Test**: `test/sprint3-ops.spec.ts` - 23 tests passing

**relu/sqrt backward** now uses comparison ops (gt) instead of toArray().

### Phase 4: Training Ops (MEDIUM effort)

| Op | Priority | Effort | Description |
|----|----------|--------|-------------|
| `dropout` | MEDIUM | Medium | Random mask with scale during training |
| `cross_entropy` | CRITICAL | Medium | `-log(softmax(logits)[target])` |

### Phase 5: Convenience Ops (LOW priority)

| Op | Priority | Effort | Description |
|----|----------|--------|-------------|
| `slice` | MEDIUM | Medium | Extract subtensor along dims |
| `split` | MEDIUM | Low | Split tensor into chunks (uses slice) |
| `cat/concat` | MEDIUM | Medium | Concatenate tensors |
| `argmax` | LOW | Medium | Index of maximum value |
| `pow` | LOW | Low | Already in fusion, needs frontend |

---

## GPT-2 Forward Pass Requirements

```typescript
// Pseudocode showing which ops are needed

// Token + Position Embedding
const embed = weights.wte.gather(tokens, { dim: 0 });      // ✅ EXISTS
const pos_embed = weights.wpe.gather(positions, { dim: 0 }); // ✅ EXISTS
let x = embed.add(pos_embed);                               // ✅ EXISTS

for (const block of blocks) {
  // === Attention Block ===
  const ln1 = layernorm(x, block.ln1_weight, block.ln1_bias); // ❌ NEED layernorm
  const qkv = ln1.matmul(block.c_attn);                       // ✅ EXISTS
  const [q, k, v] = split(qkv, 3, dim=-1);                    // ❌ NEED split

  // Attention scores
  const scores = q.matmul(k.transpose(-2, -1));               // ✅ EXISTS
  const scaled = scores.div(Math.sqrt(d_k));                  // ✅ EXISTS
  const attn_weights = softmax(scaled, dim=-1);               // ❌ NEED softmax
  const attn_out = attn_weights.matmul(v);                    // ✅ EXISTS

  const proj = attn_out.matmul(block.c_proj);                 // ✅ EXISTS
  x = x.add(proj);                                            // ✅ EXISTS (residual)

  // === FFN Block ===
  const ln2 = layernorm(x, block.ln2_weight, block.ln2_bias); // ❌ NEED layernorm
  const h = ln2.matmul(block.mlp_fc);                         // ✅ EXISTS
  const h_act = h.gelu();                                     // ❌ NEED gelu
  const ffn_out = h_act.matmul(block.mlp_proj);               // ✅ EXISTS
  x = x.add(ffn_out);                                         // ✅ EXISTS (residual)
}

// Final layer norm + output projection
const ln_f = layernorm(x, ln_f_weight, ln_f_bias);           // ❌ NEED layernorm
const logits = ln_f.matmul(weights.wte.transpose(0, 1));     // ✅ EXISTS

// Loss
const loss = cross_entropy(logits, targets);                  // ❌ NEED cross_entropy
```

---

## Implementation Order

### Sprint 1: Expose Fusion Ops ✅ COMPLETE
1. ✅ Added `exp`, `log`, `neg`, `abs` to runtime + frontend with autograd
2. ✅ Added `gelu`, `tanh`, `sigmoid`, `silu` to runtime + frontend with autograd
3. ✅ Added tests (`test/unary-ops.spec.ts` - 16 tests)

### Sprint 2: Reductions + Softmax ✅ COMPLETE
1. ✅ Added `max(dim)` reduction to both CPU and WebGPU backends
2. ✅ Implemented `softmax(dim)` using exp, max, sum
3. ✅ Added autograd for softmax
4. ✅ Added tests (`test/reduction-ops.spec.ts` - 11 tests)

### Sprint 3: LayerNorm + Comparison Ops + ArgMax/ArgMin ✅ COMPLETE
1. ✅ Added comparison ops (gt, lt, ge, le, eq, ne) to CPU and WebGPU backends
2. ✅ Added argmax/argmin reduction ops (shares reduction core with max)
3. ✅ Fixed relu/sqrt backward to use comparison ops instead of toArray()
4. ✅ Implemented layernorm as composite op with autograd
5. ✅ Added tests (`test/sprint3-ops.spec.ts` - 23 tests)

### Sprint 4: Training Ops (Est. 3-4 hours)
1. Implement `dropout` with RNG integration
2. Implement `cross_entropy` using softmax + log
3. Add tests

### Sprint 5: Convenience + Integration (Est. 2-3 hours)
1. Add `slice`, `split`, `cat` if needed
2. End-to-end GPT-2 forward pass test
3. End-to-end training loop test
4. Performance benchmarks

---

## Success Criteria

### Minimum Viable
- [ ] Forward pass through GPT-2 small works
- [ ] Backward pass computes gradients
- [ ] One training step completes without error
- [ ] Uses WebGPU acceleration

### Full Implementation
- [ ] All ops have autograd support
- [ ] Fusion works for transformer blocks
- [ ] Memory planning reduces peak memory
- [ ] Checkpointing enables gradient checkpointing for memory savings
- [ ] Training converges on small dataset

---

## Files to Modify

### Phase 1 (Fusion ops)
- `src/backend/webgpu/index.ts` - Add WebGPU kernels (may be no-op if using fusion)
- `src/backend/cpu/numeric.ts` - Add CPU implementations
- `src/engine/lazy.ts` - Add op dispatch cases
- `src/runtime/engine.ts` - Add RuntimeEngine methods
- `src/frontend.ts` - Add Tensor methods + autograd

### Phase 2 (Reductions)
- `src/backend/webgpu/index.ts` - Add max reduction kernel
- `src/backend/cpu/numeric.ts` - Add max implementation
- `src/frontend.ts` - Add softmax composite op

### Phase 3 (Comparison ops)
- `src/backend/webgpu/index.ts` - Add comparison kernels
- `src/frontend.ts` - Fix relu backward, add layernorm

### Phase 4 (Training ops)
- `src/frontend.ts` - Add dropout, cross_entropy

---

## References

- GPT-2 Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- GPT-2 Small: 124M params, 12 layers, 768 hidden, 12 heads
- Attention: `softmax(QK^T / sqrt(d_k)) * V`
- LayerNorm: `(x - mean) / sqrt(var + eps) * gamma + beta`
- GELU: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
