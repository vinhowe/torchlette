/**
 * Fused RoPE (Rotary Position Embedding) Kernel
 *
 * Rotates pairs of features in Q/K by position-dependent angles.
 *
 * Half-split convention (GPT-NeoX / Llama):
 *   input:  qk [B, H, S, D]      (contiguous, D even)
 *   cos:    [S, D/2]
 *   sin:    [S, D/2]
 *   output: out [B, H, S, D]
 *
 *   first  half (d in [0,     D/2)): out[d] = qk[d] * cos - qk[d+D/2] * sin
 *   second half (d in [D/2, D    )): out[d] = qk[d] * cos + qk[d-D/2] * sin
 *
 * Backward is the same kernel with `sin_scale=-1`, since the inverse
 * rotation has cos unchanged and sin negated.
 *
 * Single dispatch: one thread per output element.
 */

import { allocateOutputBuffer } from "./buffer-arena";
import type { GPUBuffer } from "./gpu-types";
import { createTileKernelDispatcher } from "./tile-dispatch";
import { elementwiseKernel } from "./tile-ir";
import { onTeardown } from "./webgpu-state";

const ropeSpec = elementwiseKernel({
  name: "ropeApply",
  bindings: {
    input: { storage: "read", type: "f32" },
    cos_table: { storage: "read", type: "f32" },
    sin_table: { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { seq_len: "u32", head_dim: "u32", sin_scale: "f32" },
  sizeUniform: "total",
  kernel(ctx, idx) {
    const S = ctx.uniform("seq_len");
    const D = ctx.uniform("head_dim");
    const half = ctx.emitLet("half", D.div(ctx.u32(2)));
    const sinScale = ctx.uniform("sin_scale");

    // Decompose flat idx → (outer, s, d) where outer = b*H
    const d = ctx.emitLet("d", idx.mod(D));
    const stripe = ctx.emitLet("stripe", idx.div(D));
    const s = ctx.emitLet("s", stripe.mod(S));

    // d_half = d < half ? d : d - half;  side = (d >= half) ? 1 : 0
    const isSecond = ctx.emitLet("is_second", d.ge(half));
    const dHalf = ctx.emitLet("d_half", isSecond.select(d.sub(half), d));

    // Partner index in flat buffer: same (outer, s) but with d flipped
    // across halves. Offset in elements = ±half.
    const partnerIdx = ctx.emitLet(
      "partner_idx",
      isSecond.select(idx.sub(half), idx.add(half)),
    );

    // Load inputs
    const qi = ctx.emitLet("qi", ctx.load("input", idx));
    const qp = ctx.emitLet("qp", ctx.load("input", partnerIdx));
    const cosIdx = ctx.emitLet("cos_idx", s.mul(half).add(dHalf));
    const cosV = ctx.emitLet("cos_v", ctx.load("cos_table", cosIdx));
    const sinV = ctx.emitLet(
      "sin_v",
      ctx.load("sin_table", cosIdx).mul(sinScale),
    );

    // sign = +1 for second half, -1 for first half (multiplies partner*sin)
    const sign = ctx.emitLet(
      "sign",
      isSecond.select(ctx.f32(1.0), ctx.f32(-1.0)),
    );

    ctx.emitStore("output", idx, qi.mul(cosV).add(sign.mul(qp).mul(sinV)));
  },
});

const ropeTileKernel = createTileKernelDispatcher(ropeSpec);

/**
 * Dispatch fused RoPE kernel.
 * @param qkBuffer   [B, H, S, D] f32 input
 * @param cosBuffer  [S, D/2] f32 cos table
 * @param sinBuffer  [S, D/2] f32 sin table
 * @param total      B*H*S*D  (total element count)
 * @param seqLen     S
 * @param headDim    D
 * @param sinScale   +1 for forward, -1 for backward
 */
export function dispatchRoPE(
  qkBuffer: GPUBuffer,
  cosBuffer: GPUBuffer,
  sinBuffer: GPUBuffer,
  total: number,
  seqLen: number,
  headDim: number,
  sinScale: number,
): GPUBuffer {
  const outBuf = allocateOutputBuffer(total * 4);
  ropeTileKernel.dispatch(
    {
      input: qkBuffer,
      cos_table: cosBuffer,
      sin_table: sinBuffer,
      output: outBuf,
    },
    { total, seq_len: seqLen, head_dim: headDim, sin_scale: sinScale },
  );
  return outBuf;
}

export function resetRoPEKernelState(): void {
  ropeTileKernel.reset();
}
onTeardown(resetRoPEKernelState);
