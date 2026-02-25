/**
 * Quick test: compile a LayerNorm forward kernel with the tile IR
 * and print the generated WGSL.
 */

import type { TileKernelSpec } from "../src/backend/webgpu/tile-ir";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";

const layerNormFwd: TileKernelSpec = {
  name: "layerNormFwd",
  workgroupSize: 256,
  bindings: {
    x:      { storage: "read",       type: "f32" },
    weight: { storage: "read",       type: "f32" },
    bias:   { storage: "read",       type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: {
    num_rows:    "u32",
    feature_dim: "u32",
    eps:         "f32",
  },
  grid: (u) => [u.num_rows],

  kernel(ctx) {
    const row = ctx.programId(0);
    const D   = ctx.uniform("feature_dim");
    const base = row.mul(D);
    const offs = base.add(ctx.blockRange(D));

    // Phase 0: load + reduce to mean
    const xVals = ctx.load("x", offs);
    const mean  = ctx.reduce(xVals, "sum").div(D.toF32());

    // Phase 1: compute variance and inv_std
    const diff    = xVals.sub(mean);
    const variance = ctx.reduce(diff.mul(diff), "sum").div(D.toF32());
    const invStd  = variance.add(ctx.uniform("eps").toF32()).rsqrt();

    // Phase 2: normalize + affine + store
    const normalized = diff.mul(invStd);
    const w = ctx.load("weight", ctx.blockRange(D));
    const b = ctx.load("bias", ctx.blockRange(D));
    ctx.store("output", offs, normalized.mul(w).add(b));
  },
};

console.log("=== Generated WGSL for LayerNorm Forward ===\n");
const wgsl = compileTileKernel(layerNormFwd);
console.log(wgsl);

console.log("\n=== Hand-written reference ===\n");
console.log(`struct LNConfig {
  num_rows: u32,
  feature_dim: u32,
  eps: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> config: LNConfig;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let tid = lid.x;
  let D = config.feature_dim;
  let base = row * D;

  // mean reduction
  {wgslReduce sum x[base+i] → mean, transform: _ / f32(D)}

  // variance reduction
  {wgslReduce sum (x[base+i]-mean)² → inv_std, transform: inverseSqrt(_ / f32(D) + config.eps)}

  // Normalize + affine transform
  for (var i = tid; i < D; i += 256u) {
    let normalized = (x[base + i] - mean) * inv_std;
    output[base + i] = normalized * weight[i] + bias[i];
  }
}`);

process.exit(0);
