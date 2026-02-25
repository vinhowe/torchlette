/**
 * Fused Adam/AdamW Optimizer Kernel
 *
 * A single WGSL compute shader that updates param, m, v in-place in one
 * dispatch per parameter. Supports both Adam (L2 decay on gradient) and
 * AdamW (decoupled decay on parameter). grad is read-only; param/m/v are
 * read_write (no separate output buffers needed).
 *
 * Handles large buffers (>maxStorageBufferBindingSize) via chunked dispatch.
 */

import {
  dispatchComputePass,
  getWebGPUDevice,
  getMaxStorageBufferBindingSize,
  trackSharedEncoderWrite,
  createParamsBuffer,
  releaseParamsBuffer,
  isF16Supported,
  allocateOutputBuffer,
  cachedCreateBindGroup,
} from "./index";
import type { AdamStepConfig } from "../types";
import { profileSubOpBegin, profileSubOpEnd } from "./profiler";
import type { GPUBuffer, GPUDevice, GPUComputePipeline } from "./gpu-types";

const WORKGROUP_SIZE = 256;
const MAX_WORKGROUPS_PER_DIM = 65535;

// ============================================================================
// Pipeline Cache
// ============================================================================

const pipelineCache = new Map<string, GPUComputePipeline>();

function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  code: string,
): GPUComputePipeline {
  let pipeline = pipelineCache.get(key);
  if (!pipeline) {
    const module = device.createShaderModule({ code });
    pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    pipelineCache.set(key, pipeline);
  }
  return pipeline;
}

// ============================================================================
// WGSL Shader
// ============================================================================

function adamStepShader(use2D: boolean, gridSizeX: number, useVec4 = false): string {
  const idxCompute = use2D
    ? `let flat_id = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let flat_id = gid.x;`;

  if (useVec4) {
    return `
struct AdamConfig {
  beta1: f32,
  beta2: f32,
  step_size: f32,
  eps: f32,
  weight_decay: f32,
  lr_times_wd: f32,
  decoupled_wd: u32,
  num_elements: u32,
};

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamConfig;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  let base = flat_id * 4u;
  if (base >= config.num_elements) { return; }

  var g = vec4<f32>(grad[base], grad[base+1u], grad[base+2u], grad[base+3u]);
  let p = vec4<f32>(param[base], param[base+1u], param[base+2u], param[base+3u]);
  let m_old = vec4<f32>(m[base], m[base+1u], m[base+2u], m[base+3u]);
  let v_old = vec4<f32>(v[base], v[base+1u], v[base+2u], v[base+3u]);

  // L2 weight decay (Adam): grad += wd * param
  if (config.decoupled_wd == 0u && config.weight_decay > 0.0) {
    g = g + config.weight_decay * p;
  }

  // Moment updates
  let m_new = config.beta1 * m_old + (1.0 - config.beta1) * g;
  let v_new = config.beta2 * v_old + (1.0 - config.beta2) * g * g;

  // Parameter update
  var p_new = p - config.step_size * m_new / (sqrt(v_new) + vec4<f32>(config.eps));

  // Decoupled weight decay (AdamW): param -= lr * wd * param
  if (config.decoupled_wd == 1u) {
    p_new = p_new - config.lr_times_wd * p;
  }

  param[base] = p_new.x; param[base+1u] = p_new.y;
  param[base+2u] = p_new.z; param[base+3u] = p_new.w;
  m[base] = m_new.x; m[base+1u] = m_new.y;
  m[base+2u] = m_new.z; m[base+3u] = m_new.w;
  v[base] = v_new.x; v[base+1u] = v_new.y;
  v[base+2u] = v_new.z; v[base+3u] = v_new.w;
}
`;
  }

  // Scalar path
  const scalarIdx = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  return `
struct AdamConfig {
  beta1: f32,
  beta2: f32,
  step_size: f32,
  eps: f32,
  weight_decay: f32,
  lr_times_wd: f32,
  decoupled_wd: u32,
  num_elements: u32,
};

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamConfig;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${scalarIdx}
  if (idx >= config.num_elements) { return; }

  var g = grad[idx];
  let p = param[idx];

  // L2 weight decay (Adam): grad += wd * param
  if (config.decoupled_wd == 0u && config.weight_decay > 0.0) {
    g = g + config.weight_decay * p;
  }

  // Moment updates
  let m_new = config.beta1 * m[idx] + (1.0 - config.beta1) * g;
  let v_new = config.beta2 * v[idx] + (1.0 - config.beta2) * g * g;

  // Parameter update
  var p_new = p - config.step_size * m_new / (sqrt(v_new) + config.eps);

  // Decoupled weight decay (AdamW): param -= lr * wd * param
  if (config.decoupled_wd == 1u) {
    p_new = p_new - config.lr_times_wd * p;
  }

  param[idx] = p_new;
  m[idx] = m_new;
  v[idx] = v_new;
}
`;
}

/**
 * F16 variant of the Adam shader. Identical to adamStepShader but with an
 * additional binding 5 (param_f16) that writes the updated param as f16.
 * This allows the forward pass to skip standalone f32→f16 cast ops for AMP weights.
 */
function adamStepShaderF16(use2D: boolean, gridSizeX: number, useVec4 = false): string {
  const idxCompute = use2D
    ? `let flat_id = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let flat_id = gid.x;`;

  if (useVec4) {
    return `
enable f16;

struct AdamConfig {
  beta1: f32,
  beta2: f32,
  step_size: f32,
  eps: f32,
  weight_decay: f32,
  lr_times_wd: f32,
  decoupled_wd: u32,
  num_elements: u32,
};

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamConfig;
@group(0) @binding(5) var<storage, read_write> param_f16: array<f16>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  let base = flat_id * 4u;
  if (base >= config.num_elements) { return; }

  var g = vec4<f32>(grad[base], grad[base+1u], grad[base+2u], grad[base+3u]);
  let p = vec4<f32>(param[base], param[base+1u], param[base+2u], param[base+3u]);
  let m_old = vec4<f32>(m[base], m[base+1u], m[base+2u], m[base+3u]);
  let v_old = vec4<f32>(v[base], v[base+1u], v[base+2u], v[base+3u]);

  // L2 weight decay (Adam): grad += wd * param
  if (config.decoupled_wd == 0u && config.weight_decay > 0.0) {
    g = g + config.weight_decay * p;
  }

  // Moment updates
  let m_new = config.beta1 * m_old + (1.0 - config.beta1) * g;
  let v_new = config.beta2 * v_old + (1.0 - config.beta2) * g * g;

  // Parameter update
  var p_new = p - config.step_size * m_new / (sqrt(v_new) + vec4<f32>(config.eps));

  // Decoupled weight decay (AdamW): param -= lr * wd * param
  if (config.decoupled_wd == 1u) {
    p_new = p_new - config.lr_times_wd * p;
  }

  param[base] = p_new.x; param[base+1u] = p_new.y;
  param[base+2u] = p_new.z; param[base+3u] = p_new.w;
  m[base] = m_new.x; m[base+1u] = m_new.y;
  m[base+2u] = m_new.z; m[base+3u] = m_new.w;
  v[base] = v_new.x; v[base+1u] = v_new.y;
  v[base+2u] = v_new.z; v[base+3u] = v_new.w;
  let p_f16 = vec4<f16>(p_new);
  param_f16[base] = p_f16.x; param_f16[base+1u] = p_f16.y;
  param_f16[base+2u] = p_f16.z; param_f16[base+3u] = p_f16.w;
}
`;
  }

  // Scalar path
  const scalarIdx = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  return `
enable f16;

struct AdamConfig {
  beta1: f32,
  beta2: f32,
  step_size: f32,
  eps: f32,
  weight_decay: f32,
  lr_times_wd: f32,
  decoupled_wd: u32,
  num_elements: u32,
};

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamConfig;
@group(0) @binding(5) var<storage, read_write> param_f16: array<f16>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${scalarIdx}
  if (idx >= config.num_elements) { return; }

  var g = grad[idx];
  let p = param[idx];

  // L2 weight decay (Adam): grad += wd * param
  if (config.decoupled_wd == 0u && config.weight_decay > 0.0) {
    g = g + config.weight_decay * p;
  }

  // Moment updates
  let m_new = config.beta1 * m[idx] + (1.0 - config.beta1) * g;
  let v_new = config.beta2 * v[idx] + (1.0 - config.beta2) * g * g;

  // Parameter update
  var p_new = p - config.step_size * m_new / (sqrt(v_new) + config.eps);

  // Decoupled weight decay (AdamW): param -= lr * wd * param
  if (config.decoupled_wd == 1u) {
    p_new = p_new - config.lr_times_wd * p;
  }

  param[idx] = p_new;
  m[idx] = m_new;
  v[idx] = v_new;
  param_f16[idx] = f16(p_new);
}
`;
}

/**
 * Unscale variant of the Adam shader. Adds gradient unscaling with inv_scale
 * and inf/NaN detection via a shared atomic inf_flag buffer. Non-finite
 * gradients are zeroed before the Adam update.
 *
 * Binding layout: 0-4 same as base, 5 = inf_flag (atomic u32).
 * Config struct extended with inv_scale field (offset 32).
 */
function adamStepShaderUnscale(use2D: boolean, gridSizeX: number, useVec4 = false): string {
  const idxCompute = use2D
    ? `let flat_id = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let flat_id = gid.x;`;

  if (useVec4) {
    return `
struct AdamConfig {
  beta1: f32,
  beta2: f32,
  step_size: f32,
  eps: f32,
  weight_decay: f32,
  lr_times_wd: f32,
  decoupled_wd: u32,
  num_elements: u32,
  inv_scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamConfig;
@group(0) @binding(5) var<storage, read_write> inf_flag: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  let base = flat_id * 4u;
  if (base >= config.num_elements) { return; }

  // Unscale gradient (vec4)
  var g = vec4<f32>(grad[base], grad[base+1u], grad[base+2u], grad[base+3u]) * config.inv_scale;

  // Check finite via bit pattern: inf/NaN have all exponent bits set
  let bits = bitcast<vec4<u32>>(g);
  let exponents = (bits >> vec4<u32>(23u)) & vec4<u32>(0xFFu);
  let is_inf = exponents == vec4<u32>(0xFFu);
  if (is_inf.x || is_inf.y || is_inf.z || is_inf.w) {
    atomicMax(&inf_flag[0], 1065353216u);
  }
  g = select(g, vec4<f32>(0.0), is_inf);

  let p = vec4<f32>(param[base], param[base+1u], param[base+2u], param[base+3u]);
  let m_old = vec4<f32>(m[base], m[base+1u], m[base+2u], m[base+3u]);
  let v_old = vec4<f32>(v[base], v[base+1u], v[base+2u], v[base+3u]);

  // L2 weight decay (Adam): grad += wd * param
  if (config.decoupled_wd == 0u && config.weight_decay > 0.0) {
    g = g + config.weight_decay * p;
  }

  // Moment updates
  let m_new = config.beta1 * m_old + (1.0 - config.beta1) * g;
  let v_new = config.beta2 * v_old + (1.0 - config.beta2) * g * g;

  // Parameter update
  var p_new = p - config.step_size * m_new / (sqrt(v_new) + vec4<f32>(config.eps));

  // Decoupled weight decay (AdamW): param -= lr * wd * param
  if (config.decoupled_wd == 1u) {
    p_new = p_new - config.lr_times_wd * p;
  }

  param[base] = p_new.x; param[base+1u] = p_new.y;
  param[base+2u] = p_new.z; param[base+3u] = p_new.w;
  m[base] = m_new.x; m[base+1u] = m_new.y;
  m[base+2u] = m_new.z; m[base+3u] = m_new.w;
  v[base] = v_new.x; v[base+1u] = v_new.y;
  v[base+2u] = v_new.z; v[base+3u] = v_new.w;
}
`;
  }

  // Scalar path
  const scalarIdx = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  return `
struct AdamConfig {
  beta1: f32,
  beta2: f32,
  step_size: f32,
  eps: f32,
  weight_decay: f32,
  lr_times_wd: f32,
  decoupled_wd: u32,
  num_elements: u32,
  inv_scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamConfig;
@group(0) @binding(5) var<storage, read_write> inf_flag: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${scalarIdx}
  if (idx >= config.num_elements) { return; }

  // Unscale gradient
  var g = grad[idx] * config.inv_scale;

  // Check finite via bit pattern: inf/NaN have all exponent bits set
  let bits = bitcast<u32>(g);
  let exponent = (bits >> 23u) & 0xFFu;
  if (exponent == 0xFFu) {
    // Non-finite: set inf flag and zero gradient
    atomicMax(&inf_flag[0], 1065353216u); // bitcast<u32>(1.0f)
    g = 0.0;
  }

  let p = param[idx];

  // L2 weight decay (Adam): grad += wd * param
  if (config.decoupled_wd == 0u && config.weight_decay > 0.0) {
    g = g + config.weight_decay * p;
  }

  // Moment updates
  let m_new = config.beta1 * m[idx] + (1.0 - config.beta1) * g;
  let v_new = config.beta2 * v[idx] + (1.0 - config.beta2) * g * g;

  // Parameter update
  var p_new = p - config.step_size * m_new / (sqrt(v_new) + config.eps);

  // Decoupled weight decay (AdamW): param -= lr * wd * param
  if (config.decoupled_wd == 1u) {
    p_new = p_new - config.lr_times_wd * p;
  }

  param[idx] = p_new;
  m[idx] = m_new;
  v[idx] = v_new;
}
`;
}

/**
 * F16 + Unscale variant. Combines f16 output, gradient unscaling, and inf detection.
 * Binding layout: 0-4 same as base, 5 = param_f16, 6 = inf_flag.
 */
function adamStepShaderF16Unscale(use2D: boolean, gridSizeX: number, useVec4 = false): string {
  const idxCompute = use2D
    ? `let flat_id = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let flat_id = gid.x;`;

  if (useVec4) {
    return `
enable f16;

struct AdamConfig {
  beta1: f32,
  beta2: f32,
  step_size: f32,
  eps: f32,
  weight_decay: f32,
  lr_times_wd: f32,
  decoupled_wd: u32,
  num_elements: u32,
  inv_scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamConfig;
@group(0) @binding(5) var<storage, read_write> param_f16: array<f16>;
@group(0) @binding(6) var<storage, read_write> inf_flag: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  let base = flat_id * 4u;
  if (base >= config.num_elements) { return; }

  // Unscale gradient (vec4)
  var g = vec4<f32>(grad[base], grad[base+1u], grad[base+2u], grad[base+3u]) * config.inv_scale;

  // Check finite via bit pattern
  let bits = bitcast<vec4<u32>>(g);
  let exponents = (bits >> vec4<u32>(23u)) & vec4<u32>(0xFFu);
  let is_inf = exponents == vec4<u32>(0xFFu);
  if (is_inf.x || is_inf.y || is_inf.z || is_inf.w) {
    atomicMax(&inf_flag[0], 1065353216u);
  }
  g = select(g, vec4<f32>(0.0), is_inf);

  let p = vec4<f32>(param[base], param[base+1u], param[base+2u], param[base+3u]);
  let m_old = vec4<f32>(m[base], m[base+1u], m[base+2u], m[base+3u]);
  let v_old = vec4<f32>(v[base], v[base+1u], v[base+2u], v[base+3u]);

  // L2 weight decay (Adam): grad += wd * param
  if (config.decoupled_wd == 0u && config.weight_decay > 0.0) {
    g = g + config.weight_decay * p;
  }

  // Moment updates
  let m_new = config.beta1 * m_old + (1.0 - config.beta1) * g;
  let v_new = config.beta2 * v_old + (1.0 - config.beta2) * g * g;

  // Parameter update
  var p_new = p - config.step_size * m_new / (sqrt(v_new) + vec4<f32>(config.eps));

  // Decoupled weight decay (AdamW): param -= lr * wd * param
  if (config.decoupled_wd == 1u) {
    p_new = p_new - config.lr_times_wd * p;
  }

  param[base] = p_new.x; param[base+1u] = p_new.y;
  param[base+2u] = p_new.z; param[base+3u] = p_new.w;
  m[base] = m_new.x; m[base+1u] = m_new.y;
  m[base+2u] = m_new.z; m[base+3u] = m_new.w;
  v[base] = v_new.x; v[base+1u] = v_new.y;
  v[base+2u] = v_new.z; v[base+3u] = v_new.w;
  let p_f16 = vec4<f16>(p_new);
  param_f16[base] = p_f16.x; param_f16[base+1u] = p_f16.y;
  param_f16[base+2u] = p_f16.z; param_f16[base+3u] = p_f16.w;
}
`;
  }

  // Scalar path
  const scalarIdx = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  return `
enable f16;

struct AdamConfig {
  beta1: f32,
  beta2: f32,
  step_size: f32,
  eps: f32,
  weight_decay: f32,
  lr_times_wd: f32,
  decoupled_wd: u32,
  num_elements: u32,
  inv_scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamConfig;
@group(0) @binding(5) var<storage, read_write> param_f16: array<f16>;
@group(0) @binding(6) var<storage, read_write> inf_flag: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${scalarIdx}
  if (idx >= config.num_elements) { return; }

  // Unscale gradient
  var g = grad[idx] * config.inv_scale;

  // Check finite via bit pattern
  let bits = bitcast<u32>(g);
  let exponent = (bits >> 23u) & 0xFFu;
  if (exponent == 0xFFu) {
    atomicMax(&inf_flag[0], 1065353216u);
    g = 0.0;
  }

  let p = param[idx];

  // L2 weight decay (Adam): grad += wd * param
  if (config.decoupled_wd == 0u && config.weight_decay > 0.0) {
    g = g + config.weight_decay * p;
  }

  // Moment updates
  let m_new = config.beta1 * m[idx] + (1.0 - config.beta1) * g;
  let v_new = config.beta2 * v[idx] + (1.0 - config.beta2) * g * g;

  // Parameter update
  var p_new = p - config.step_size * m_new / (sqrt(v_new) + config.eps);

  // Decoupled weight decay (AdamW): param -= lr * wd * param
  if (config.decoupled_wd == 1u) {
    p_new = p_new - config.lr_times_wd * p;
  }

  param[idx] = p_new;
  m[idx] = m_new;
  v[idx] = v_new;
  param_f16[idx] = f16(p_new);
}
`;
}

// ============================================================================
// Config Buffer
// ============================================================================

// Pre-allocated typed arrays for config buffer construction.
// Eliminates 228 short-lived allocations per step (76 calls × 3 arrays).
const _configData32 = new ArrayBuffer(32);
const _configF32_32 = new Float32Array(_configData32);
const _configU32_32 = new Uint32Array(_configData32);
const _configData48 = new ArrayBuffer(48);
const _configF32_48 = new Float32Array(_configData48);
const _configU32_48 = new Uint32Array(_configData48);

function createConfigBuffer(
  device: GPUDevice,
  config: AdamStepConfig,
  numElements: number,
  includeInvScale: boolean,
): GPUBuffer {
  if (includeInvScale) {
    // 12 x f32/u32 = 48 bytes (padded to 16-byte alignment)
    _configF32_48[0] = config.beta1;
    _configF32_48[1] = config.beta2;
    _configF32_48[2] = config.stepSize;
    _configF32_48[3] = config.eps;
    _configF32_48[4] = config.weightDecay;
    _configF32_48[5] = config.lrTimesWd;
    _configU32_48[6] = config.decoupledWd ? 1 : 0;
    _configU32_48[7] = numElements;
    _configF32_48[8] = config.invScale ?? 1.0;
    _configU32_48[9] = 0; // pad
    _configU32_48[10] = 0; // pad
    _configU32_48[11] = 0; // pad

    return createParamsBuffer(device, _configU32_48);
  }

  // 8 x f32/u32 = 32 bytes (original layout)
  _configF32_32[0] = config.beta1;
  _configF32_32[1] = config.beta2;
  _configF32_32[2] = config.stepSize;
  _configF32_32[3] = config.eps;
  _configF32_32[4] = config.weightDecay;
  _configF32_32[5] = config.lrTimesWd;
  _configU32_32[6] = config.decoupledWd ? 1 : 0;
  _configU32_32[7] = numElements;

  return createParamsBuffer(device, _configU32_32);
}

// ============================================================================
// Output Buffer Allocation
// ============================================================================

/**
 * Allocate an output buffer for Adam via the buffer pool.
 */
function allocateAdamOutputBuffer(sizeBytes: number): GPUBuffer {
  const buf = allocateOutputBuffer(sizeBytes);
  trackSharedEncoderWrite(buf);
  return buf;
}

// ============================================================================
// Dispatch
// ============================================================================

interface AdamStepResult {
  paramBuffer: GPUBuffer;
  mBuffer: GPUBuffer;
  vBuffer: GPUBuffer;
  paramF16Buffer?: GPUBuffer;
}

/**
 * Dispatch the fused Adam/AdamW kernel.
 *
 * Handles chunking for buffers larger than maxStorageBufferBindingSize.
 * When infFlagBuffer is provided, uses fused unscale+inf-check variants
 * that multiply grad by invScale and detect non-finite values.
 */
export function dispatchAdamStep(
  gradBuffer: GPUBuffer,
  paramBuffer: GPUBuffer,
  mBuffer: GPUBuffer,
  vBuffer: GPUBuffer,
  numElements: number,
  config: AdamStepConfig,
  emitF16 = false,
  infFlagBuffer: GPUBuffer | null = null,
): AdamStepResult {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device;

  // Only emit f16 if requested AND the device actually supports shader-f16
  const doF16 = emitF16 && isF16Supported();
  const doUnscale = infFlagBuffer !== null;

  const bytesPerElement = 4; // f32
  const f16BytesPerElement = 2;
  const totalBytes = numElements * bytesPerElement;
  const maxBindingSize = getMaxStorageBufferBindingSize();

  // Determine if chunking is needed
  const needsChunking = totalBytes > maxBindingSize;

  // Align chunk size for sub-range bindings.
  // When emitting f16, chunk alignment must satisfy both f32 (4B) and f16 (2B)
  // offset alignment requirements: offset must be a multiple of minAlignment (256).
  // For f32: 256/4 = 64 elements. For f16: 256/2 = 128 elements. Use the larger.
  const minAlignment = 256; // minStorageBufferOffsetAlignment
  const elementsPerAlignment = doF16
    ? minAlignment / f16BytesPerElement  // 128 — ensures f16 offsets are 256-aligned
    : minAlignment / bytesPerElement;    // 64
  const maxElementsPerChunk = Math.floor(maxBindingSize / bytesPerElement);
  const elementsPerChunk = needsChunking
    ? Math.floor(maxElementsPerChunk / elementsPerAlignment) *
      elementsPerAlignment
    : numElements;
  const numChunks = Math.ceil(numElements / elementsPerChunk);

  // In-place: param/m/v are read_write, no output buffer allocation needed.
  // Track ALL input buffers (including grad, which is read-only) in the write
  // set BEFORE any allocation, so the pool won't hand back these buffers for
  // the f16 output allocation. Without this, the pool may return grad's buffer
  // for the f16 output, causing a read/read_write conflict on the same buffer.
  let _st = profileSubOpBegin();
  trackSharedEncoderWrite(gradBuffer);
  trackSharedEncoderWrite(paramBuffer);
  trackSharedEncoderWrite(mBuffer);
  trackSharedEncoderWrite(vBuffer);

  // Only allocate f16 output buffer (different size).
  const totalF16Bytes = numElements * f16BytesPerElement;
  const paramF16Out = doF16
    ? allocateAdamOutputBuffer(totalF16Bytes)
    : null;
  profileSubOpEnd("adam.allocBufs", _st);

  // Vec4 coalescing: use when total elements divisible by 4
  // (elementsPerChunk is aligned to 64 or 128 elements, both multiples of 4)
  const useVec4 = numElements % 4 === 0;

  // Determine dispatch dimensions (vec4 processes 4 elements per thread)
  const workItemsPerChunk = useVec4 ? elementsPerChunk / 4 : elementsPerChunk;
  const maxWorkgroups = Math.ceil(workItemsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  // Select shader variant based on f16, unscale, and vec4 flags
  _st = profileSubOpBegin();
  const vec4Tag = useVec4 ? "Vec4" : "";
  const dimTag = use2D ? `2d:${gridSizeX}` : "1d";
  let key: string;
  let code: string;
  if (doF16 && doUnscale) {
    key = `adamStepF16Unscale${vec4Tag}:${dimTag}`;
    code = adamStepShaderF16Unscale(use2D, gridSizeX, useVec4);
  } else if (doF16) {
    key = `adamStepF16${vec4Tag}:${dimTag}`;
    code = adamStepShaderF16(use2D, gridSizeX, useVec4);
  } else if (doUnscale) {
    key = `adamStepUnscale${vec4Tag}:${dimTag}`;
    code = adamStepShaderUnscale(use2D, gridSizeX, useVec4);
  } else {
    key = `adamStep${vec4Tag}:${dimTag}`;
    code = adamStepShader(use2D, gridSizeX, useVec4);
  }
  const pipeline = getOrCreatePipeline(device, key, code);
  profileSubOpEnd("adam.pipeline", _st);

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, numElements);
    const chunkSize = chunkEnd - chunkStart;

    // Create config buffer for this chunk
    _st = profileSubOpBegin();
    const configBuf = createConfigBuffer(device, config, chunkSize, doUnscale);
    profileSubOpEnd("adam.configBuf", _st);

    // Build bind group entries with sub-range bindings for chunked access
    _st = profileSubOpBegin();
    const mkBinding = (buf: GPUBuffer, bpe = bytesPerElement) =>
      needsChunking
        ? { buffer: buf, offset: chunkStart * bpe, size: chunkSize * bpe }
        : { buffer: buf };

    // In-place layout: grad(read), param(rw), m(rw), v(rw), config(uniform)
    // Optional: param_f16(rw), inf_flag(rw)
    const entries: Array<{
      binding: number;
      resource: { buffer: GPUBuffer; offset?: number; size?: number };
    }> = [
      { binding: 0, resource: mkBinding(gradBuffer) },
      { binding: 1, resource: mkBinding(paramBuffer) },
      { binding: 2, resource: mkBinding(mBuffer) },
      { binding: 3, resource: mkBinding(vBuffer) },
      { binding: 4, resource: { buffer: configBuf } },
    ];

    let nextBinding = 5;
    if (doF16 && paramF16Out) {
      entries.push({
        binding: nextBinding++,
        resource: mkBinding(paramF16Out, f16BytesPerElement),
      });
    }

    if (doUnscale) {
      entries.push({
        binding: nextBinding++,
        resource: { buffer: infFlagBuffer! }, // always full 4 bytes, not chunked
      });
    }

    profileSubOpEnd("adam.entries", _st);

    _st = profileSubOpBegin();
    let bindGroup: any;
    if (needsChunking) {
      // Chunked: entries have offset/size, cannot cache
      bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
    } else {
      // Simple bindings: use cache
      const bgBuffers: GPUBuffer[] = [gradBuffer, paramBuffer, mBuffer, vBuffer, configBuf];
      if (doF16 && paramF16Out) bgBuffers.push(paramF16Out);
      if (doUnscale) bgBuffers.push(infFlagBuffer!);
      bindGroup = cachedCreateBindGroup(device, pipeline, bgBuffers);
    }
    profileSubOpEnd("adam.bindGroup", _st);

    const chunkWorkItems = useVec4 ? chunkSize / 4 : chunkSize;
    const chunkWorkgroups = Math.ceil(chunkWorkItems / WORKGROUP_SIZE);
    const dispatchX = use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = use2D
      ? Math.ceil(chunkWorkgroups / dispatchX)
      : 1;

    _st = profileSubOpBegin();
    dispatchComputePass(
      pipeline,
      bindGroup,
      dispatchX,
      dispatchY,
    );
    profileSubOpEnd("adam.dispatch", _st);

    // Config buffer must NOT be destroyed while shared encoder is active,
    // because the compute pass hasn't been submitted yet.
    // Use releaseParamsBuffer for safe deferred destruction.
    releaseParamsBuffer(configBuf);
  }

  // In-place: return the same input buffers (updated in-place by the shader)
  const result: AdamStepResult = {
    paramBuffer,
    mBuffer,
    vBuffer,
  };
  if (paramF16Out) {
    result.paramF16Buffer = paramF16Out;
  }
  return result;
}

/**
 * Reset all module-local mutable state (pipeline cache).
 */
export function resetAdamKernelState(): void {
  pipelineCache.clear();
}

