import { getBackend } from "../backend/registry";
import type {
  AdamStepConfig,
  Backend,
  BackendTensor,
  DeviceKind,
  DType,
} from "../backend/types";
import { sizeOf } from "../core/shape";
import { createStorageHandle } from "../graph/node-factory";
import {
  getCurrentOpLabel,
  profileOpBegin,
  profileOpEnd,
  setCurrentOpLabel,
  setProfileModule,
} from "../graph/profiler";
import type { LazyIRNode, LazyRef, StorageHandle } from "../graph/types";
import { OP_REGISTRY } from "../ops/registry";

/**
 * Execute a function within a profiling context.
 *
 * Sets up the op label and module for profiling, times the execution,
 * and cleans up after. Handles both sync and async functions.
 */
export async function withProfileContext<T>(
  label: string,
  module: string | undefined,
  fn: () => T | Promise<T>,
): Promise<T> {
  setCurrentOpLabel(label);
  setProfileModule(module ?? "unknown");
  const t0 = profileOpBegin(label);
  try {
    return await fn();
  } finally {
    profileOpEnd(label, t0);
    setCurrentOpLabel(null);
    setProfileModule("unknown");
  }
}

export function getInputStorage(
  ref: LazyRef,
  backend?: Backend,
): StorageHandle {
  if (ref.kind === "materialized") {
    return ref.storage;
  }
  if (ref.kind === "scalar") {
    // Materialize scalar ref on-the-fly for non-fused execution
    const b = backend ?? getBackend("cpu");
    if (!b) throw new Error("No backend available to materialize scalar ref");
    const prevLabel = getCurrentOpLabel();
    setCurrentOpLabel("full");
    const bt = b.ops.full
      ? b.ops.full([], ref.value)
      : b.ops.tensorFromArray([ref.value], []);
    setCurrentOpLabel(prevLabel);
    return createStorageHandle("cpu", bt);
  }
  // Multi-output: check outputIndex for secondary results
  const idx = ref.outputIndex ?? 0;
  if (idx === 0 && ref.node.result) {
    return ref.node.result;
  }
  if (ref.node.results?.[idx]) {
    return ref.node.results[idx];
  }
  throw new Error(
    `Input not ready: node id=${ref.node.id} op=${ref.node.op}[${idx}] shape=[${ref.node.shape}] caller=${new Error().stack?.split("\n")[2]?.trim()}`,
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Typed payload extraction — avoids repetitive inline `as` casts. */
function getPayload<T>(node: LazyIRNode): T | undefined {
  return node.payload as T | undefined;
}

/** Like getPayload but throws if payload is missing. */
function requirePayload<T>(node: LazyIRNode): T {
  const p = node.payload as T | undefined;
  if (!p) throw new Error(`${node.op} requires payload`);
  return p;
}

/** Assert that an optional backend op exists, throwing a descriptive error if not. */
function assertOpSupported(op: string, fn: unknown): asserts fn {
  if (!fn) throw new Error(`${op} not supported by backend`);
}

// ---------------------------------------------------------------------------
// Creation op dispatch table
// ---------------------------------------------------------------------------

type CreationHandler = (
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
) => BackendTensor;

const CREATION_OP_TABLE: Record<string, CreationHandler> = {
  tensorFromArray(node, _bi, backend) {
    const payload = getPayload<{
      values: number[] | Float32Array | Int32Array | Uint32Array;
      dtype?: DType;
    }>(node);
    if (!payload?.values)
      throw new Error("tensorFromArray requires values in payload");
    return backend.ops.tensorFromArray(
      payload.values,
      node.shape,
      payload.dtype ?? node.dtype,
    );
  },

  zeros(node, _bi, backend) {
    const payload = getPayload<{ dtype?: DType }>(node);
    const dtype = payload?.dtype ?? node.dtype;
    if (backend.ops.zeros) return backend.ops.zeros(node.shape, dtype);
    const numEl = sizeOf(node.shape);
    return backend.ops.tensorFromArray(
      new Array(numEl).fill(0),
      node.shape,
      dtype,
    );
  },

  full(node, _bi, backend) {
    const payload = requirePayload<{ fillValue: number; dtype?: DType }>(node);
    const dtype = payload.dtype ?? node.dtype;
    if (backend.ops.full)
      return backend.ops.full(node.shape, payload.fillValue, dtype);
    const numEl = sizeOf(node.shape);
    return backend.ops.tensorFromArray(
      new Array(numEl).fill(payload.fillValue),
      node.shape,
      dtype,
    );
  },

  arange(node, _bi, backend) {
    const ap = requirePayload<{ end: number; start: number; step: number }>(
      node,
    );
    if (backend.ops.arange)
      return backend.ops.arange(ap.end, ap.start, ap.step);
    const n = Math.max(0, Math.ceil((ap.end - ap.start) / ap.step));
    const vals = new Array(n);
    for (let i = 0; i < n; i++) vals[i] = ap.start + i * ap.step;
    return backend.ops.tensorFromArray(vals, node.shape);
  },

  tril(node, backendInputs, backend) {
    assertOpSupported("tril", backend.ops.tril);
    return backend.ops.tril(
      backendInputs[0],
      getPayload<{ k: number }>(node)?.k ?? 0,
    );
  },

  triu(node, backendInputs, backend) {
    assertOpSupported("triu", backend.ops.triu);
    return backend.ops.triu(
      backendInputs[0],
      getPayload<{ k: number }>(node)?.k ?? 0,
    );
  },

  rand(node, _bi, backend) {
    const rp = requirePayload<{ seed: number }>(node);
    if (backend.ops.rand) return backend.ops.rand(node.shape, rp.seed);
    const n = sizeOf(node.shape);
    const vals = new Array(n);
    for (let i = 0; i < n; i++) vals[i] = Math.random();
    return backend.ops.tensorFromArray(vals, node.shape);
  },

  randn(node, _bi, backend) {
    const rp = requirePayload<{ seed: number }>(node);
    if (backend.ops.randn) return backend.ops.randn(node.shape, rp.seed);
    const n = sizeOf(node.shape);
    const vals = new Array(n);
    for (let i = 0; i < n; i += 2) {
      const u1 = Math.random(),
        u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1 || 1e-10)),
        theta = 2 * Math.PI * u2;
      vals[i] = r * Math.cos(theta);
      if (i + 1 < n) vals[i + 1] = r * Math.sin(theta);
    }
    return backend.ops.tensorFromArray(vals, node.shape);
  },

  bernoulli(node, _bi, backend) {
    const bp = requirePayload<{ seed: number; p: number }>(node);
    if (backend.ops.bernoulli)
      return backend.ops.bernoulli(node.shape, bp.p, bp.seed);
    const n = sizeOf(node.shape);
    const vals = new Array(n);
    for (let i = 0; i < n; i++) vals[i] = Math.random() < bp.p ? 1 : 0;
    return backend.ops.tensorFromArray(vals, node.shape);
  },
};

// ---------------------------------------------------------------------------
// Shape op dispatch table
// ---------------------------------------------------------------------------

type ShapeHandler = (
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
) => BackendTensor;

const SHAPE_OP_TABLE: Record<string, ShapeHandler> = {
  reshape(node, backendInputs, backend) {
    const payload = getPayload<{ targetShape: number[] }>(node);
    return backend.ops.reshape(
      backendInputs[0],
      payload?.targetShape ?? node.shape,
    );
  },

  expand(node, backendInputs, backend) {
    return backend.ops.expand(backendInputs[0], node.shape);
  },

  transpose(node, backendInputs, backend) {
    const payload = requirePayload<{ dim0: number; dim1: number }>(node);
    return backend.ops.transpose(backendInputs[0], payload);
  },

  permute(node, backendInputs, backend) {
    const payload = requirePayload<{ dims: number[] }>(node);
    return backend.ops.permute(backendInputs[0], payload.dims);
  },

  contiguous(_node, backendInputs, backend) {
    return backend.ops.contiguous(backendInputs[0]);
  },

  narrow(node, backendInputs, backend) {
    const p = requirePayload<{ dim: number; start: number; length: number }>(
      node,
    );
    assertOpSupported("narrow", backend.ops.narrow);
    return backend.ops.narrow(backendInputs[0], p.dim, p.start, p.length);
  },

  narrowBackward(node, backendInputs, backend) {
    const p = requirePayload<{
      dim: number;
      start: number;
      originalLength: number;
    }>(node);
    assertOpSupported("narrowBackward", backend.ops.narrowBackward);
    return backend.ops.narrowBackward(
      backendInputs[0],
      p.dim,
      p.start,
      p.originalLength,
    );
  },

  cast(node, backendInputs, backend) {
    const payload = requirePayload<{ dtype: DType }>(node);
    assertOpSupported("cast", backend.ops.cast);
    return backend.ops.cast(backendInputs[0], payload.dtype);
  },
};

// ---------------------------------------------------------------------------
// Generic op dispatch table
// ---------------------------------------------------------------------------

type ReductionDesc = {
  /** Number of backendInputs to pass individually, or "all" to pass the array itself. */
  arity: number | "all";
  /** Whether payload is required, optional, or not passed. */
  payload: "required" | "optional" | "none";
};

// Payload overrides: lazy engine ops that accept options not captured in OP_REGISTRY
const PAYLOAD_OVERRIDES: Record<string, "optional"> = {
  gelu: "optional", // GeluOptions (approximation mode)
  sub: "optional", // SubOptions (alpha)
  div: "optional", // DivOptions (rounding_mode)
};

// Auto-derive elementwise op entries from OP_REGISTRY (single source of truth)
const GENERIC_OP_TABLE: Record<string, ReductionDesc> = {};
for (const [name, def] of Object.entries(OP_REGISTRY)) {
  // Skip cast variants (lazy engine uses "cast" with separate handler)
  if (name.startsWith("cast_")) continue;
  // Skip gelu aliases (lazy engine dispatches "gelu" with payload)
  if (name === "gelu_tanh" || name === "gelu_erf") continue;
  // Skip bitwise ops (not lazy engine ops)
  if (def.category === "bitwise") continue;
  // Note: min/max are no longer in OP_REGISTRY (renamed to minimum/maximum).
  // The reduction ops are listed below with arity 1.

  GENERIC_OP_TABLE[name] = {
    arity: def.arity,
    payload: PAYLOAD_OVERRIDES[name] ?? "none",
  };
}
// Non-registry ops (reductions, multi-input, mutations)
Object.assign(GENERIC_OP_TABLE, {
  matmul: { arity: 2, payload: "none" },
  sum: { arity: 1, payload: "optional" },
  max: { arity: 1, payload: "optional" },
  min: { arity: 1, payload: "optional" },
  mean: { arity: 1, payload: "optional" },
  argmax: { arity: 1, payload: "required" },
  argmin: { arity: 1, payload: "required" },
  conv2d: { arity: 3, payload: "optional" },
  gather: { arity: 2, payload: "required" },
  scatterAdd: { arity: 3, payload: "required" },
  cat: { arity: "all", payload: "required" },
  stridedScatterCopy: { arity: 2, payload: "required" },
  stridedScatterAdd: { arity: 2, payload: "required" },
});

function executeGenericOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor {
  const desc = GENERIC_OP_TABLE[node.op];
  if (!desc) throw new Error(`Unknown reduction op: ${node.op}`);

  const fn = backend.ops[node.op as keyof Backend["ops"]] as
    | AnyOpFn
    | undefined;
  assertOpSupported(node.op, fn);

  const args: unknown[] =
    desc.arity === "all" ? [backendInputs] : backendInputs.slice(0, desc.arity);
  if (desc.payload === "required") args.push(requirePayload(node));
  else if (desc.payload === "optional") args.push(getPayload(node));

  return fn(...args) as BackendTensor;
}

/** adamStep is special-cased: async, multi-output (param, m, v). */
function executeAdamStep(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): Promise<BackendTensor> {
  assertOpSupported("adamStep", backend.ops.adamStep);
  const payload = requirePayload<AdamStepConfig>(node);
  return (async () => {
    const adamResult = await backend.ops.adamStep!(
      backendInputs[0],
      backendInputs[1],
      backendInputs[2],
      backendInputs[3],
      payload,
    );
    // m/v are fresh output buffers from the kernel. Store in node.results so
    // that RuntimeTensors referencing outputIndex 1/2 can resolve them via
    // materializePendingTensors. The param result is returned and wrapped in
    // an owning StorageHandle by executeNode (→ node.result).
    const mStorage = createStorageHandle(node.device, adamResult.m);
    const vStorage = createStorageHandle(node.device, adamResult.v);
    node.results = [null as unknown as StorageHandle, mStorage, vStorage];
    return adamResult.param;
  })();
}

// ---------------------------------------------------------------------------
// Fused op dispatch table
// ---------------------------------------------------------------------------

type FusedOpDesc =
  | { kind: "dispatch" }
  | {
      kind: "multiOutput";
      returnField: string;
      extraFields: string[];
    };

/** Maps fused op names to their dispatch configuration. */
const FUSED_OP_TABLE: Record<string, FusedOpDesc> = {
  fusedCrossEntropyForward: { kind: "dispatch" },
  fusedCrossEntropyBackward: { kind: "dispatch" },
  fusedLayerNormForward: { kind: "dispatch" },
  fusedLayerNormBackwardGradX: { kind: "dispatch" },
  fusedLayerNormBackwardGradWeightBias: {
    kind: "multiOutput",
    returnField: "gradWeight",
    extraFields: ["gradBias"],
  },
  fusedRMSNormForward: { kind: "dispatch" },
  fusedRMSNormBackwardGradX: { kind: "dispatch" },
  fusedRMSNormBackwardGradWeight: { kind: "dispatch" },
  fusedRoPE: { kind: "dispatch" },
  fusedAttentionForward: {
    kind: "multiOutput",
    returnField: "output",
    extraFields: ["logsumexp"],
  },
  fusedAttentionBackward: {
    kind: "multiOutput",
    returnField: "dQ",
    extraFields: ["dK", "dV"],
  },
};

type AnyOpFn = (...args: unknown[]) => unknown;

function executeFusedOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor | Promise<BackendTensor> {
  const desc = FUSED_OP_TABLE[node.op];
  if (desc) {
    const payload = requirePayload(node);
    const fn = backend.ops[node.op as keyof Backend["ops"]] as
      | AnyOpFn
      | undefined;
    assertOpSupported(node.op, fn);
    const result = fn(...backendInputs, payload);

    if (desc.kind === "dispatch") return result as BackendTensor;

    // Multi-output: store all outputs in node.results
    const r = result as Record<string, BackendTensor>;
    const primarySH = createStorageHandle(node.device, r[desc.returnField]);
    const extraSHs = desc.extraFields.map((field) =>
      createStorageHandle(node.device, r[field]),
    );
    node.results = [primarySH, ...extraSHs];
    return r[desc.returnField];
  }

  // Special case: transfer (multi-backend, async)
  if (node.op === "transfer") {
    const transferPayload = getPayload<{ sourceDevice?: DeviceKind }>(node);
    const sourceDevice =
      transferPayload?.sourceDevice ??
      (node.inputs[0]?.kind === "materialized"
        ? node.inputs[0].storage.device
        : undefined);
    const sourceBackend = sourceDevice ? getBackend(sourceDevice) : backend;
    if (!sourceBackend)
      throw new Error(
        `Transfer failed: backend not available for source device ${sourceDevice}`,
      );
    const targetBackend = getBackend(node.device);
    if (!targetBackend)
      throw new Error(
        `Transfer failed: backend not available for ${node.device}`,
      );
    if (sourceDevice === node.device) return backendInputs[0]; // no-op
    return (async () => {
      const data = await sourceBackend.ops.read(backendInputs[0]);
      return targetBackend.ops.tensorFromArray(data, node.shape);
    })();
  }

  throw new Error(`Unknown fused op: ${node.op}`);
}

// ---------------------------------------------------------------------------
// Unified op routing table
// ---------------------------------------------------------------------------

type OpHandler = (
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
) => BackendTensor | Promise<BackendTensor>;

function buildOpTable(): Map<string, OpHandler> {
  const t = new Map<string, OpHandler>();

  // Creation ops (table-driven)
  for (const [op, handler] of Object.entries(CREATION_OP_TABLE))
    t.set(op, handler);

  // Shape ops (table-driven)
  for (const [op, handler] of Object.entries(SHAPE_OP_TABLE))
    t.set(op, handler);

  // Generic ops (auto-derived from OP_REGISTRY + manual entries)
  for (const op of Object.keys(GENERIC_OP_TABLE)) t.set(op, executeGenericOp);

  // clamp: outlier signature (min/max as separate nullable args)
  t.set("clamp", (node, backendInputs, backend) => {
    assertOpSupported("clamp", backend.ops.clamp);
    const p = getPayload<{ min: number | null; max: number | null }>(node);
    return backend.ops.clamp(backendInputs[0], p?.min ?? null, p?.max ?? null);
  });

  // unscaleGrad: outlier signature (payload fields spread as separate args)
  t.set("unscaleGrad", (node, backendInputs, backend) => {
    const payload = requirePayload<{
      invScale: number;
      infFlagBuffer: unknown;
    }>(node);
    assertOpSupported("unscaleGrad", backend.ops.unscaleGrad);
    return backend.ops.unscaleGrad(
      backendInputs[0],
      payload.invScale,
      payload.infFlagBuffer,
    );
  });

  // adamStep: special-cased (async, side outputs, markReachable)
  t.set("adamStep", executeAdamStep);

  return t;
}

const OP_TABLE = buildOpTable();

/**
 * Execute a single op on the backend.
 */
export async function executeOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): Promise<BackendTensor> {
  setCurrentOpLabel(node.op);
  const nodeModule = node.module ?? "unknown";
  setProfileModule(nodeModule);
  const _profT0 = profileOpBegin(node.op);
  try {
    const handler = OP_TABLE.get(node.op);
    if (handler) return handler(node, backendInputs, backend);
    // Fused ops and extract ops — no set needed, they're the remainder
    return executeFusedOp(node, backendInputs, backend);
  } finally {
    profileOpEnd(node.op, _profT0);
    setCurrentOpLabel(null);
    setProfileModule("unknown");
  }
}

/**
 * Synchronous version of executeOp for the lowered plan fast path.
 * Avoids async/Promise overhead (~5-15µs per microtask boundary) for ops
 * that are known to be synchronous (everything except adamStep and transfer).
 * Falls back to executeOp for async ops.
 */
export function executeOpSync(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor | Promise<BackendTensor> {
  setCurrentOpLabel(node.op);
  setProfileModule(node.module ?? "unknown");
  const _profT0 = profileOpBegin(node.op);
  try {
    const handler = OP_TABLE.get(node.op);
    if (handler) return handler(node, backendInputs, backend);
    return executeFusedOp(node, backendInputs, backend);
  } finally {
    profileOpEnd(node.op, _profT0);
    setCurrentOpLabel(null);
    setProfileModule("unknown");
  }
}
