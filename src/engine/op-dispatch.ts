import { getBackend } from "../backend/registry";
import type {
  AdamStepConfig,
  Backend,
  BackendTensor,
  DeviceKind,
  DType,
  GeluOptions,
} from "../backend/types";
import { getCurrentOpLabel, setCurrentOpLabel } from "../backend/webgpu";
import {
  profileOpBegin,
  profileOpEnd,
  setProfileModule,
} from "../backend/webgpu/profiler";
import { sizeOf } from "../core/shape";
import type { LazyIRNode, LazyRef, StorageHandle } from "./lazy-types";
import { createStorageHandle } from "./node-factory";
import { storageTracker } from "./storage-tracker";

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

/** Extract a side output from a parent node and clear it.
 *  raw=false (default): value is a StorageHandle — unregister it and return the BackendTensor.
 *  raw=true: value is already a BackendTensor — return it directly. */
function extractSideOutput(
  node: LazyIRNode,
  key: string,
  raw = false,
): BackendTensor {
  const parent = node.inputs[0].node;
  const value = parent._sideOutputs?.[key];
  if (!value) throw new Error(`${node.op}: parent has no ${key} side output`);
  if (parent._sideOutputs) parent._sideOutputs[key] = undefined;
  if (raw) return value as BackendTensor;
  const sh = value as StorageHandle;
  storageTracker.unregister(sh.id);
  return sh.backendTensor;
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
  if (ref.node.result) {
    return ref.node.result;
  }
  throw new Error(
    `Input not ready: node id=${ref.node.id} op=${ref.node.op} shape=[${ref.node.shape}] caller=${new Error().stack?.split("\n")[2]?.trim()}`,
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

/** Create a StorageHandle for a side output, optionally mark it reachable, and store it on the node. */
function storeSideOutput(
  node: LazyIRNode,
  key: string,
  tensor: BackendTensor,
  reachableAnchor?: object,
): StorageHandle {
  const sh = createStorageHandle(node.device, tensor);
  if (reachableAnchor) storageTracker.markReachable(sh.id, reachableAnchor);
  if (!node._sideOutputs) node._sideOutputs = {};
  node._sideOutputs[key] = sh;
  return sh;
}

// ---------------------------------------------------------------------------
// Handler groups
// ---------------------------------------------------------------------------

function executeCreationOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor {
  switch (node.op) {
    case "tensorFromArray": {
      const payload = getPayload<{ values: number[] | Float32Array }>(node);
      if (!payload?.values) {
        throw new Error("tensorFromArray requires values in payload");
      }
      return backend.ops.tensorFromArray(payload.values, node.shape);
    }

    case "zeros": {
      if (backend.ops.zeros) {
        return backend.ops.zeros(node.shape);
      }
      const numEl = sizeOf(node.shape);
      return backend.ops.tensorFromArray(new Array(numEl).fill(0), node.shape);
    }

    case "full": {
      const payload = requirePayload<{ fillValue: number }>(node);
      if (backend.ops.full) {
        return backend.ops.full(node.shape, payload.fillValue);
      }
      const numEl = sizeOf(node.shape);
      return backend.ops.tensorFromArray(
        new Array(numEl).fill(payload.fillValue),
        node.shape,
      );
    }

    case "arange": {
      const ap = requirePayload<{ end: number; start: number; step: number }>(
        node,
      );
      if (backend.ops.arange) {
        return backend.ops.arange(ap.end, ap.start, ap.step);
      }
      const n = Math.max(0, Math.ceil((ap.end - ap.start) / ap.step));
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = ap.start + i * ap.step;
      return backend.ops.tensorFromArray(vals, node.shape);
    }

    case "tril": {
      assertOpSupported("tril", backend.ops.tril);
      return backend.ops.tril(
        backendInputs[0],
        getPayload<{ k: number }>(node)?.k ?? 0,
      );
    }

    case "triu": {
      assertOpSupported("triu", backend.ops.triu);
      return backend.ops.triu(
        backendInputs[0],
        getPayload<{ k: number }>(node)?.k ?? 0,
      );
    }

    case "rand": {
      const rp = requirePayload<{ seed: number }>(node);
      if (backend.ops.rand) return backend.ops.rand(node.shape, rp.seed);
      const n = sizeOf(node.shape);
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = Math.random();
      return backend.ops.tensorFromArray(vals, node.shape);
    }

    case "randn": {
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
    }

    case "bernoulli": {
      const bp = requirePayload<{ seed: number; p: number }>(node);
      if (backend.ops.bernoulli)
        return backend.ops.bernoulli(node.shape, bp.p, bp.seed);
      const n = sizeOf(node.shape);
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = Math.random() < bp.p ? 1 : 0;
      return backend.ops.tensorFromArray(vals, node.shape);
    }

    default:
      throw new Error(`Unknown creation op: ${node.op}`);
  }
}

function executeUnaryOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor {
  const input = backendInputs[0];
  switch (node.op) {
    case "sqrt":
    case "relu":
    case "exp":
    case "log":
    case "neg":
    case "abs":
    case "tanh":
    case "sigmoid":
    case "silu":
    case "sin":
    case "cos":
    case "rsqrt":
    case "floor":
    case "ceil":
    case "round":
    case "isfinite":
    case "sign": {
      const fn = backend.ops[node.op as "exp"];
      assertOpSupported(node.op, fn);
      return fn(input);
    }
    case "gelu": {
      assertOpSupported("gelu", backend.ops.gelu);
      const geluOpts = getPayload<GeluOptions>(node);
      return backend.ops.gelu(input, geluOpts);
    }
    case "clamp": {
      assertOpSupported("clamp", backend.ops.clamp);
      const clampPayload = getPayload<{
        min: number | null;
        max: number | null;
      }>(node);
      return backend.ops.clamp(
        input,
        clampPayload?.min ?? null,
        clampPayload?.max ?? null,
      );
    }
    case "pow":
      assertOpSupported("pow", backend.ops.pow);
      return backend.ops.pow(input, backendInputs[1]);
    default:
      throw new Error(`Unknown unary op: ${node.op}`);
  }
}

function executeBinaryOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor {
  switch (node.op) {
    case "add":
      return backend.ops.add(backendInputs[0], backendInputs[1]);
    case "sub": {
      const payload = getPayload<{ alpha?: number }>(node);
      return backend.ops.sub(backendInputs[0], backendInputs[1], payload);
    }
    case "mul":
      return backend.ops.mul(backendInputs[0], backendInputs[1]);
    case "div": {
      const payload = getPayload<{ roundingMode?: "trunc" | "floor" }>(node);
      return backend.ops.div(backendInputs[0], backendInputs[1], payload);
    }
    case "matmul":
      return backend.ops.matmul(backendInputs[0], backendInputs[1]);
    default:
      throw new Error(`Unknown binary op: ${node.op}`);
  }
}

function executeShapeOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor {
  switch (node.op) {
    case "reshape": {
      const payload = getPayload<{ targetShape: number[] }>(node);
      return backend.ops.reshape(
        backendInputs[0],
        payload?.targetShape ?? node.shape,
      );
    }
    case "expand":
      return backend.ops.expand(backendInputs[0], node.shape);
    case "transpose": {
      const payload = requirePayload<{ dim0: number; dim1: number }>(node);
      return backend.ops.transpose(backendInputs[0], payload);
    }
    case "permute": {
      const payload = requirePayload<{ dims: number[] }>(node);
      return backend.ops.permute(backendInputs[0], payload.dims);
    }
    case "contiguous":
      return backend.ops.contiguous(backendInputs[0]);
    case "narrow": {
      const p = requirePayload<{ dim: number; start: number; length: number }>(
        node,
      );
      assertOpSupported("narrow", backend.ops.narrow);
      return backend.ops.narrow(backendInputs[0], p.dim, p.start, p.length);
    }
    case "narrowBackward": {
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
    }
    case "cast": {
      const payload = requirePayload<{ dtype: DType }>(node);
      assertOpSupported("cast", backend.ops.cast);
      return backend.ops.cast(backendInputs[0], payload.dtype);
    }
    default:
      throw new Error(`Unknown shape op: ${node.op}`);
  }
}

// ---------------------------------------------------------------------------
// Reduction op dispatch table
// ---------------------------------------------------------------------------

type ReductionDesc = {
  /** Number of backendInputs to pass individually, or "all" to pass the array itself. */
  arity: number | "all";
  /** Whether payload is required, optional, or not passed. */
  payload: "required" | "optional" | "none";
};

const REDUCTION_TABLE: Record<string, ReductionDesc> = {
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
  gt: { arity: 2, payload: "none" },
  lt: { arity: 2, payload: "none" },
  ge: { arity: 2, payload: "none" },
  le: { arity: 2, payload: "none" },
  eq: { arity: 2, payload: "none" },
  ne: { arity: 2, payload: "none" },
  where: { arity: 3, payload: "none" },
};

function executeReductionOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor {
  const desc = REDUCTION_TABLE[node.op];
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

function executeMutationOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor | Promise<BackendTensor> {
  switch (node.op) {
    case "stridedScatterCopy":
    case "stridedScatterAdd": {
      const payload = requirePayload<{
        offset: number;
        viewShape: number[];
        viewStrides: number[];
      }>(node);
      const fn = backend.ops[node.op as "stridedScatterCopy"];
      assertOpSupported(node.op, fn);
      return fn(backendInputs[0], backendInputs[1], payload);
    }
    case "adamStep": {
      assertOpSupported("adamStep", backend.ops.adamStep);
      const payload = requirePayload<AdamStepConfig>(node);
      return (async () => {
        const adamResult = await backend.ops.adamStep?.(
          backendInputs[0],
          backendInputs[1],
          backendInputs[2],
          backendInputs[3],
          payload,
        );
        const mStorage = createStorageHandle(node.device, adamResult.m);
        const vStorage = createStorageHandle(node.device, adamResult.v);
        const adamMV = { m: mStorage, v: vStorage };
        storageTracker.markReachable(mStorage.id, adamMV);
        storageTracker.markReachable(vStorage.id, adamMV);
        if (!node._sideOutputs) node._sideOutputs = {};
        node._sideOutputs.adamMV = adamMV;
        return adamResult.param;
      })();
    }
    case "unscaleGrad": {
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
    }
    default:
      throw new Error(`Unknown mutation op: ${node.op}`);
  }
}

// ---------------------------------------------------------------------------
// Fused op dispatch table
// ---------------------------------------------------------------------------

type FusedOpDesc =
  | { kind: "dispatch" }
  | {
      kind: "sideOutputs";
      returnField: string;
      sideOutputs: [storeKey: string, resultField: string][];
    }
  | {
      kind: "rawSideOutputs";
      returnField: string;
      rawSideOutputs: [storeKey: string, resultField: string][];
    }
  | { kind: "extract"; key: string }
  | { kind: "extractRaw"; key: string };

/** Maps fused op names to their dispatch configuration. */
const FUSED_OP_TABLE: Record<string, FusedOpDesc> = {
  fusedCrossEntropyForward: { kind: "dispatch" },
  fusedCrossEntropyBackward: { kind: "dispatch" },
  fusedLayerNormForward: { kind: "dispatch" },
  fusedLayerNormBackwardGradX: { kind: "dispatch" },
  fusedLayerNormBackwardGradWeightBias: {
    kind: "rawSideOutputs",
    returnField: "gradWeight",
    rawSideOutputs: [["lnBwdGradBias", "gradBias"]],
  },
  fusedRMSNormForward: { kind: "dispatch" },
  fusedRMSNormBackwardGradX: { kind: "dispatch" },
  fusedRMSNormBackwardGradWeight: { kind: "dispatch" },
  fusedAttentionForward: {
    kind: "sideOutputs",
    returnField: "output",
    sideOutputs: [["attnLogsumexp", "logsumexp"]],
  },
  fusedAttentionBackward: {
    kind: "sideOutputs",
    returnField: "dQ",
    sideOutputs: [
      ["attnBwdDK", "dK"],
      ["attnBwdDV", "dV"],
    ],
  },
  extractAttentionLogsumexp: { kind: "extract", key: "attnLogsumexp" },
  extractAttentionDK: { kind: "extract", key: "attnBwdDK" },
  extractAttentionDV: { kind: "extract", key: "attnBwdDV" },
  extractLnBwdGradBias: { kind: "extractRaw", key: "lnBwdGradBias" },
};

type AnyOpFn = (...args: unknown[]) => unknown;

function executeFusedOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): BackendTensor | Promise<BackendTensor> {
  const desc = FUSED_OP_TABLE[node.op];
  if (desc) {
    if (desc.kind === "extract") return extractSideOutput(node, desc.key);
    if (desc.kind === "extractRaw")
      return extractSideOutput(node, desc.key, true);

    const payload = requirePayload(node);
    const fn = backend.ops[node.op as keyof Backend["ops"]] as
      | AnyOpFn
      | undefined;
    assertOpSupported(node.op, fn);
    const result = fn(...backendInputs, payload);

    if (desc.kind === "dispatch") return result as BackendTensor;

    const r = result as Record<string, BackendTensor>;
    if (desc.kind === "sideOutputs") {
      for (const [storeKey, field] of desc.sideOutputs) {
        storeSideOutput(node, storeKey, r[field]);
      }
    } else {
      if (!node._sideOutputs) node._sideOutputs = {};
      for (const [storeKey, field] of desc.rawSideOutputs) {
        node._sideOutputs[storeKey] = r[field];
      }
    }
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
  const add = (ops: string[], h: OpHandler) => {
    for (const op of ops) t.set(op, h);
  };
  add(
    [
      "tensorFromArray",
      "zeros",
      "full",
      "arange",
      "tril",
      "triu",
      "rand",
      "randn",
      "bernoulli",
    ],
    executeCreationOp,
  );
  add(
    [
      "sqrt",
      "relu",
      "exp",
      "log",
      "neg",
      "abs",
      "tanh",
      "sigmoid",
      "gelu",
      "silu",
      "pow",
      "isfinite",
      "sin",
      "cos",
      "rsqrt",
      "floor",
      "ceil",
      "round",
      "sign",
      "clamp",
    ],
    executeUnaryOp,
  );
  add(["add", "sub", "mul", "div", "matmul"], executeBinaryOp);
  add(
    [
      "reshape",
      "expand",
      "transpose",
      "permute",
      "contiguous",
      "narrow",
      "narrowBackward",
      "cast",
    ],
    executeShapeOp,
  );
  for (const op of Object.keys(REDUCTION_TABLE)) t.set(op, executeReductionOp);
  add(
    ["stridedScatterCopy", "stridedScatterAdd", "adamStep", "unscaleGrad"],
    executeMutationOp,
  );
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
