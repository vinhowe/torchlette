import type {
  AdamStepConfig,
  Backend,
  BackendTensor,
  DeviceKind,
  DType,
  FusedAttentionConfig,
  FusedCrossEntropyConfig,
  FusedLayerNormConfig,
  GeluOptions,
} from "../backend/types";
import { getBackend } from "../backend/registry";
import { setCurrentOpLabel } from "../backend/webgpu";
import { sizeOf } from "../core/shape";
import { profileOpBegin, profileOpEnd, setProfileModule } from "../backend/webgpu/profiler";
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

/** Extract a StorageHandle side output from a parent node, unregister it, and return the BackendTensor. */
function extractSideOutput(node: LazyIRNode, key: string): BackendTensor {
  const parent = node.inputs[0].node;
  const sh = parent._sideOutputs?.[key] as StorageHandle | undefined;
  if (!sh) throw new Error(`${node.op}: parent has no ${key} side output`);
  const bt = sh.backendTensor;
  storageTracker.unregister(sh.id);
  parent._sideOutputs![key] = undefined;
  return bt;
}

/** Extract a raw BackendTensor side output from a parent node and clear it. */
function extractRawSideOutput(node: LazyIRNode, key: string): BackendTensor {
  const parent = node.inputs[0].node;
  const bt = parent._sideOutputs?.[key] as BackendTensor | undefined;
  if (!bt) throw new Error(`${node.op}: parent has no ${key} side output`);
  parent._sideOutputs![key] = undefined;
  return bt;
}

export function getInputStorage(ref: LazyRef, backend?: Backend): StorageHandle {
  if (ref.kind === "materialized") {
    return ref.storage;
  }
  if (ref.kind === "scalar") {
    // Materialize scalar ref on-the-fly for non-fused execution
    const b = backend ?? getBackend("cpu");
    if (!b) throw new Error("No backend available to materialize scalar ref");
    const bt = b.ops.full ? b.ops.full([], ref.value) : b.ops.tensorFromArray([ref.value], []);
    return createStorageHandle("cpu", bt);
  }
  if (ref.node.result) {
    return ref.node.result;
  }
  throw new Error(`Input not ready: node id=${ref.node.id} op=${ref.node.op} shape=[${ref.node.shape}] caller=${new Error().stack?.split("\n")[2]?.trim()}`);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Typed payload extraction — avoids repetitive inline `as` casts. */
function getPayload<T>(node: LazyIRNode): T | undefined {
  return node.payload as T | undefined;
}

/** Assert that an optional backend op exists, throwing a descriptive error if not. */
function assertOpSupported(op: string, fn: unknown): asserts fn {
  if (!fn) throw new Error(`${op} not supported by backend`);
}

/** Create a StorageHandle for a side output, optionally mark it reachable, and store it on the node. */
function storeSideOutput(
  node: LazyIRNode, key: string, tensor: BackendTensor, reachableAnchor?: object,
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
  node: LazyIRNode, backendInputs: BackendTensor[], backend: Backend,
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
      const payload = getPayload<{ fillValue: number }>(node)!;
      if (backend.ops.full) {
        return backend.ops.full(node.shape, payload.fillValue);
      }
      const numEl = sizeOf(node.shape);
      return backend.ops.tensorFromArray(new Array(numEl).fill(payload.fillValue), node.shape);
    }

    case "arange": {
      const ap = getPayload<{ end: number; start: number; step: number }>(node)!;
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
      return backend.ops.tril(backendInputs[0], getPayload<{ k: number }>(node)?.k ?? 0);
    }

    case "triu": {
      assertOpSupported("triu", backend.ops.triu);
      return backend.ops.triu(backendInputs[0], getPayload<{ k: number }>(node)?.k ?? 0);
    }

    case "rand": {
      const rp = getPayload<{ seed: number }>(node)!;
      if (backend.ops.rand) return backend.ops.rand(node.shape, rp.seed);
      const n = sizeOf(node.shape);
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = Math.random();
      return backend.ops.tensorFromArray(vals, node.shape);
    }

    case "randn": {
      const rp = getPayload<{ seed: number }>(node)!;
      if (backend.ops.randn) return backend.ops.randn(node.shape, rp.seed);
      const n = sizeOf(node.shape);
      const vals = new Array(n);
      for (let i = 0; i < n; i += 2) {
        const u1 = Math.random(), u2 = Math.random();
        const r = Math.sqrt(-2 * Math.log(u1 || 1e-10)), theta = 2 * Math.PI * u2;
        vals[i] = r * Math.cos(theta);
        if (i + 1 < n) vals[i + 1] = r * Math.sin(theta);
      }
      return backend.ops.tensorFromArray(vals, node.shape);
    }

    case "bernoulli": {
      const bp = getPayload<{ seed: number; p: number }>(node)!;
      if (backend.ops.bernoulli) return backend.ops.bernoulli(node.shape, bp.p, bp.seed);
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
  node: LazyIRNode, backendInputs: BackendTensor[], backend: Backend,
  donationOpts?: { outBuffer: unknown },
): BackendTensor {
  const input = backendInputs[0];
  switch (node.op) {
    case "sqrt":
      return backend.ops.sqrt(input, donationOpts);
    case "relu":
      return backend.ops.relu(input, donationOpts);
    case "exp":
      assertOpSupported("exp", backend.ops.exp);
      return backend.ops.exp(input, donationOpts);
    case "log":
      assertOpSupported("log", backend.ops.log);
      return backend.ops.log(input, donationOpts);
    case "neg":
      assertOpSupported("neg", backend.ops.neg);
      return backend.ops.neg(input, donationOpts);
    case "abs":
      assertOpSupported("abs", backend.ops.abs);
      return backend.ops.abs(input, donationOpts);
    case "tanh":
      assertOpSupported("tanh", backend.ops.tanh);
      return backend.ops.tanh(input, donationOpts);
    case "sigmoid":
      assertOpSupported("sigmoid", backend.ops.sigmoid);
      return backend.ops.sigmoid(input, donationOpts);
    case "gelu": {
      assertOpSupported("gelu", backend.ops.gelu);
      const geluOpts = getPayload<GeluOptions>(node);
      return backend.ops.gelu(input, { ...geluOpts, ...donationOpts });
    }
    case "silu":
      assertOpSupported("silu", backend.ops.silu);
      return backend.ops.silu(input, donationOpts);
    case "isfinite":
      assertOpSupported("isfinite", backend.ops.isfinite);
      return backend.ops.isfinite(input);
    case "pow":
      assertOpSupported("pow", backend.ops.pow);
      return backend.ops.pow(input, backendInputs[1]);
    default:
      throw new Error(`Unknown unary op: ${node.op}`);
  }
}

function executeBinaryOp(
  node: LazyIRNode, backendInputs: BackendTensor[], backend: Backend,
  donationOpts?: { outBuffer: unknown },
): BackendTensor {
  switch (node.op) {
    case "add":
      return backend.ops.add(backendInputs[0], backendInputs[1], donationOpts);
    case "sub": {
      const payload = getPayload<{ alpha?: number }>(node);
      return backend.ops.sub(backendInputs[0], backendInputs[1], { ...payload, ...donationOpts });
    }
    case "mul":
      return backend.ops.mul(backendInputs[0], backendInputs[1], donationOpts);
    case "div": {
      const payload = getPayload<{ roundingMode?: "trunc" | "floor" }>(node);
      return backend.ops.div(backendInputs[0], backendInputs[1], { ...payload, ...donationOpts });
    }
    case "matmul":
      return backend.ops.matmul(backendInputs[0], backendInputs[1], donationOpts);
    default:
      throw new Error(`Unknown binary op: ${node.op}`);
  }
}

function executeShapeOp(
  node: LazyIRNode, backendInputs: BackendTensor[], backend: Backend,
): BackendTensor {
  switch (node.op) {
    case "reshape": {
      const payload = getPayload<{ targetShape: number[] }>(node);
      return backend.ops.reshape(backendInputs[0], payload?.targetShape ?? node.shape);
    }
    case "expand":
      return backend.ops.expand(backendInputs[0], node.shape);
    case "transpose": {
      const payload = getPayload<{ dim0: number; dim1: number }>(node);
      if (!payload) throw new Error("transpose requires dim0 and dim1 in payload");
      return backend.ops.transpose(backendInputs[0], payload);
    }
    case "permute": {
      const payload = getPayload<{ dims: number[] }>(node);
      if (!payload) throw new Error("permute requires dims in payload");
      return backend.ops.permute(backendInputs[0], payload.dims);
    }
    case "contiguous":
      return backend.ops.contiguous(backendInputs[0]);
    case "narrow": {
      const p = getPayload<{ dim: number; start: number; length: number }>(node)!;
      assertOpSupported("narrow", backend.ops.narrow);
      return backend.ops.narrow(backendInputs[0], p.dim, p.start, p.length);
    }
    case "narrowBackward": {
      const p = getPayload<{ dim: number; start: number; originalLength: number }>(node)!;
      assertOpSupported("narrowBackward", backend.ops.narrowBackward);
      return backend.ops.narrowBackward(backendInputs[0], p.dim, p.start, p.originalLength);
    }
    case "cast": {
      const payload = getPayload<{ dtype: DType }>(node);
      if (!payload) throw new Error("cast requires dtype in payload");
      assertOpSupported("cast", backend.ops.cast);
      return backend.ops.cast(backendInputs[0], payload.dtype);
    }
    default:
      throw new Error(`Unknown shape op: ${node.op}`);
  }
}

function executeReductionOp(
  node: LazyIRNode, backendInputs: BackendTensor[], backend: Backend,
  donationOpts?: { outBuffer: unknown },
): BackendTensor {
  switch (node.op) {
    case "sum":
      return backend.ops.sum(backendInputs[0], getPayload<{ dim?: number | number[] | null; keepdim?: boolean }>(node));
    case "max":
      return backend.ops.max(backendInputs[0], getPayload<{ dim?: number | number[] | null; keepdim?: boolean }>(node));
    case "mean":
      return backend.ops.mean(backendInputs[0], getPayload<{ dim?: number | number[] | null; keepdim?: boolean }>(node));
    case "argmax": {
      assertOpSupported("argmax", backend.ops.argmax);
      return backend.ops.argmax(backendInputs[0], getPayload<{ dim: number; keepdim?: boolean }>(node)!);
    }
    case "argmin": {
      assertOpSupported("argmin", backend.ops.argmin);
      return backend.ops.argmin(backendInputs[0], getPayload<{ dim: number; keepdim?: boolean }>(node)!);
    }
    case "gather": {
      const payload = getPayload<{ dim: number }>(node);
      if (!payload) throw new Error("gather requires dim in payload");
      return backend.ops.gather(backendInputs[0], backendInputs[1], payload);
    }
    case "scatterAdd": {
      const payload = getPayload<{ dim: number }>(node);
      if (!payload) throw new Error("scatterAdd requires dim in payload");
      return backend.ops.scatterAdd(backendInputs[0], backendInputs[1], backendInputs[2], payload);
    }
    case "gt":
      assertOpSupported("gt", backend.ops.gt);
      return backend.ops.gt(backendInputs[0], backendInputs[1], donationOpts);
    case "lt":
      assertOpSupported("lt", backend.ops.lt);
      return backend.ops.lt(backendInputs[0], backendInputs[1], donationOpts);
    case "ge":
      assertOpSupported("ge", backend.ops.ge);
      return backend.ops.ge(backendInputs[0], backendInputs[1], donationOpts);
    case "le":
      assertOpSupported("le", backend.ops.le);
      return backend.ops.le(backendInputs[0], backendInputs[1], donationOpts);
    case "eq":
      assertOpSupported("eq", backend.ops.eq);
      return backend.ops.eq(backendInputs[0], backendInputs[1], donationOpts);
    case "ne":
      assertOpSupported("ne", backend.ops.ne);
      return backend.ops.ne(backendInputs[0], backendInputs[1], donationOpts);
    case "where":
      return backend.ops.where(backendInputs[0], backendInputs[1], backendInputs[2], donationOpts);
    default:
      throw new Error(`Unknown reduction op: ${node.op}`);
  }
}

function executeMutationOp(
  node: LazyIRNode, backendInputs: BackendTensor[], backend: Backend,
): BackendTensor | Promise<BackendTensor> {
  switch (node.op) {
    case "stridedScatterCopy": {
      const payload = getPayload<{ offset: number; viewShape: number[]; viewStrides: number[] }>(node);
      if (!payload) throw new Error("stridedScatterCopy requires options in payload");
      return backend.ops.stridedScatterCopy(backendInputs[0], backendInputs[1], payload);
    }
    case "stridedScatterAdd": {
      const payload = getPayload<{ offset: number; viewShape: number[]; viewStrides: number[] }>(node);
      if (!payload) throw new Error("stridedScatterAdd requires options in payload");
      return backend.ops.stridedScatterAdd(backendInputs[0], backendInputs[1], payload);
    }
    case "adamStep": {
      assertOpSupported("adamStep", backend.ops.adamStep);
      const payload = getPayload<AdamStepConfig>(node)!;
      return (async () => {
        const adamResult = await backend.ops.adamStep!(
          backendInputs[0], backendInputs[1], backendInputs[2], backendInputs[3], payload,
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
      const payload = getPayload<{ invScale: number; infFlagBuffer: unknown }>(node)!;
      assertOpSupported("unscaleGrad", backend.ops.unscaleGrad);
      return backend.ops.unscaleGrad(backendInputs[0], payload.invScale, payload.infFlagBuffer);
    }
    default:
      throw new Error(`Unknown mutation op: ${node.op}`);
  }
}

function executeFusedOp(
  node: LazyIRNode, backendInputs: BackendTensor[], backend: Backend,
): BackendTensor | Promise<BackendTensor> {
  switch (node.op) {
    case "fusedAttentionForward": {
      const payload = getPayload<FusedAttentionConfig>(node)!;
      assertOpSupported("fusedAttentionForward", backend.ops.fusedAttentionForward);
      const result = backend.ops.fusedAttentionForward(
        backendInputs[0], backendInputs[1], backendInputs[2], payload,
      );
      storeSideOutput(node, "attnLogsumexp", result.logsumexp);
      return result.output;
    }
    case "extractAttentionLogsumexp":
      return extractSideOutput(node, "attnLogsumexp");
    case "fusedAttentionBackward": {
      const payload = getPayload<FusedAttentionConfig>(node)!;
      assertOpSupported("fusedAttentionBackward", backend.ops.fusedAttentionBackward);
      const result = backend.ops.fusedAttentionBackward(
        backendInputs[0], backendInputs[1], backendInputs[2],
        backendInputs[3], backendInputs[4], backendInputs[5], payload,
      );
      storeSideOutput(node, "attnBwdDK", result.dK);
      storeSideOutput(node, "attnBwdDV", result.dV);
      return result.dQ;
    }
    case "extractAttentionDK":
      return extractSideOutput(node, "attnBwdDK");
    case "extractAttentionDV":
      return extractSideOutput(node, "attnBwdDV");
    case "fusedCrossEntropyForward": {
      const payload = getPayload<FusedCrossEntropyConfig>(node)!;
      assertOpSupported("fusedCrossEntropyForward", backend.ops.fusedCrossEntropyForward);
      return backend.ops.fusedCrossEntropyForward(backendInputs[0], backendInputs[1], payload);
    }
    case "fusedCrossEntropyBackward": {
      const payload = getPayload<FusedCrossEntropyConfig>(node)!;
      assertOpSupported("fusedCrossEntropyBackward", backend.ops.fusedCrossEntropyBackward);
      return backend.ops.fusedCrossEntropyBackward(backendInputs[0], backendInputs[1], backendInputs[2], payload);
    }
    case "fusedLayerNormForward": {
      const payload = getPayload<FusedLayerNormConfig>(node)!;
      assertOpSupported("fusedLayerNormForward", backend.ops.fusedLayerNormForward);
      return backend.ops.fusedLayerNormForward(backendInputs[0], backendInputs[1], backendInputs[2], payload);
    }
    case "fusedLayerNormBackwardGradX": {
      const payload = getPayload<FusedLayerNormConfig>(node)!;
      assertOpSupported("fusedLayerNormBackwardGradX", backend.ops.fusedLayerNormBackwardGradX);
      return backend.ops.fusedLayerNormBackwardGradX(backendInputs[0], backendInputs[1], backendInputs[2], payload);
    }
    case "fusedLayerNormBackwardGradWeightBias": {
      const payload = getPayload<FusedLayerNormConfig>(node)!;
      assertOpSupported("fusedLayerNormBackwardGradWeightBias", backend.ops.fusedLayerNormBackwardGradWeightBias);
      const result = backend.ops.fusedLayerNormBackwardGradWeightBias(
        backendInputs[0], backendInputs[1], payload,
      );
      // Store raw gradBias BackendTensor for extractLnBwdGradBias
      if (!node._sideOutputs) node._sideOutputs = {};
      node._sideOutputs.lnBwdGradBias = result.gradBias;
      return result.gradWeight;
    }
    case "extractLnBwdGradBias":
      return extractRawSideOutput(node, "lnBwdGradBias");
    case "transfer": {
      const transferPayload = getPayload<{ sourceDevice?: DeviceKind }>(node);
      const sourceDevice = transferPayload?.sourceDevice
        ?? (node.inputs[0]?.kind === "materialized" ? node.inputs[0].storage.device : undefined);
      const sourceBackend = sourceDevice ? getBackend(sourceDevice) : backend;
      if (!sourceBackend) throw new Error(`Transfer failed: backend not available for source device ${sourceDevice}`);
      const targetBackend = getBackend(node.device);
      if (!targetBackend) throw new Error(`Transfer failed: backend not available for ${node.device}`);
      if (sourceDevice === node.device) return backendInputs[0]; // no-op
      return (async () => {
        const data = await sourceBackend.ops.read(backendInputs[0]);
        return targetBackend.ops.tensorFromArray(data, node.shape);
      })();
    }
    default:
      throw new Error(`Unknown fused op: ${node.op}`);
  }
}

// ---------------------------------------------------------------------------
// Op category sets for fast routing
// ---------------------------------------------------------------------------

const CREATION_OPS = new Set([
  "tensorFromArray", "zeros", "full", "arange", "tril", "triu", "rand", "randn", "bernoulli",
]);
const UNARY_OPS = new Set([
  "sqrt", "relu", "exp", "log", "neg", "abs", "tanh", "sigmoid", "gelu", "silu", "isfinite", "pow",
]);
const BINARY_OPS = new Set(["add", "sub", "mul", "div", "matmul"]);
const SHAPE_OPS = new Set([
  "reshape", "expand", "transpose", "permute", "contiguous", "narrow", "narrowBackward", "cast",
]);
const REDUCTION_OPS = new Set([
  "sum", "max", "mean", "argmax", "argmin", "gather", "scatterAdd",
  "gt", "lt", "ge", "le", "eq", "ne", "where",
]);
const MUTATION_OPS = new Set([
  "stridedScatterCopy", "stridedScatterAdd", "adamStep", "unscaleGrad",
]);

/**
 * Execute a single op on the backend.
 */
export async function executeOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
  donationOpts?: { outBuffer: unknown },
): Promise<BackendTensor> {
  setCurrentOpLabel(node.op);
  setProfileModule(node.module ?? "unknown");
  const _profT0 = profileOpBegin(node.op);
  try {
    const op = node.op;
    if (CREATION_OPS.has(op)) return executeCreationOp(node, backendInputs, backend);
    if (UNARY_OPS.has(op)) return executeUnaryOp(node, backendInputs, backend, donationOpts);
    if (BINARY_OPS.has(op)) return executeBinaryOp(node, backendInputs, backend, donationOpts);
    if (SHAPE_OPS.has(op)) return executeShapeOp(node, backendInputs, backend);
    if (REDUCTION_OPS.has(op)) return executeReductionOp(node, backendInputs, backend, donationOpts);
    if (MUTATION_OPS.has(op)) return executeMutationOp(node, backendInputs, backend);
    // Fused ops and extract ops — no set needed, they're the remainder
    return executeFusedOp(node, backendInputs, backend);
  } finally {
    profileOpEnd(node.op, _profT0);
    setCurrentOpLabel(null);
    setProfileModule("unknown");
  }
}
