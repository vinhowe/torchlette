import type { Backend, BackendTensor, DeviceKind, DType, GeluOptions } from "../backend/types";
import { getBackend } from "../backend/registry";
import {
  beginBatchExecution,
  endBatchExecution,
  isBatchActive,
  abortBatch,
  flushBufferPool,
  beginSharedEncoder,
  endSharedEncoder,
  setCurrentOpLabel,
} from "../backend/webgpu";
import { profileOpBegin, profileOpEnd, setProfileModule } from "../backend/webgpu/profiler";
import {
  analyzeLifetimes,
  findDeadTensorsAtStep,
  type TensorLifetime,
} from "./memory-planning";
import type { LazyIRNode, LazyRef, StorageHandle, ExecutionPlan, ExecutePlanOptions } from "./lazy-types";
import { createStorageHandle } from "./node-factory";
import { storageTracker, canSafelyRelease, releaseBufferImmediate } from "./storage-tracker";
import { extractPlanMetadata, pretunePlanMatmuls, segmentPlanAtCheckpoints } from "./plan-builder";

// ============================================================================
// Constants
// ============================================================================

/** Ops safe to execute during tape replay fill-in: pure views + data sources. */
export const FILL_IN_OPS: ReadonlySet<string> = new Set([
  // Pure view ops (no GPU dispatch, same buffer)
  "reshape", "transpose", "permute", "expand", "narrow", "contiguous",
  // Data source ops (create new buffers from host data)
  "tensorFromArray", "zeros", "full", "arange",
]);

// ============================================================================
// Internal Helpers
// ============================================================================

function getInputStorage(ref: LazyRef, backend?: Backend): StorageHandle {
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

export function createStorageHandleInternal(
  device: DeviceKind,
  backendTensor: BackendTensor,
  baseStorageId?: number,
): StorageHandle {
  return createStorageHandle(device, backendTensor, baseStorageId);
}

// ============================================================================
// Single-Op Execution
// ============================================================================

/**
 * Internal helper to execute a single op.
 * This is a subset of the switch statement in executePlan, factored out for reuse.
 */
async function executeOpInternal(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  nodeBackend: Backend,
): Promise<BackendTensor> {
  setCurrentOpLabel(node.op);
  setProfileModule(node.module ?? "unknown");
  const _profT0 = profileOpBegin(node.op);
  try {
  switch (node.op) {
    case "tensorFromArray": {
      const payload = node.payload as { values: number[] | Float32Array } | undefined;
      if (!payload?.values) {
        throw new Error("tensorFromArray requires values in payload");
      }
      return nodeBackend.ops.tensorFromArray(payload.values, node.shape);
    }

    case "zeros": {
      if (nodeBackend.ops.zeros) {
        return nodeBackend.ops.zeros(node.shape);
      }
      const numEl = node.shape.reduce((a: number, b: number) => a * b, 1);
      return nodeBackend.ops.tensorFromArray(new Array(numEl).fill(0), node.shape);
    }

    case "full": {
      const fullPayload = node.payload as { fillValue: number };
      if (nodeBackend.ops.full) {
        return nodeBackend.ops.full(node.shape, fullPayload.fillValue);
      }
      const numElFull = node.shape.reduce((a: number, b: number) => a * b, 1);
      return nodeBackend.ops.tensorFromArray(new Array(numElFull).fill(fullPayload.fillValue), node.shape);
    }

    case "arange": {
      const ap = node.payload as { end: number; start: number; step: number };
      if (nodeBackend.ops.arange) {
        return nodeBackend.ops.arange(ap.end, ap.start, ap.step);
      }
      const n = Math.max(0, Math.ceil((ap.end - ap.start) / ap.step));
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = ap.start + i * ap.step;
      return nodeBackend.ops.tensorFromArray(vals, node.shape);
    }

    case "tril": {
      if (!nodeBackend.ops.tril) throw new Error("tril not supported by backend");
      return nodeBackend.ops.tril(backendInputs[0], (node.payload as { k: number })?.k ?? 0);
    }

    case "triu": {
      if (!nodeBackend.ops.triu) throw new Error("triu not supported by backend");
      return nodeBackend.ops.triu(backendInputs[0], (node.payload as { k: number })?.k ?? 0);
    }

    case "rand": {
      const rp = node.payload as { seed: number };
      if (nodeBackend.ops.rand) return nodeBackend.ops.rand(node.shape, rp.seed);
      const n = node.shape.reduce((a: number, b: number) => a * b, 1);
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = Math.random();
      return nodeBackend.ops.tensorFromArray(vals, node.shape);
    }

    case "randn": {
      const rp = node.payload as { seed: number };
      if (nodeBackend.ops.randn) return nodeBackend.ops.randn(node.shape, rp.seed);
      const n = node.shape.reduce((a: number, b: number) => a * b, 1);
      const vals = new Array(n);
      for (let i = 0; i < n; i += 2) {
        const u1 = Math.random(), u2 = Math.random();
        const r = Math.sqrt(-2 * Math.log(u1 || 1e-10)), theta = 2 * Math.PI * u2;
        vals[i] = r * Math.cos(theta);
        if (i + 1 < n) vals[i + 1] = r * Math.sin(theta);
      }
      return nodeBackend.ops.tensorFromArray(vals, node.shape);
    }

    case "bernoulli": {
      const bp = node.payload as { seed: number; p: number };
      if (nodeBackend.ops.bernoulli) return nodeBackend.ops.bernoulli(node.shape, bp.p, bp.seed);
      const n = node.shape.reduce((a: number, b: number) => a * b, 1);
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = Math.random() < bp.p ? 1 : 0;
      return nodeBackend.ops.tensorFromArray(vals, node.shape);
    }

    case "add":
      return nodeBackend.ops.add(backendInputs[0], backendInputs[1]);

    case "sub": {
      const subPayload = node.payload as { alpha?: number } | undefined;
      return nodeBackend.ops.sub(backendInputs[0], backendInputs[1], subPayload);
    }

    case "mul":
      return nodeBackend.ops.mul(backendInputs[0], backendInputs[1]);

    case "div": {
      const divPayload = node.payload as { roundingMode?: "trunc" | "floor" } | undefined;
      return nodeBackend.ops.div(backendInputs[0], backendInputs[1], divPayload);
    }

    case "matmul":
      return nodeBackend.ops.matmul(backendInputs[0], backendInputs[1]);

    case "sqrt":
      return nodeBackend.ops.sqrt(backendInputs[0]);

    case "relu":
      return nodeBackend.ops.relu(backendInputs[0]);

    case "exp":
      if (!nodeBackend.ops.exp) throw new Error("exp not supported on this backend");
      return nodeBackend.ops.exp(backendInputs[0]);

    case "log":
      if (!nodeBackend.ops.log) throw new Error("log not supported on this backend");
      return nodeBackend.ops.log(backendInputs[0]);

    case "neg":
      if (!nodeBackend.ops.neg) throw new Error("neg not supported on this backend");
      return nodeBackend.ops.neg(backendInputs[0]);

    case "abs":
      if (!nodeBackend.ops.abs) throw new Error("abs not supported on this backend");
      return nodeBackend.ops.abs(backendInputs[0]);

    case "tanh":
      if (!nodeBackend.ops.tanh) throw new Error("tanh not supported on this backend");
      return nodeBackend.ops.tanh(backendInputs[0]);

    case "sigmoid":
      if (!nodeBackend.ops.sigmoid) throw new Error("sigmoid not supported on this backend");
      return nodeBackend.ops.sigmoid(backendInputs[0]);

    case "gelu": {
      if (!nodeBackend.ops.gelu) throw new Error("gelu not supported on this backend");
      const geluPayload = node.payload as GeluOptions | undefined;
      return nodeBackend.ops.gelu(backendInputs[0], geluPayload);
    }

    case "silu":
      if (!nodeBackend.ops.silu) throw new Error("silu not supported on this backend");
      return nodeBackend.ops.silu(backendInputs[0]);

    case "cast": {
      if (!nodeBackend.ops.cast) throw new Error("cast not supported on this backend");
      const castPayload = node.payload as { dtype: DType } | undefined;
      if (!castPayload?.dtype) throw new Error("cast requires dtype in payload");
      return nodeBackend.ops.cast(backendInputs[0], castPayload.dtype);
    }

    case "pow":
      if (!nodeBackend.ops.pow) throw new Error("pow not supported on this backend");
      return nodeBackend.ops.pow(backendInputs[0], backendInputs[1]);

    case "reshape":
      return nodeBackend.ops.reshape(backendInputs[0], node.shape);

    case "expand":
      return nodeBackend.ops.expand(backendInputs[0], node.shape);

    case "transpose": {
      const transposePayload = node.payload as { dim0: number; dim1: number } | undefined;
      if (!transposePayload) throw new Error("transpose requires dim0, dim1 in payload");
      return nodeBackend.ops.transpose(backendInputs[0], transposePayload);
    }

    case "permute": {
      const permutePayload = node.payload as { dims: number[] } | undefined;
      if (!permutePayload?.dims) throw new Error("permute requires dims in payload");
      return nodeBackend.ops.permute(backendInputs[0], permutePayload.dims);
    }

    case "contiguous":
      return nodeBackend.ops.contiguous(backendInputs[0]);

    case "narrow": {
      const p = node.payload as { dim: number; start: number; length: number };
      if (!nodeBackend.ops.narrow) throw new Error("narrow not supported by backend");
      return nodeBackend.ops.narrow(backendInputs[0], p.dim, p.start, p.length);
    }

    case "narrowBackward": {
      const p = node.payload as { dim: number; start: number; originalLength: number };
      if (!nodeBackend.ops.narrowBackward) throw new Error("narrowBackward not supported by backend");
      return nodeBackend.ops.narrowBackward(backendInputs[0], p.dim, p.start, p.originalLength);
    }

    case "gather": {
      const gatherPayload = node.payload as { dim: number } | undefined;
      if (gatherPayload?.dim === undefined) throw new Error("gather requires dim in payload");
      return nodeBackend.ops.gather(backendInputs[0], backendInputs[1], gatherPayload);
    }

    case "scatterAdd": {
      const scatterAddPayload = node.payload as { dim: number } | undefined;
      if (scatterAddPayload?.dim === undefined) throw new Error("scatterAdd requires dim in payload");
      return nodeBackend.ops.scatterAdd(backendInputs[0], backendInputs[1], backendInputs[2], scatterAddPayload);
    }

    case "sum": {
      const sumPayload = node.payload as { dim?: number | number[] | null; keepdim?: boolean } | undefined;
      return nodeBackend.ops.sum(backendInputs[0], sumPayload);
    }

    case "mean": {
      const meanPayload = node.payload as { dim?: number | number[] | null; keepdim?: boolean } | undefined;
      return nodeBackend.ops.mean(backendInputs[0], meanPayload);
    }

    case "max": {
      const maxPayload = node.payload as { dim?: number | number[] | null; keepdim?: boolean } | undefined;
      return nodeBackend.ops.max(backendInputs[0], maxPayload);
    }

    case "argmax": {
      const argmaxPayload = node.payload as { dim: number; keepdim?: boolean } | undefined;
      if (argmaxPayload?.dim === undefined) throw new Error("argmax requires dim in payload");
      return nodeBackend.ops.argmax(backendInputs[0], argmaxPayload);
    }

    case "argmin": {
      const argminPayload = node.payload as { dim: number; keepdim?: boolean } | undefined;
      if (argminPayload?.dim === undefined) throw new Error("argmin requires dim in payload");
      return nodeBackend.ops.argmin(backendInputs[0], argminPayload);
    }

    case "gt":
      return nodeBackend.ops.gt(backendInputs[0], backendInputs[1]);

    case "lt":
      return nodeBackend.ops.lt(backendInputs[0], backendInputs[1]);

    case "ge":
      return nodeBackend.ops.ge(backendInputs[0], backendInputs[1]);

    case "le":
      return nodeBackend.ops.le(backendInputs[0], backendInputs[1]);

    case "eq":
      return nodeBackend.ops.eq(backendInputs[0], backendInputs[1]);

    case "ne":
      return nodeBackend.ops.ne(backendInputs[0], backendInputs[1]);

    case "where":
      return nodeBackend.ops.where(backendInputs[0], backendInputs[1], backendInputs[2]);

    case "stridedScatterCopy": {
      const scatterCopyPayload = node.payload as { offset: number; viewShape: number[]; viewStrides: number[] } | undefined;
      if (!scatterCopyPayload) throw new Error("stridedScatterCopy requires offset, viewShape, viewStrides in payload");
      return nodeBackend.ops.stridedScatterCopy(backendInputs[0], backendInputs[1], scatterCopyPayload);
    }

    case "stridedScatterAdd": {
      const scatterAddPayload = node.payload as { offset: number; viewShape: number[]; viewStrides: number[] } | undefined;
      if (!scatterAddPayload) throw new Error("stridedScatterAdd requires offset, viewShape, viewStrides in payload");
      return nodeBackend.ops.stridedScatterAdd(backendInputs[0], backendInputs[1], scatterAddPayload);
    }

    case "adamStep": {
      const adamPayload = node.payload as import("../backend/types").AdamStepConfig;
      if (!nodeBackend.ops.adamStep) throw new Error("adamStep not supported by backend");
      const adamResult = await nodeBackend.ops.adamStep(
        backendInputs[0], backendInputs[1], backendInputs[2], backendInputs[3],
        adamPayload,
      );
      const mStorage2 = createStorageHandle(node.device, adamResult.m);
      const vStorage2 = createStorageHandle(node.device, adamResult.v);
      const sideOutputs2 = { m: mStorage2, v: vStorage2 };
      storageTracker.markReachable(mStorage2.id, sideOutputs2);
      storageTracker.markReachable(vStorage2.id, sideOutputs2);
      (node as any)._adamSideOutputs = sideOutputs2;
      return adamResult.param;
    }

    case "unscaleGrad": {
      const unscalePayload2 = node.payload as { invScale: number; infFlagBuffer: unknown };
      if (!nodeBackend.ops.unscaleGrad) throw new Error("unscaleGrad not supported by backend");
      return nodeBackend.ops.unscaleGrad(
        backendInputs[0], unscalePayload2.invScale, unscalePayload2.infFlagBuffer,
      );
    }

    case "fusedAttentionForward": {
      const faPayload2 = node.payload as import("../backend/types").FusedAttentionConfig;
      if (!nodeBackend.ops.fusedAttentionForward) throw new Error("fusedAttentionForward not supported by backend");
      const faResult2 = nodeBackend.ops.fusedAttentionForward(
        backendInputs[0], backendInputs[1], backendInputs[2], faPayload2,
      );
      const lseSH2 = createStorageHandle(node.device, faResult2.logsumexp);
      (node as any)._attnSideOutput = lseSH2;
      return faResult2.output;
    }

    case "extractAttentionLogsumexp": {
      const parentNodeFA2 = node.inputs[0].node;
      const lseSH2 = (parentNodeFA2 as any)._attnSideOutput as StorageHandle | undefined;
      if (!lseSH2) throw new Error("extractAttentionLogsumexp: parent has no _attnSideOutput");
      const lseResult2 = lseSH2.backendTensor;
      storageTracker.unregister(lseSH2.id);
      (parentNodeFA2 as any)._attnSideOutput = undefined;
      return lseResult2;
    }

    case "fusedAttentionBackward": {
      const faBwdPayload2 = node.payload as import("../backend/types").FusedAttentionConfig;
      if (!nodeBackend.ops.fusedAttentionBackward) throw new Error("fusedAttentionBackward not supported by backend");
      const faBwdResult2 = nodeBackend.ops.fusedAttentionBackward(
        backendInputs[0], backendInputs[1], backendInputs[2],
        backendInputs[3], backendInputs[4], backendInputs[5], faBwdPayload2,
      );
      const dkSH2 = createStorageHandle(node.device, faBwdResult2.dK);
      const dvSH2 = createStorageHandle(node.device, faBwdResult2.dV);
      (node as any)._attnBwdDK = dkSH2;
      (node as any)._attnBwdDV = dvSH2;
      return faBwdResult2.dQ;
    }

    case "extractAttentionDK": {
      const parentDK2 = node.inputs[0].node;
      const dkSH2 = (parentDK2 as any)._attnBwdDK as StorageHandle | undefined;
      if (!dkSH2) throw new Error("extractAttentionDK: parent has no _attnBwdDK");
      const dkResult2 = dkSH2.backendTensor;
      storageTracker.unregister(dkSH2.id);
      (parentDK2 as any)._attnBwdDK = undefined;
      return dkResult2;
    }

    case "extractAttentionDV": {
      const parentDV2 = node.inputs[0].node;
      const dvSH2 = (parentDV2 as any)._attnBwdDV as StorageHandle | undefined;
      if (!dvSH2) throw new Error("extractAttentionDV: parent has no _attnBwdDV");
      const dvResult2 = dvSH2.backendTensor;
      storageTracker.unregister(dvSH2.id);
      (parentDV2 as any)._attnBwdDV = undefined;
      return dvResult2;
    }

    case "fusedCrossEntropyForward": {
      const cePayload3 = node.payload as import("../backend/types").FusedCrossEntropyConfig;
      if (!nodeBackend.ops.fusedCrossEntropyForward) throw new Error("fusedCrossEntropyForward not supported by backend");
      return nodeBackend.ops.fusedCrossEntropyForward(
        backendInputs[0], backendInputs[1], cePayload3,
      );
    }

    case "fusedCrossEntropyBackward": {
      const cePayload4 = node.payload as import("../backend/types").FusedCrossEntropyConfig;
      if (!nodeBackend.ops.fusedCrossEntropyBackward) throw new Error("fusedCrossEntropyBackward not supported by backend");
      return nodeBackend.ops.fusedCrossEntropyBackward(
        backendInputs[0], backendInputs[1], backendInputs[2], cePayload4,
      );
    }

    case "fusedLayerNormForward": {
      const lnPayload3 = node.payload as import("../backend/types").FusedLayerNormConfig;
      if (!nodeBackend.ops.fusedLayerNormForward) throw new Error("fusedLayerNormForward not supported by backend");
      return nodeBackend.ops.fusedLayerNormForward(
        backendInputs[0], backendInputs[1], backendInputs[2], lnPayload3,
      );
    }

    case "fusedLayerNormBackwardGradX": {
      const lnPayload4 = node.payload as import("../backend/types").FusedLayerNormConfig;
      if (!nodeBackend.ops.fusedLayerNormBackwardGradX) throw new Error("fusedLayerNormBackwardGradX not supported by backend");
      return nodeBackend.ops.fusedLayerNormBackwardGradX(
        backendInputs[0], backendInputs[1], backendInputs[2], lnPayload4,
      );
    }

    case "fusedLayerNormBackwardGradWeightBias": {
      const lnGWBPayload2 = node.payload as import("../backend/types").FusedLayerNormConfig;
      if (!nodeBackend.ops.fusedLayerNormBackwardGradWeightBias) {
        throw new Error("fusedLayerNormBackwardGradWeightBias not supported by backend");
      }
      const gwResult2 = nodeBackend.ops.fusedLayerNormBackwardGradWeightBias(
        backendInputs[0], backendInputs[1], lnGWBPayload2,
      );
      // Store raw gradBias BackendTensor for extractLnBwdGradBias
      (node as any)._lnBwdSideOutput = gwResult2.gradBias;
      return gwResult2.gradWeight;
    }

    case "extractLnBwdGradBias": {
      const parentNode2 = node.inputs[0].node;
      const sideOutputBT2 = (parentNode2 as any)._lnBwdSideOutput as BackendTensor | undefined;
      if (!sideOutputBT2) {
        throw new Error("extractLnBwdGradBias: parent node has no _lnBwdSideOutput");
      }
      (parentNode2 as any)._lnBwdSideOutput = undefined;
      return sideOutputBT2;
    }

    case "transfer": {
      const transferPayload = node.payload as { targetDevice: DeviceKind } | undefined;
      if (!transferPayload?.targetDevice) throw new Error("transfer requires targetDevice in payload");
      const targetBackend = getBackend(transferPayload.targetDevice);
      if (!targetBackend) throw new Error(`No backend found for device: ${transferPayload.targetDevice}`);
      // Read from source backend, create in target backend
      const data = await nodeBackend.ops.read(backendInputs[0]);
      const sourceShape = (backendInputs[0] as { shape: number[] }).shape;
      return targetBackend.ops.tensorFromArray(data, sourceShape);
    }

    default:
      throw new Error(`Unknown op: ${node.op}`);
  }
  } finally {
    profileOpEnd(node.op, _profT0);
    setCurrentOpLabel(null);
    setProfileModule("unknown");
  }
}

// ============================================================================
// Sequential Plan Execution
// ============================================================================

export async function executePlan(
  plan: ExecutionPlan,
  backend: Backend,
  options?: ExecutePlanOptions,
): Promise<StorageHandle> {
  if (plan.nodes.length === 0) {
    throw new Error("Cannot execute empty plan");
  }

  // Pre-tune matmul shapes if backend supports it
  await pretunePlanMatmuls(plan, backend);

  // Set up lifetime analysis if early release is enabled
  let lifetimes: Map<number, TensorLifetime> | null = null;
  let outputNodeIds: Set<number> | null = null;
  const alreadyReleased = new Set<number>();
  const nodeToStorage = new Map<number, StorageHandle>();

  if (options?.enableEarlyRelease) {
    const { nodeOrder, nodeInputs, nodeSizes } = extractPlanMetadata(plan);
    const lastNodeId = plan.nodes[plan.nodes.length - 1].id;
    outputNodeIds = new Set([lastNodeId]);
    // Protect externally-referenced nodes (saved for backward, user-held tensors)
    // from early release — later plans need their buffers intact.
    try {
      const { getPendingNodeIds } = await import("../runtime/tensor");
      for (const id of getPendingNodeIds()) outputNodeIds.add(id);
    } catch { /* runtime/tensor not available */ }
    lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, outputNodeIds, nodeSizes);
  }

  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {

  const viewOnly = options?.viewOpsOnly === true;

  for (let step = 0; step < plan.nodes.length; step++) {
    const node = plan.nodes[step];

    // Skip nodes that already have results (from a prior plan execution within this step).
    if (node.result) continue;

    // In view-only mode (tape replay fill-in), skip compute ops.
    // Only view ops and data-source ops need execution; compute ops are
    // either already handled by replay or are intermediate fused nodes.
    if (viewOnly && !FILL_IN_OPS.has(node.op)) continue;

    // For multi-device graphs, use the node's device backend
    const nodeBackend = getBackend(node.device) ?? backend;

    const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
    const backendInputs = inputs.map((s) => s.backendTensor);

    let resultTensor: BackendTensor;

    setCurrentOpLabel(node.op);
    setProfileModule(node.module ?? "unknown");
    const _profT0 = profileOpBegin(node.op);

    switch (node.op) {
      case "tensorFromArray": {
        const payload = node.payload as { values: number[] | Float32Array } | undefined;
        if (!payload?.values) {
          throw new Error("tensorFromArray requires values in payload");
        }
        resultTensor = nodeBackend.ops.tensorFromArray(
          payload.values,
          node.shape,
        );
        break;
      }

      case "zeros": {
        if (nodeBackend.ops.zeros) {
          resultTensor = nodeBackend.ops.zeros(node.shape);
        } else {
          // Fallback for backends without dedicated zeros op
          const numElements = node.shape.reduce((a, b) => a * b, 1);
          resultTensor = nodeBackend.ops.tensorFromArray(
            new Array(numElements).fill(0),
            node.shape,
          );
        }
        break;
      }

      case "full": {
        const fullPayload = node.payload as { fillValue: number };
        if (nodeBackend.ops.full) {
          resultTensor = nodeBackend.ops.full(node.shape, fullPayload.fillValue);
        } else {
          // Fallback: create array filled with value
          const numElements = node.shape.reduce((a, b) => a * b, 1);
          resultTensor = nodeBackend.ops.tensorFromArray(
            new Array(numElements).fill(fullPayload.fillValue),
            node.shape,
          );
        }
        break;
      }

      case "arange": {
        const arangePayload = node.payload as { end: number; start: number; step: number };
        if (nodeBackend.ops.arange) {
          resultTensor = nodeBackend.ops.arange(arangePayload.end, arangePayload.start, arangePayload.step);
        } else {
          // Fallback: create array on CPU
          const n = Math.max(0, Math.ceil((arangePayload.end - arangePayload.start) / arangePayload.step));
          const vals = new Array(n);
          for (let i = 0; i < n; i++) vals[i] = arangePayload.start + i * arangePayload.step;
          resultTensor = nodeBackend.ops.tensorFromArray(vals, node.shape);
        }
        break;
      }

      case "tril": {
        if (!nodeBackend.ops.tril) throw new Error("tril not supported by backend");
        const trilK = (node.payload as { k: number })?.k ?? 0;
        resultTensor = nodeBackend.ops.tril(backendInputs[0], trilK);
        break;
      }

      case "triu": {
        if (!nodeBackend.ops.triu) throw new Error("triu not supported by backend");
        const triuK = (node.payload as { k: number })?.k ?? 0;
        resultTensor = nodeBackend.ops.triu(backendInputs[0], triuK);
        break;
      }

      case "rand": {
        const randPayload = node.payload as { seed: number };
        if (nodeBackend.ops.rand) {
          resultTensor = nodeBackend.ops.rand(node.shape, randPayload.seed);
        } else {
          const n = node.shape.reduce((a: number, b: number) => a * b, 1);
          const vals = new Array(n);
          for (let i = 0; i < n; i++) vals[i] = Math.random();
          resultTensor = nodeBackend.ops.tensorFromArray(vals, node.shape);
        }
        break;
      }

      case "randn": {
        const randnPayload = node.payload as { seed: number };
        if (nodeBackend.ops.randn) {
          resultTensor = nodeBackend.ops.randn(node.shape, randnPayload.seed);
        } else {
          const n = node.shape.reduce((a: number, b: number) => a * b, 1);
          const vals = new Array(n);
          for (let i = 0; i < n; i += 2) {
            const u1 = Math.random();
            const u2 = Math.random();
            const r = Math.sqrt(-2 * Math.log(u1 || 1e-10));
            const theta = 2 * Math.PI * u2;
            vals[i] = r * Math.cos(theta);
            if (i + 1 < n) vals[i + 1] = r * Math.sin(theta);
          }
          resultTensor = nodeBackend.ops.tensorFromArray(vals, node.shape);
        }
        break;
      }

      case "bernoulli": {
        const bernPayload = node.payload as { seed: number; p: number };
        if (nodeBackend.ops.bernoulli) {
          resultTensor = nodeBackend.ops.bernoulli(node.shape, bernPayload.p, bernPayload.seed);
        } else {
          const n = node.shape.reduce((a: number, b: number) => a * b, 1);
          const vals = new Array(n);
          for (let i = 0; i < n; i++) vals[i] = Math.random() < bernPayload.p ? 1 : 0;
          resultTensor = nodeBackend.ops.tensorFromArray(vals, node.shape);
        }
        break;
      }

      case "add":
        resultTensor = nodeBackend.ops.add(backendInputs[0], backendInputs[1]);
        break;

      case "sub": {
        const subPayload = node.payload as { alpha?: number } | undefined;
        resultTensor = nodeBackend.ops.sub(
          backendInputs[0],
          backendInputs[1],
          subPayload,
        );
        break;
      }

      case "mul":
        resultTensor = nodeBackend.ops.mul(backendInputs[0], backendInputs[1]);
        break;

      case "div": {
        const divPayload = node.payload as
          | { roundingMode?: "trunc" | "floor" }
          | undefined;
        resultTensor = nodeBackend.ops.div(
          backendInputs[0],
          backendInputs[1],
          divPayload,
        );
        break;
      }

      case "matmul":
        resultTensor = nodeBackend.ops.matmul(
          backendInputs[0],
          backendInputs[1],
        );
        break;

      case "sqrt":
        resultTensor = nodeBackend.ops.sqrt(backendInputs[0]);
        break;

      case "relu":
        resultTensor = nodeBackend.ops.relu(backendInputs[0]);
        break;

      case "exp":
        if (!nodeBackend.ops.exp)
          throw new Error("exp not supported by backend");
        resultTensor = nodeBackend.ops.exp(backendInputs[0]);
        break;

      case "log":
        if (!nodeBackend.ops.log)
          throw new Error("log not supported by backend");
        resultTensor = nodeBackend.ops.log(backendInputs[0]);
        break;

      case "neg":
        if (!nodeBackend.ops.neg)
          throw new Error("neg not supported by backend");
        resultTensor = nodeBackend.ops.neg(backendInputs[0]);
        break;

      case "abs":
        if (!nodeBackend.ops.abs)
          throw new Error("abs not supported by backend");
        resultTensor = nodeBackend.ops.abs(backendInputs[0]);
        break;

      case "tanh":
        if (!nodeBackend.ops.tanh)
          throw new Error("tanh not supported by backend");
        resultTensor = nodeBackend.ops.tanh(backendInputs[0]);
        break;

      case "sigmoid":
        if (!nodeBackend.ops.sigmoid)
          throw new Error("sigmoid not supported by backend");
        resultTensor = nodeBackend.ops.sigmoid(backendInputs[0]);
        break;

      case "gelu": {
        if (!nodeBackend.ops.gelu)
          throw new Error("gelu not supported by backend");
        const geluOpts = node.payload as GeluOptions | undefined;
        resultTensor = nodeBackend.ops.gelu(backendInputs[0], geluOpts);
        break;
      }

      case "silu":
        if (!nodeBackend.ops.silu)
          throw new Error("silu not supported by backend");
        resultTensor = nodeBackend.ops.silu(backendInputs[0]);
        break;

      case "isfinite":
        if (!nodeBackend.ops.isfinite)
          throw new Error("isfinite not supported by backend");
        resultTensor = nodeBackend.ops.isfinite(backendInputs[0]);
        break;

      case "reshape": {
        const payload = node.payload as { targetShape: number[] } | undefined;
        const targetShape = payload?.targetShape ?? node.shape;
        resultTensor = nodeBackend.ops.reshape(backendInputs[0], targetShape);
        break;
      }

      case "expand":
        resultTensor = nodeBackend.ops.expand(backendInputs[0], node.shape);
        break;

      case "transpose": {
        const payload = node.payload as
          | { dim0: number; dim1: number }
          | undefined;
        if (!payload) {
          throw new Error("transpose requires dim0 and dim1 in payload");
        }
        resultTensor = nodeBackend.ops.transpose(backendInputs[0], payload);
        break;
      }

      case "permute": {
        const payload = node.payload as { dims: number[] } | undefined;
        if (!payload) {
          throw new Error("permute requires dims in payload");
        }
        resultTensor = nodeBackend.ops.permute(backendInputs[0], payload.dims);
        break;
      }

      case "contiguous":
        resultTensor = nodeBackend.ops.contiguous(backendInputs[0]);
        break;

      case "narrow": {
        const p = node.payload as { dim: number; start: number; length: number };
        if (!nodeBackend.ops.narrow) throw new Error("narrow not supported by backend");
        resultTensor = nodeBackend.ops.narrow(backendInputs[0], p.dim, p.start, p.length);
        break;
      }

      case "narrowBackward": {
        const p = node.payload as { dim: number; start: number; originalLength: number };
        if (!nodeBackend.ops.narrowBackward) throw new Error("narrowBackward not supported by backend");
        resultTensor = nodeBackend.ops.narrowBackward(backendInputs[0], p.dim, p.start, p.originalLength);
        break;
      }

      case "cast": {
        const payload = node.payload as
          | { dtype: import("../backend/types").DType }
          | undefined;
        if (!payload) {
          throw new Error("cast requires dtype in payload");
        }
        if (!nodeBackend.ops.cast) {
          throw new Error("cast not supported by backend");
        }
        resultTensor = nodeBackend.ops.cast(backendInputs[0], payload.dtype);
        break;
      }

      case "gather": {
        const payload = node.payload as { dim: number } | undefined;
        if (!payload) {
          throw new Error("gather requires dim in payload");
        }
        resultTensor = nodeBackend.ops.gather(
          backendInputs[0],
          backendInputs[1],
          payload,
        );
        break;
      }

      case "scatterAdd": {
        const payload = node.payload as { dim: number } | undefined;
        if (!payload) {
          throw new Error("scatterAdd requires dim in payload");
        }
        resultTensor = nodeBackend.ops.scatterAdd(
          backendInputs[0],
          backendInputs[1],
          backendInputs[2],
          payload,
        );
        break;
      }

      case "sum": {
        const payload = node.payload as
          | { dim?: number | number[] | null; keepdim?: boolean }
          | undefined;
        resultTensor = nodeBackend.ops.sum(backendInputs[0], payload);
        break;
      }

      case "max": {
        const payload = node.payload as
          | { dim?: number | number[] | null; keepdim?: boolean }
          | undefined;
        resultTensor = nodeBackend.ops.max(backendInputs[0], payload);
        break;
      }

      case "mean": {
        const payload = node.payload as
          | { dim?: number | number[] | null; keepdim?: boolean }
          | undefined;
        resultTensor = nodeBackend.ops.mean(backendInputs[0], payload);
        break;
      }

      case "argmax": {
        const payload = node.payload as { dim: number; keepdim?: boolean };
        if (!nodeBackend.ops.argmax)
          throw new Error("argmax not supported by backend");
        resultTensor = nodeBackend.ops.argmax(backendInputs[0], payload);
        break;
      }

      case "argmin": {
        const payload = node.payload as { dim: number; keepdim?: boolean };
        if (!nodeBackend.ops.argmin)
          throw new Error("argmin not supported by backend");
        resultTensor = nodeBackend.ops.argmin(backendInputs[0], payload);
        break;
      }

      case "gt":
        if (!nodeBackend.ops.gt) throw new Error("gt not supported by backend");
        resultTensor = nodeBackend.ops.gt(backendInputs[0], backendInputs[1]);
        break;

      case "lt":
        if (!nodeBackend.ops.lt) throw new Error("lt not supported by backend");
        resultTensor = nodeBackend.ops.lt(backendInputs[0], backendInputs[1]);
        break;

      case "ge":
        if (!nodeBackend.ops.ge) throw new Error("ge not supported by backend");
        resultTensor = nodeBackend.ops.ge(backendInputs[0], backendInputs[1]);
        break;

      case "le":
        if (!nodeBackend.ops.le) throw new Error("le not supported by backend");
        resultTensor = nodeBackend.ops.le(backendInputs[0], backendInputs[1]);
        break;

      case "eq":
        if (!nodeBackend.ops.eq) throw new Error("eq not supported by backend");
        resultTensor = nodeBackend.ops.eq(backendInputs[0], backendInputs[1]);
        break;

      case "ne":
        if (!nodeBackend.ops.ne) throw new Error("ne not supported by backend");
        resultTensor = nodeBackend.ops.ne(backendInputs[0], backendInputs[1]);
        break;

      case "where":
        resultTensor = nodeBackend.ops.where(
          backendInputs[0],
          backendInputs[1],
          backendInputs[2],
        );
        break;

      case "stridedScatterCopy": {
        const payload = node.payload as {
          offset: number;
          viewShape: number[];
          viewStrides: number[];
        };
        if (!payload) {
          throw new Error("stridedScatterCopy requires options in payload");
        }
        resultTensor = nodeBackend.ops.stridedScatterCopy(
          backendInputs[0],
          backendInputs[1],
          payload,
        );
        break;
      }

      case "stridedScatterAdd": {
        const payload = node.payload as {
          offset: number;
          viewShape: number[];
          viewStrides: number[];
        };
        if (!payload) {
          throw new Error("stridedScatterAdd requires options in payload");
        }
        resultTensor = nodeBackend.ops.stridedScatterAdd(
          backendInputs[0],
          backendInputs[1],
          payload,
        );
        break;
      }

      case "adamStep": {
        const adamPayload = node.payload as import("../backend/types").AdamStepConfig;
        if (!nodeBackend.ops.adamStep) {
          throw new Error("adamStep not supported by backend");
        }
        const adamResult = await nodeBackend.ops.adamStep(
          backendInputs[0], backendInputs[1], backendInputs[2], backendInputs[3],
          adamPayload,
        );
        resultTensor = adamResult.param;
        // Wrap side outputs in StorageHandles so they're tracked by the storage system
        // and won't be destroyed prematurely by the buffer pool.
        // Mark them as externally reachable so forceAllPending's destroyUnreachableSince
        // won't reclaim them before the optimizer extracts them on the next step.
        // Pass the sideOutputs object as the WeakRef target so destroyUnreachable()
        // can detect orphaned side outputs via GC (the node holds _adamSideOutputs
        // alive, and Adam._pendingNodes holds the node alive until _resolvePendingState).
        const mStorage = createStorageHandle(node.device, adamResult.m);
        const vStorage = createStorageHandle(node.device, adamResult.v);
        const sideOutputs = { m: mStorage, v: vStorage };
        storageTracker.markReachable(mStorage.id, sideOutputs);
        storageTracker.markReachable(vStorage.id, sideOutputs);
        (node as any)._adamSideOutputs = sideOutputs;
        break;
      }

      case "unscaleGrad": {
        const unscalePayload = node.payload as { invScale: number; infFlagBuffer: unknown };
        if (!nodeBackend.ops.unscaleGrad) throw new Error("unscaleGrad not supported by backend");
        resultTensor = nodeBackend.ops.unscaleGrad(
          backendInputs[0], unscalePayload.invScale, unscalePayload.infFlagBuffer,
        );
        break;
      }

      case "fusedAttentionForward": {
        const faPayload = node.payload as import("../backend/types").FusedAttentionConfig;
        if (!nodeBackend.ops.fusedAttentionForward) throw new Error("fusedAttentionForward not supported by backend");
        const faResult = nodeBackend.ops.fusedAttentionForward(
          backendInputs[0], backendInputs[1], backendInputs[2], faPayload,
        );
        resultTensor = faResult.output;
        // Wrap logsumexp in a StorageHandle so destroyUnreachable() can clean it up
        // if extractAttentionLogsumexp never executes (e.g., checkpoint disposes the
        // consumer before the plan runs). Without this, the raw GPUBuffer leaks because
        // allocateOutputBuffer called trackAllocation but nothing calls trackDeallocation.
        const lseSH = createStorageHandle(node.device, faResult.logsumexp);
        (node as any)._attnSideOutput = lseSH;
        break;
      }

      case "extractAttentionLogsumexp": {
        const parentNodeFA = node.inputs[0].node;
        const lseSH = (parentNodeFA as any)._attnSideOutput as StorageHandle | undefined;
        if (!lseSH) {
          throw new Error("extractAttentionLogsumexp: parent node has no _attnSideOutput");
        }
        resultTensor = lseSH.backendTensor;
        // Unregister the StorageHandle — the engine will create a new one for this node's output
        storageTracker.unregister(lseSH.id);
        (parentNodeFA as any)._attnSideOutput = undefined;
        break;
      }

      case "fusedAttentionBackward": {
        const faBwdPayload = node.payload as import("../backend/types").FusedAttentionConfig;
        if (!nodeBackend.ops.fusedAttentionBackward) throw new Error("fusedAttentionBackward not supported by backend");
        const faBwdResult = nodeBackend.ops.fusedAttentionBackward(
          backendInputs[0], backendInputs[1], backendInputs[2],
          backendInputs[3], backendInputs[4], backendInputs[5], faBwdPayload,
        );
        resultTensor = faBwdResult.dQ;
        // Wrap dK/dV in StorageHandles for proper cleanup (same reason as logsumexp above)
        const dkSH = createStorageHandle(node.device, faBwdResult.dK);
        const dvSH = createStorageHandle(node.device, faBwdResult.dV);
        (node as any)._attnBwdDK = dkSH;
        (node as any)._attnBwdDV = dvSH;
        break;
      }

      case "extractAttentionDK": {
        const parentNodeDK = node.inputs[0].node;
        const dkSH = (parentNodeDK as any)._attnBwdDK as StorageHandle | undefined;
        if (!dkSH) {
          throw new Error("extractAttentionDK: parent node has no _attnBwdDK");
        }
        resultTensor = dkSH.backendTensor;
        storageTracker.unregister(dkSH.id);
        (parentNodeDK as any)._attnBwdDK = undefined;
        break;
      }

      case "extractAttentionDV": {
        const parentNodeDV = node.inputs[0].node;
        const dvSH = (parentNodeDV as any)._attnBwdDV as StorageHandle | undefined;
        if (!dvSH) {
          throw new Error("extractAttentionDV: parent node has no _attnBwdDV");
        }
        resultTensor = dvSH.backendTensor;
        storageTracker.unregister(dvSH.id);
        (parentNodeDV as any)._attnBwdDV = undefined;
        break;
      }

      case "fusedCrossEntropyForward": {
        const cePayload = node.payload as import("../backend/types").FusedCrossEntropyConfig;
        if (!nodeBackend.ops.fusedCrossEntropyForward) throw new Error("fusedCrossEntropyForward not supported by backend");
        resultTensor = nodeBackend.ops.fusedCrossEntropyForward(
          backendInputs[0], backendInputs[1], cePayload,
        );
        break;
      }

      case "fusedCrossEntropyBackward": {
        const cePayload2 = node.payload as import("../backend/types").FusedCrossEntropyConfig;
        if (!nodeBackend.ops.fusedCrossEntropyBackward) throw new Error("fusedCrossEntropyBackward not supported by backend");
        resultTensor = nodeBackend.ops.fusedCrossEntropyBackward(
          backendInputs[0], backendInputs[1], backendInputs[2], cePayload2,
        );
        break;
      }

      case "fusedLayerNormForward": {
        const lnPayload = node.payload as import("../backend/types").FusedLayerNormConfig;
        if (!nodeBackend.ops.fusedLayerNormForward) throw new Error("fusedLayerNormForward not supported by backend");
        resultTensor = nodeBackend.ops.fusedLayerNormForward(
          backendInputs[0], backendInputs[1], backendInputs[2], lnPayload,
        );
        break;
      }

      case "fusedLayerNormBackwardGradX": {
        const lnPayload2 = node.payload as import("../backend/types").FusedLayerNormConfig;
        if (!nodeBackend.ops.fusedLayerNormBackwardGradX) throw new Error("fusedLayerNormBackwardGradX not supported by backend");
        resultTensor = nodeBackend.ops.fusedLayerNormBackwardGradX(
          backendInputs[0], backendInputs[1], backendInputs[2], lnPayload2,
        );
        break;
      }

      case "fusedLayerNormBackwardGradWeightBias": {
        const lnGWBPayload = node.payload as import("../backend/types").FusedLayerNormConfig;
        if (!nodeBackend.ops.fusedLayerNormBackwardGradWeightBias) {
          throw new Error("fusedLayerNormBackwardGradWeightBias not supported by backend");
        }
        const gwResult = nodeBackend.ops.fusedLayerNormBackwardGradWeightBias(
          backendInputs[0], backendInputs[1], lnGWBPayload,
        );
        resultTensor = gwResult.gradWeight;
        // Store raw gradBias BackendTensor for extractLnBwdGradBias
        (node as any)._lnBwdSideOutput = gwResult.gradBias;
        break;
      }

      case "extractLnBwdGradBias": {
        // Input[0] is the gradWeight tensor from fusedLayerNormBackwardGradWeightBias
        const parentNode = node.inputs[0].node;
        const sideOutputBT = (parentNode as any)._lnBwdSideOutput as BackendTensor | undefined;
        if (!sideOutputBT) {
          throw new Error("extractLnBwdGradBias: parent node has no _lnBwdSideOutput");
        }
        resultTensor = sideOutputBT;
        (parentNode as any)._lnBwdSideOutput = undefined;
        break;
      }

      case "transfer": {
        // Transfer from source device to target device
        const sourceStorage = inputs[0];
        const targetDevice = node.device;
        const sourceDevice = sourceStorage.device;

        if (sourceDevice === targetDevice) {
          // No transfer needed
          resultTensor = sourceStorage.backendTensor;
        } else {
          // Get target backend and transfer via CPU
          const targetBackend = getBackend(targetDevice);
          if (!targetBackend) {
            throw new Error(
              `Transfer failed: backend not available for ${targetDevice}`,
            );
          }

          // Read from source, create on target
          const sourceBackend = getBackend(sourceDevice);
          if (!sourceBackend) {
            throw new Error(
              `Transfer failed: backend not available for ${sourceDevice}`,
            );
          }

          const values = await sourceBackend.ops.read(
            sourceStorage.backendTensor,
          );
          resultTensor = targetBackend.ops.tensorFromArray(values, node.shape);
        }
        break;
      }

      default:
        throw new Error(`Unknown op: ${node.op}`);
    }

    profileOpEnd(node.op, _profT0);
    setCurrentOpLabel(null);
    setProfileModule("unknown");

    // Safety: if a backend op returned the exact same tensor object as one of
    // its inputs (e.g. contiguous on an already-contiguous tensor), creating a
    // separate owning StorageHandle would double-free the underlying buffer.
    // Detect this and wrap the result as a non-owning view.
    const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
    if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
      // Clone the tensor object so we don't mutate the input's ownsBuffer field.
      // Only applies to backends with explicit ownsBuffer (e.g. WebGPU).
      resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
    }
    const isView =
      (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
    const baseStorageId =
      isView && inputs.length > 0
        ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
        : undefined;
    node.result = createStorageHandle(node.device, resultTensor, baseStorageId);

    // Track storage for early release
    if (options?.enableEarlyRelease) {
      nodeToStorage.set(node.id, node.result);

      // Release dead buffers after each step
      if (lifetimes && outputNodeIds) {
        const deadNodeIds = findDeadTensorsAtStep(
          lifetimes,
          step + 1, // We just completed this step
          outputNodeIds,
          alreadyReleased,
        );
        for (const deadId of deadNodeIds) {
          const storage = nodeToStorage.get(deadId);
          if (storage && canSafelyRelease(storage, nodeToStorage)) {
            releaseBufferImmediate(storage);
            nodeToStorage.delete(deadId);
            alreadyReleased.add(deadId);
          }
        }
      }
    }
  }

  } finally {
    if (useSharedEncoder) endSharedEncoder();
  }

  const lastNode = plan.nodes[plan.nodes.length - 1];
  if (!lastNode.result) {
    throw new Error("Execution failed: no result for last node");
  }

  // Clear results for nodes whose buffers were destroyed by early release.
  // Later plans skip nodes with results (if (node.result) continue), so stale
  // results pointing to destroyed buffers would cause silent data corruption.
  if (alreadyReleased.size > 0) {
    for (const node of plan.nodes) {
      if (alreadyReleased.has(node.id)) {
        node.result = undefined;
      }
    }
  }

  return lastNode.result;
}

// ============================================================================
// Segmented Execution for Checkpointing
// ============================================================================

/**
 * Execute a plan with segmentation at checkpoint boundaries.
 *
 * This enables memory savings for large models by:
 * 1. Executing each segment (up to a checkpoint boundary)
 * 2. Flushing the buffer pool after each segment
 * 3. Making released buffers available for subsequent segments
 *
 * @param plan - The execution plan
 * @param backend - The backend to use
 * @param options - Execution options
 * @param flushBufferPool - Callback to flush the buffer pool (backend-specific)
 */
export async function executePlanWithCheckpointSegments(
  plan: ExecutionPlan,
  backend: Backend,
  options: ExecutePlanOptions | undefined,
  flushBufferPool: () => void,
): Promise<StorageHandle> {
  // Check if plan has any checkpoint boundaries
  const hasCheckpointBoundaries = plan.nodes.some((n) => n.isCheckpointBoundary);

  if (!hasCheckpointBoundaries) {
    // No segmentation needed - use regular execution
    return executePlan(plan, backend, options);
  }

  // Segment the plan at checkpoint boundaries
  const segments = segmentPlanAtCheckpoints(plan);

  if (segments.length === 1) {
    // Only one segment - use regular execution
    return executePlan(plan, backend, options);
  }

  // Execute each segment, flushing buffers between them
  let lastResult: StorageHandle | null = null;

  // Track all materialized storages across segments
  const materializedStorages = new Map<number, StorageHandle>();

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];

    // Clone node inputs before mutation to preserve plan immutability.
    // Nodes are shared across steps; mutating inputs in-place would cause
    // stale storage refs to accumulate, producing NaN after ~25 steps.
    for (const node of segment.nodes) {
      let needsClone = false;
      for (let j = 0; j < node.inputs.length; j++) {
        const input = node.inputs[j];
        if (input.kind === "pending") {
          const materialized = materializedStorages.get(input.node.id);
          if (materialized) {
            if (!needsClone) {
              node.inputs = [...node.inputs];
              needsClone = true;
            }
            node.inputs[j] = { kind: "materialized", storage: materialized };
          }
        }
      }
    }

    // Execute this segment
    lastResult = await executePlan(segment, backend, options);

    // Track all materialized results from this segment
    for (const node of segment.nodes) {
      if (node.result) {
        materializedStorages.set(node.id, node.result);
      }
    }

    // Flush buffer pool after each segment (except the last)
    // This makes released buffers available for the next segment
    if (i < segments.length - 1) {
      flushBufferPool();
    }
  }

  if (!lastResult) {
    throw new Error("Segmented execution failed: no result");
  }

  return lastResult;
}

// ============================================================================
// True Segmented Execution with GPU Synchronization
// ============================================================================

/**
 * Execute a plan with true segmented execution using GPU synchronization.
 *
 * Unlike executePlanWithCheckpointSegments which just flushes the buffer pool,
 * this version:
 * 1. Batches all ops in a segment into a single command buffer
 * 2. Submits and waits for GPU completion between segments
 * 3. Actually frees GPU memory before next segment starts
 *
 * This enables running models that wouldn't fit in GPU memory when using
 * checkpoint-based training.
 *
 * @param plan - The execution plan
 * @param backend - The backend to use
 * @param options - Execution options
 */
export async function executePlanWithTrueSegments(
  plan: ExecutionPlan,
  backend: Backend,
  options?: ExecutePlanOptions,
): Promise<StorageHandle> {
  // Check if plan has any checkpoint boundaries
  const hasCheckpointBoundaries = plan.nodes.some((n) => n.isCheckpointBoundary);

  if (!hasCheckpointBoundaries) {
    // No segmentation needed - use regular execution
    return executePlan(plan, backend, options);
  }

  // Segment the plan at checkpoint boundaries
  const segments = segmentPlanAtCheckpoints(plan);

  if (segments.length === 1) {
    // Only one segment - use regular execution
    return executePlan(plan, backend, options);
  }

  // Track cross-segment data flow
  const materializedStorages = new Map<number, StorageHandle>();
  let lastResult: StorageHandle | null = null;
  const finalOutputId = plan.nodes[plan.nodes.length - 1].id;

  for (let segIdx = 0; segIdx < segments.length; segIdx++) {
    const segment = segments[segIdx];
    const isLastSegment = segIdx === segments.length - 1;

    // Find outputs needed by later segments
    const survivingNodeIds = findSurvivingOutputs(
      segment,
      segments.slice(segIdx + 1),
      finalOutputId,
    );

    // Clone node inputs before mutation to preserve plan immutability.
    for (const node of segment.nodes) {
      let needsClone = false;
      for (let j = 0; j < node.inputs.length; j++) {
        const input = node.inputs[j];
        if (input.kind === "pending") {
          const materialized = materializedStorages.get(input.node.id);
          if (materialized) {
            if (!needsClone) {
              node.inputs = [...node.inputs];
              needsClone = true;
            }
            node.inputs[j] = { kind: "materialized", storage: materialized };
          }
        }
      }
    }

    // Begin batched execution - all ops encode to shared command buffer
    beginBatchExecution();

    try {
      const nodeToStorage = new Map<number, StorageHandle>();

      // Execute all ops in segment (encode to shared encoder, no GPU submit yet)
      for (const node of segment.nodes) {
        const nodeBackend = getBackend(node.device) ?? backend;
        const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
        const backendInputs = inputs.map((s) => s.backendTensor);

        let resultTensor = await executeOpInternal(
          node,
          backendInputs,
          nodeBackend,
        );

        // Safety: detect aliased result (e.g. contiguous on contiguous tensor)
        const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
        if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
          resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
        }
        const isView = (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
        const baseStorageId = isView && inputs.length > 0
          ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
          : undefined;
        node.result = createStorageHandleInternal(node.device, resultTensor, baseStorageId);
        nodeToStorage.set(node.id, node.result);
        materializedStorages.set(node.id, node.result);
      }

      // End batch - submits command buffer and WAITS for GPU completion
      await endBatchExecution();

      // NOW safe to release dead buffers (GPU work is complete)
      if (!isLastSegment) {
        for (const node of segment.nodes) {
          if (!survivingNodeIds.has(node.id)) {
            const storage = nodeToStorage.get(node.id);
            if (storage && canSafelyRelease(storage, nodeToStorage)) {
              releaseBufferImmediate(storage);
              nodeToStorage.delete(node.id);
              materializedStorages.delete(node.id);
            }
          }
        }

        // Flush buffer pool - buffers now available for next segment
        flushBufferPool();
      }

      lastResult = segment.nodes[segment.nodes.length - 1].result!;

    } catch (error) {
      // Clean up batch on error
      if (isBatchActive()) {
        abortBatch();
      }
      throw error;
    }
  }

  if (!lastResult) {
    throw new Error("True segmented execution failed: no result");
  }

  return lastResult;
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Find node IDs that must survive this segment (used by later segments).
 */
export function findSurvivingOutputs(
  segment: ExecutionPlan,
  laterSegments: ExecutionPlan[],
  finalOutputId: number,
): Set<number> {
  const surviving = new Set<number>();
  surviving.add(finalOutputId);

  // Find all nodes from this segment that are used as inputs in later segments
  for (const laterSegment of laterSegments) {
    for (const node of laterSegment.nodes) {
      for (const input of node.inputs) {
        if (input.kind === "pending") {
          surviving.add(input.node.id);
        }
      }
    }
  }

  return surviving;
}
