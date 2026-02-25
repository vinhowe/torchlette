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
import { profileOpBegin, profileOpEnd, setProfileModule } from "../backend/webgpu/profiler";
import type { LazyIRNode, LazyRef, StorageHandle } from "./lazy-types";
import { createStorageHandle } from "./node-factory";
import { storageTracker } from "./storage-tracker";

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
  switch (node.op) {
    case "tensorFromArray": {
      const payload = node.payload as { values: number[] | Float32Array } | undefined;
      if (!payload?.values) {
        throw new Error("tensorFromArray requires values in payload");
      }
      return backend.ops.tensorFromArray(payload.values, node.shape);
    }

    case "zeros": {
      if (backend.ops.zeros) {
        return backend.ops.zeros(node.shape);
      }
      const numEl = node.shape.reduce((a: number, b: number) => a * b, 1);
      return backend.ops.tensorFromArray(new Array(numEl).fill(0), node.shape);
    }

    case "full": {
      const fullPayload = node.payload as { fillValue: number };
      if (backend.ops.full) {
        return backend.ops.full(node.shape, fullPayload.fillValue);
      }
      const numElFull = node.shape.reduce((a: number, b: number) => a * b, 1);
      return backend.ops.tensorFromArray(new Array(numElFull).fill(fullPayload.fillValue), node.shape);
    }

    case "arange": {
      const ap = node.payload as { end: number; start: number; step: number };
      if (backend.ops.arange) {
        return backend.ops.arange(ap.end, ap.start, ap.step);
      }
      const n = Math.max(0, Math.ceil((ap.end - ap.start) / ap.step));
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = ap.start + i * ap.step;
      return backend.ops.tensorFromArray(vals, node.shape);
    }

    case "tril": {
      if (!backend.ops.tril) throw new Error("tril not supported by backend");
      return backend.ops.tril(backendInputs[0], (node.payload as { k: number })?.k ?? 0);
    }

    case "triu": {
      if (!backend.ops.triu) throw new Error("triu not supported by backend");
      return backend.ops.triu(backendInputs[0], (node.payload as { k: number })?.k ?? 0);
    }

    case "rand": {
      const rp = node.payload as { seed: number };
      if (backend.ops.rand) return backend.ops.rand(node.shape, rp.seed);
      const n = node.shape.reduce((a: number, b: number) => a * b, 1);
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = Math.random();
      return backend.ops.tensorFromArray(vals, node.shape);
    }

    case "randn": {
      const rp = node.payload as { seed: number };
      if (backend.ops.randn) return backend.ops.randn(node.shape, rp.seed);
      const n = node.shape.reduce((a: number, b: number) => a * b, 1);
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
      const bp = node.payload as { seed: number; p: number };
      if (backend.ops.bernoulli) return backend.ops.bernoulli(node.shape, bp.p, bp.seed);
      const n = node.shape.reduce((a: number, b: number) => a * b, 1);
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = Math.random() < bp.p ? 1 : 0;
      return backend.ops.tensorFromArray(vals, node.shape);
    }

    case "add":
      return backend.ops.add(backendInputs[0], backendInputs[1], donationOpts);

    case "sub": {
      const subPayload = node.payload as { alpha?: number } | undefined;
      return backend.ops.sub(backendInputs[0], backendInputs[1], { ...subPayload, ...donationOpts });
    }

    case "mul":
      return backend.ops.mul(backendInputs[0], backendInputs[1], donationOpts);

    case "div": {
      const divPayload = node.payload as
        | { roundingMode?: "trunc" | "floor" }
        | undefined;
      return backend.ops.div(backendInputs[0], backendInputs[1], { ...divPayload, ...donationOpts });
    }

    case "matmul":
      return backend.ops.matmul(backendInputs[0], backendInputs[1], donationOpts);

    case "sqrt":
      return backend.ops.sqrt(backendInputs[0], donationOpts);

    case "relu":
      return backend.ops.relu(backendInputs[0], donationOpts);

    case "exp":
      if (!backend.ops.exp) throw new Error("exp not supported by backend");
      return backend.ops.exp(backendInputs[0], donationOpts);

    case "log":
      if (!backend.ops.log) throw new Error("log not supported by backend");
      return backend.ops.log(backendInputs[0], donationOpts);

    case "neg":
      if (!backend.ops.neg) throw new Error("neg not supported by backend");
      return backend.ops.neg(backendInputs[0], donationOpts);

    case "abs":
      if (!backend.ops.abs) throw new Error("abs not supported by backend");
      return backend.ops.abs(backendInputs[0], donationOpts);

    case "tanh":
      if (!backend.ops.tanh) throw new Error("tanh not supported by backend");
      return backend.ops.tanh(backendInputs[0], donationOpts);

    case "sigmoid":
      if (!backend.ops.sigmoid)
        throw new Error("sigmoid not supported by backend");
      return backend.ops.sigmoid(backendInputs[0], donationOpts);

    case "gelu": {
      if (!backend.ops.gelu) throw new Error("gelu not supported by backend");
      const geluOpts = node.payload as GeluOptions | undefined;
      return backend.ops.gelu(backendInputs[0], { ...geluOpts, ...donationOpts });
    }

    case "silu":
      if (!backend.ops.silu) throw new Error("silu not supported by backend");
      return backend.ops.silu(backendInputs[0], donationOpts);

    case "isfinite":
      if (!backend.ops.isfinite)
        throw new Error("isfinite not supported by backend");
      return backend.ops.isfinite(backendInputs[0]);

    case "pow":
      if (!backend.ops.pow) throw new Error("pow not supported by backend");
      return backend.ops.pow(backendInputs[0], backendInputs[1]);

    case "reshape": {
      const payload = node.payload as { targetShape: number[] } | undefined;
      const targetShape = payload?.targetShape ?? node.shape;
      return backend.ops.reshape(backendInputs[0], targetShape);
    }

    case "expand":
      return backend.ops.expand(backendInputs[0], node.shape);

    case "transpose": {
      const payload = node.payload as
        | { dim0: number; dim1: number }
        | undefined;
      if (!payload) {
        throw new Error("transpose requires dim0 and dim1 in payload");
      }
      return backend.ops.transpose(backendInputs[0], payload);
    }

    case "permute": {
      const payload = node.payload as { dims: number[] } | undefined;
      if (!payload) {
        throw new Error("permute requires dims in payload");
      }
      return backend.ops.permute(backendInputs[0], payload.dims);
    }

    case "contiguous":
      return backend.ops.contiguous(backendInputs[0]);

    case "narrow": {
      const p = node.payload as { dim: number; start: number; length: number };
      if (!backend.ops.narrow) throw new Error("narrow not supported by backend");
      return backend.ops.narrow(backendInputs[0], p.dim, p.start, p.length);
    }

    case "narrowBackward": {
      const p = node.payload as { dim: number; start: number; originalLength: number };
      if (!backend.ops.narrowBackward) throw new Error("narrowBackward not supported by backend");
      return backend.ops.narrowBackward(backendInputs[0], p.dim, p.start, p.originalLength);
    }

    case "cast": {
      const payload = node.payload as
        | { dtype: DType }
        | undefined;
      if (!payload) {
        throw new Error("cast requires dtype in payload");
      }
      if (!backend.ops.cast) {
        throw new Error("cast not supported by backend");
      }
      return backend.ops.cast(backendInputs[0], payload.dtype);
    }

    case "gather": {
      const payload = node.payload as { dim: number } | undefined;
      if (!payload) {
        throw new Error("gather requires dim in payload");
      }
      return backend.ops.gather(backendInputs[0], backendInputs[1], payload);
    }

    case "scatterAdd": {
      const payload = node.payload as { dim: number } | undefined;
      if (!payload) {
        throw new Error("scatterAdd requires dim in payload");
      }
      return backend.ops.scatterAdd(
        backendInputs[0],
        backendInputs[1],
        backendInputs[2],
        payload,
      );
    }

    case "sum": {
      const payload = node.payload as
        | { dim?: number | number[] | null; keepdim?: boolean }
        | undefined;
      return backend.ops.sum(backendInputs[0], payload);
    }

    case "max": {
      const payload = node.payload as
        | { dim?: number | number[] | null; keepdim?: boolean }
        | undefined;
      return backend.ops.max(backendInputs[0], payload);
    }

    case "mean": {
      const payload = node.payload as
        | { dim?: number | number[] | null; keepdim?: boolean }
        | undefined;
      return backend.ops.mean(backendInputs[0], payload);
    }

    case "argmax": {
      const payload = node.payload as { dim: number; keepdim?: boolean };
      if (!backend.ops.argmax)
        throw new Error("argmax not supported by backend");
      return backend.ops.argmax(backendInputs[0], payload);
    }

    case "argmin": {
      const payload = node.payload as { dim: number; keepdim?: boolean };
      if (!backend.ops.argmin)
        throw new Error("argmin not supported by backend");
      return backend.ops.argmin(backendInputs[0], payload);
    }

    case "gt":
      if (!backend.ops.gt) throw new Error("gt not supported by backend");
      return backend.ops.gt(backendInputs[0], backendInputs[1], donationOpts);

    case "lt":
      if (!backend.ops.lt) throw new Error("lt not supported by backend");
      return backend.ops.lt(backendInputs[0], backendInputs[1], donationOpts);

    case "ge":
      if (!backend.ops.ge) throw new Error("ge not supported by backend");
      return backend.ops.ge(backendInputs[0], backendInputs[1], donationOpts);

    case "le":
      if (!backend.ops.le) throw new Error("le not supported by backend");
      return backend.ops.le(backendInputs[0], backendInputs[1], donationOpts);

    case "eq":
      if (!backend.ops.eq) throw new Error("eq not supported by backend");
      return backend.ops.eq(backendInputs[0], backendInputs[1], donationOpts);

    case "ne":
      if (!backend.ops.ne) throw new Error("ne not supported by backend");
      return backend.ops.ne(backendInputs[0], backendInputs[1], donationOpts);

    case "where":
      return backend.ops.where(
        backendInputs[0],
        backendInputs[1],
        backendInputs[2],
        donationOpts,
      );

    case "stridedScatterCopy": {
      const payload = node.payload as {
        offset: number;
        viewShape: number[];
        viewStrides: number[];
      };
      if (!payload) {
        throw new Error("stridedScatterCopy requires options in payload");
      }
      return backend.ops.stridedScatterCopy(
        backendInputs[0],
        backendInputs[1],
        payload,
      );
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
      return backend.ops.stridedScatterAdd(
        backendInputs[0],
        backendInputs[1],
        payload,
      );
    }

    case "adamStep": {
      if (!backend.ops.adamStep) throw new Error("adamStep not supported by backend");
      const payload = node.payload as AdamStepConfig;
      const adamResult = await backend.ops.adamStep(
        backendInputs[0], backendInputs[1], backendInputs[2], backendInputs[3],
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
    }

    case "unscaleGrad": {
      const payload = node.payload as { invScale: number; infFlagBuffer: unknown };
      if (!backend.ops.unscaleGrad) throw new Error("unscaleGrad not supported by backend");
      return backend.ops.unscaleGrad(
        backendInputs[0], payload.invScale, payload.infFlagBuffer,
      );
    }

    case "fusedAttentionForward": {
      const payload = node.payload as FusedAttentionConfig;
      if (!backend.ops.fusedAttentionForward) throw new Error("fusedAttentionForward not supported by backend");
      const result = backend.ops.fusedAttentionForward(
        backendInputs[0], backendInputs[1], backendInputs[2], payload,
      );
      const lseSH = createStorageHandle(node.device, result.logsumexp);
      if (!node._sideOutputs) node._sideOutputs = {};
      node._sideOutputs.attnLogsumexp = lseSH;
      return result.output;
    }

    case "extractAttentionLogsumexp":
      return extractSideOutput(node, "attnLogsumexp");

    case "fusedAttentionBackward": {
      const payload = node.payload as FusedAttentionConfig;
      if (!backend.ops.fusedAttentionBackward) throw new Error("fusedAttentionBackward not supported by backend");
      const result = backend.ops.fusedAttentionBackward(
        backendInputs[0], backendInputs[1], backendInputs[2],
        backendInputs[3], backendInputs[4], backendInputs[5], payload,
      );
      const dkSH = createStorageHandle(node.device, result.dK);
      const dvSH = createStorageHandle(node.device, result.dV);
      if (!node._sideOutputs) node._sideOutputs = {};
      node._sideOutputs.attnBwdDK = dkSH;
      node._sideOutputs.attnBwdDV = dvSH;
      return result.dQ;
    }

    case "extractAttentionDK":
      return extractSideOutput(node, "attnBwdDK");

    case "extractAttentionDV":
      return extractSideOutput(node, "attnBwdDV");

    case "fusedCrossEntropyForward": {
      const payload = node.payload as FusedCrossEntropyConfig;
      if (!backend.ops.fusedCrossEntropyForward) throw new Error("fusedCrossEntropyForward not supported by backend");
      return backend.ops.fusedCrossEntropyForward(
        backendInputs[0], backendInputs[1], payload,
      );
    }

    case "fusedCrossEntropyBackward": {
      const payload = node.payload as FusedCrossEntropyConfig;
      if (!backend.ops.fusedCrossEntropyBackward) throw new Error("fusedCrossEntropyBackward not supported by backend");
      return backend.ops.fusedCrossEntropyBackward(
        backendInputs[0], backendInputs[1], backendInputs[2], payload,
      );
    }

    case "fusedLayerNormForward": {
      const payload = node.payload as FusedLayerNormConfig;
      if (!backend.ops.fusedLayerNormForward) throw new Error("fusedLayerNormForward not supported by backend");
      return backend.ops.fusedLayerNormForward(
        backendInputs[0], backendInputs[1], backendInputs[2], payload,
      );
    }

    case "fusedLayerNormBackwardGradX": {
      const payload = node.payload as FusedLayerNormConfig;
      if (!backend.ops.fusedLayerNormBackwardGradX) throw new Error("fusedLayerNormBackwardGradX not supported by backend");
      return backend.ops.fusedLayerNormBackwardGradX(
        backendInputs[0], backendInputs[1], backendInputs[2], payload,
      );
    }

    case "fusedLayerNormBackwardGradWeightBias": {
      const payload = node.payload as FusedLayerNormConfig;
      if (!backend.ops.fusedLayerNormBackwardGradWeightBias) {
        throw new Error("fusedLayerNormBackwardGradWeightBias not supported by backend");
      }
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
      // node.device = target device; payload.sourceDevice or input ref's device = source
      // backend parameter may be the target device's backend (from getBackend(node.device))
      // so we resolve source backend from the payload or input ref
      const transferPayload = node.payload as { sourceDevice?: DeviceKind } | undefined;
      const sourceDevice = transferPayload?.sourceDevice
        ?? (node.inputs[0]?.kind === "materialized" ? node.inputs[0].storage.device : undefined);
      const sourceBackend = sourceDevice ? getBackend(sourceDevice) : backend;
      if (!sourceBackend) throw new Error(`Transfer failed: backend not available for source device ${sourceDevice}`);
      const targetBackend = getBackend(node.device);
      if (!targetBackend) throw new Error(`Transfer failed: backend not available for ${node.device}`);
      if (sourceDevice === node.device) return backendInputs[0]; // no-op
      const data = await sourceBackend.ops.read(backendInputs[0]);
      return targetBackend.ops.tensorFromArray(data, node.shape);
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
