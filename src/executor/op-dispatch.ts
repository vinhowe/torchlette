import { getBackend } from "../backend/registry";
import type {
  AdamStepConfig,
  Backend,
  BackendTensor,
  DeviceKind,
  DType,
} from "../backend/types";
import { ENV } from "../core/env";
import { sizeOf } from "../core/shape";
import {
  createStorageHandle,
  wrapResultAsStorage,
} from "../graph/node-factory";
import {
  getCurrentOpLabel,
  profileOpBegin,
  profileOpEnd,
  setCurrentOpLabel,
  setProfileModule,
} from "../graph/profiler";
import { storageTracker } from "../graph/storage-tracker";
import type { LazyIRNode, LazyRef, StorageHandle } from "../graph/types";
import { OP_REGISTRY } from "../ops/registry";
import { isStepTapeReplayActive } from "./observed-liveness";
import { lookupScalarStorage } from "./scalar-table";

/** One-shot flag for the reclaimed-storage-read warning. */
let _warnedReclaimedRead = false;
let _warnedReleasedRead = false;

// ---------------------------------------------------------------------------
// Multi-output op protocol
// ---------------------------------------------------------------------------

/**
 * Explicit multi-output return value for op handlers like adamStep,
 * fusedAttentionForward, fusedLayerNormBackwardGradWeightBias, etc.
 *
 * Returning this from a handler tells the wrapper "this op has more than one
 * output buffer; here are all of them in order". The wrapper is then
 * responsible for creating StorageHandles and assigning both `node.result`
 * (= results[0]) and `node.results` consistently.
 *
 * Before this protocol existed, multi-output handlers poked `node.results`
 * as a side channel from inside the handler. That was the source of the
 * "Input not ready: adamStep[0]" bug: executeAdamStep set
 * `node.results = [null, m, v]` because the param result couldn't be wrapped
 * until after the handler returned, and any wrapper that forgot to backfill
 * `node.results[0]` would leave a `null` there, breaking the read fallback.
 */
export type MultiOutputResult = {
  /** Primary output (also bound to node.result and node.results[0]). */
  primary: BackendTensor;
  /** Side outputs in order — bound to node.results[1..N]. */
  extras: BackendTensor[];
};

export function isMultiOutputResult(
  v: BackendTensor | MultiOutputResult,
): v is MultiOutputResult {
  // BackendTensor objects don't have a `primary` field; MultiOutputResult does.
  return typeof v === "object" && v !== null && "primary" in v && "extras" in v;
}

/**
 * Assign the result of executing a single op to its lazy node, handling both
 * single-output (`BackendTensor`) and multi-output (`MultiOutputResult`)
 * handler returns uniformly. Use this from every callsite that wraps an
 * op-handler return into the lazy graph; it guarantees `node.result` and
 * `node.results[0]` agree.
 */
export function assignNodeResult(
  node: LazyIRNode,
  handlerResult: BackendTensor | MultiOutputResult,
  backendInputs: BackendTensor[],
  inputs: StorageHandle[],
): void {
  if (isMultiOutputResult(handlerResult)) {
    const primary = wrapResultAsStorage(
      node.device,
      handlerResult.primary,
      backendInputs,
      inputs,
    );
    node.result = primary;
    node.results = [
      primary,
      ...handlerResult.extras.map((bt) => createStorageHandle(node.device, bt)),
    ];
    return;
  }
  node.result = wrapResultAsStorage(
    node.device,
    handlerResult,
    backendInputs,
    inputs,
  );
}

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
    // Lifetime seam guard: a materialized ref whose storage was RECLAIMED
    // means a tensor outlived its buffer — the silent-UAF class (markStep's
    // step-scoped demotion reclaiming a mid-step-created tensor that user
    // code still holds; the buffer is back in the pool and may carry another
    // op's data). Warn by default; TORCHLETTE_STRICT_LIFETIME=1 throws.
    // [2b §5] Inside a multi-plan TAPE replay the whole step's dataflow is
    // DECLARED: cross-plan buffers re-bound by the planner are reachable for
    // the replay even though their recording-era handle was demoted at the
    // recording markStep. The per-handle liveness verdicts (releasedOverlay,
    // isDestroyed) are observation-layer facts that the declaration supersedes
    // — correctness proven by trajectory parity within the fp-noise floor. Skip
    // both guards during the replay; outside it they are unchanged.
    const declaredReplay = isStepTapeReplayActive();
    if (ref.storage.releasedOverlay && !declaredReplay) {
      const msg =
        `[lifetime] reading step-globally RELEASED storage id=${ref.storage.id} ` +
        `(shape=${JSON.stringify(ref.storage.backendTensor.shape)}). Its registry entry was ` +
        `overlaid by the last observed reader's temps (stage-3 B); the bytes ` +
        `may be garbage. This read was invisible to observation — report it.`;
      if (ENV.TORCHLETTE_STRICT_LIFETIME === "1") throw new Error(msg);
      if (!_warnedReleasedRead) {
        _warnedReleasedRead = true;
        console.warn(msg);
      }
    }
    if (storageTracker.isDestroyed(ref.storage.id) && !declaredReplay) {
      const msg =
        `[lifetime] reading RECLAIMED storage id=${ref.storage.id} ` +
        `(shape=${JSON.stringify(ref.storage.backendTensor.shape)}). A tensor created ` +
        `mid-step and held across markStep was demoted as a step temporary; ` +
        `its buffer may have been reused. Create persistent state outside ` +
        `the step, update it in place via copy_, or mark it with ` +
        `runtime.persist().`;
      if (ENV.TORCHLETTE_STRICT_LIFETIME === "1") {
        throw new Error(msg);
      }
      if (!_warnedReclaimedRead) {
        _warnedReclaimedRead = true;
        console.warn(msg);
      }
    }
    return ref.storage;
  }
  if (ref.kind === "scalar") {
    // Per-step-varying scalars resolve through the plan's scalar table: a
    // persistent buffer the executor refreshed from THIS step's value before
    // execution. No fill dispatch, stable buffer identity, and — critically —
    // value-independent compiled plans (the legacy full([], v) path baked the
    // value into the fill kernel's recorded params; see scalar-table.ts).
    const tableStorage = lookupScalarStorage(ref);
    if (tableStorage) return tableStorage;
    // Legacy fallback (CPU backends, non-f32 scalars, refs outside the plan)
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
  // node.result is a derived view of node.results[0], so a single lookup
  // through node.results handles both primary and side outputs uniformly.
  const idx = ref.outputIndex ?? 0;
  const sh = ref.node.results?.[idx];
  if (sh) {
    if (sh.releasedOverlay) {
      const msg = `[lifetime] reading step-globally RELEASED result node=${ref.node.id} op=${ref.node.op}[${idx}] — overlaid by the last observed reader's temps (stage-3 B).`;
      if (ENV.TORCHLETTE_STRICT_LIFETIME === "1") throw new Error(msg);
      if (!_warnedReleasedRead) {
        _warnedReleasedRead = true;
        console.warn(msg);
      }
    }
    return sh;
  }
  throw new Error(
    `Input not ready: node id=${ref.node.id} op=${ref.node.op}[${idx}] shape=${JSON.stringify(ref.node.shape)} caller=${new Error().stack?.split("\n")[2]?.trim()}`,
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

/** adamStep is async + multi-output (param, m, v). */
function executeAdamStep(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): Promise<MultiOutputResult> {
  assertOpSupported("adamStep", backend.ops.adamStep);
  const payload = requirePayload<AdamStepConfig>(node);
  return (async () => {
    const adamResult = await backend.ops.adamStep!(
      backendInputs[0],
      backendInputs[1],
      backendInputs[2],
      backendInputs[3],
      backendInputs[4], // t
      backendInputs[5], // lr
      payload,
    );
    // Return all three outputs explicitly. The wrapper at the call site
    // (assignNodeResult) handles wrapping the param into an owning
    // StorageHandle and creating storage handles for m/v in one place,
    // so node.result and node.results[0] are guaranteed to agree.
    return {
      primary: adamResult.param,
      extras: [adamResult.m, adamResult.v],
    };
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
): BackendTensor | MultiOutputResult | Promise<BackendTensor> {
  const desc = FUSED_OP_TABLE[node.op];
  if (desc) {
    const payload = requirePayload(node);
    const fn = backend.ops[node.op as keyof Backend["ops"]] as
      | AnyOpFn
      | undefined;
    assertOpSupported(node.op, fn);
    const result = fn(...backendInputs, payload);

    if (desc.kind === "dispatch") return result as BackendTensor;

    // Multi-output: return primary + extras explicitly. The wrapper at
    // the call site (assignNodeResult) creates StorageHandles for all
    // outputs uniformly, so node.result and node.results[0] are
    // guaranteed to agree.
    const r = result as Record<string, BackendTensor>;
    return {
      primary: r[desc.returnField],
      extras: desc.extraFields.map((field) => r[field]),
    } satisfies MultiOutputResult;
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
) =>
  | BackendTensor
  | MultiOutputResult
  | Promise<BackendTensor | MultiOutputResult>;

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
 *
 * Returns a `BackendTensor` for single-output ops or a `MultiOutputResult`
 * for multi-output ops (adamStep, fusedAttention*, fusedLayerNormBackward*).
 * Use `assignNodeResult` at the call site to bind the result to the lazy node.
 */
export async function executeOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): Promise<BackendTensor | MultiOutputResult> {
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
 * Falls back to the handler's Promise return for async ops.
 *
 * Returns a `BackendTensor` for single-output ops or a `MultiOutputResult`
 * for multi-output ops. Use `assignNodeResult` at the call site to bind the
 * result to the lazy node.
 */
export function executeOpSync(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
):
  | BackendTensor
  | MultiOutputResult
  | Promise<BackendTensor | MultiOutputResult> {
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
