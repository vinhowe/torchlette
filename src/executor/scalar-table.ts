/**
 * Per-plan scalar table — per-step-varying scalars as DATA.
 *
 * Scalar LazyRefs (`mul(x, 1 - beta1 ** t)`) carry VALUES that may change
 * every step, but the plan-template fingerprint deliberately excludes scalar
 * values — only structure. Every cache keyed under that fingerprint must
 * therefore be value-independent, and historically three were not (the
 * frozen-scalar bug class, docs/architecture-debt.md):
 *   - the fused adamStep kernel's uniform (fixed via TAG_UNIFORM),
 *   - cached fusion recipes inlining scalar values into WGSL,
 *   - compiled-plan params slots baking the `full([], v)` fill dispatch
 *     that getInputStorage used to materialize scalar refs.
 *
 * This module repairs the contract for sequential execution: each template
 * gets ONE persistent 4-byte GPU buffer per scalar-ref position, wrapped in
 * a persistent 0-d StorageHandle. The executor refreshes the buffers from
 * the CURRENT step's node refs at the start of EVERY execution — lowered or
 * compiled replay — via queue.writeBuffer (queue-ordered before any later
 * submit, and the value is constant within a step, so all of this plan's
 * passes see the fresh value). `getInputStorage` then resolves scalar refs
 * to these storages instead of dispatching a value-baked fill: no dispatch,
 * stable buffer identity (bind-group cache hits), and the compiled replay
 * binds the same buffer whose contents the refresh just updated — correct
 * by construction, no record/replay hook needed.
 *
 * Positions (nodeIndex, inputIndex) are structural, so they are identical
 * across template reuses; the ref OBJECTS are per-step, so the ref→storage
 * map is rebuilt at each refresh.
 */

import { ENV } from "../core/env";
import type { Backend } from "../backend/types";
import type { GPUBuffer, GPUDevice } from "../backend/webgpu/gpu-types";
import { bufferPool } from "../backend/webgpu/buffer-pool";
import { GPUBufferUsage } from "../backend/webgpu/gpu-types";
import { flushSharedEncoder } from "../backend/webgpu/shared-encoder";
import { sharedEncoderActive } from "../backend/webgpu/webgpu-state";
import { createStorageHandle } from "../graph/node-factory";
import type { LazyIRNode, LazyRef, StorageHandle } from "../graph/types";

/** Structural position of a scalar ref within a plan. */
export interface ScalarSlot {
  nodeIndex: number;
  inputIndex: number;
}

/** Per-template scalar table state (lives on the LoweredPlan). */
export interface PlanScalarTable {
  buffers: GPUBuffer[];
  storages: StorageHandle[];
  /** CPU mirror of current buffer contents, for change detection. */
  values: Float32Array;
  destroyed: boolean;
}

/**
 * Collect scalar-ref positions from the final plan nodes. f32 only — the
 * legacy `full([], v)` materialization is f32, and non-f32 scalar refs keep
 * the legacy path.
 */
export function collectScalarSlots(planNodes: LazyIRNode[]): ScalarSlot[] {
  const slots: ScalarSlot[] = [];
  for (let ni = 0; ni < planNodes.length; ni++) {
    const inputs = planNodes[ni].inputs;
    for (let ii = 0; ii < inputs.length; ii++) {
      const ref = inputs[ii];
      if (ref.kind === "scalar" && ref.dtype === "f32") {
        slots.push({ nodeIndex: ni, inputIndex: ii });
      }
    }
  }
  return slots;
}

// ----------------------------------------------------------------------------
// Active ref→storage map — set by the executor for the duration of one plan
// execution, consulted by getInputStorage. Ref objects are per-step, so
// identity lookup is exact.
// ----------------------------------------------------------------------------

let activeScalarStorages: Map<LazyRef, StorageHandle> | null = null;

export function lookupScalarStorage(ref: LazyRef): StorageHandle | undefined {
  return activeScalarStorages?.get(ref);
}

export function clearActiveScalarTable(): void {
  activeScalarStorages = null;
}

/** Minimal WebGPU 0-d backend-tensor wrapper around a table buffer. */
function wrapScalarBuffer(buffer: GPUBuffer): StorageHandle {
  return createStorageHandle("webgpu", {
    buffer,
    shape: [],
    dtype: "f32",
    size: 1,
    strides: [],
    offset: 0,
    isContiguous: true,
    ownsBuffer: false,
    toArray() {
      return [];
    },
    destroy() {
      /* plan-owned; destroyed via destroyScalarTable */
    },
    // biome-ignore lint/suspicious/noExplicitAny: minimal BackendTensor shape
  } as any);
}

/**
 * Refresh the plan's scalar table from the CURRENT step's nodes and activate
 * the ref→storage map. Call at the start of every plan execution (before the
 * compiled fast path). Returns immediately for plans without scalar slots.
 */
export function refreshScalarTable(
  loweredPlan: {
    scalarSlots?: ScalarSlot[];
    scalarTable?: PlanScalarTable;
  },
  planNodes: LazyIRNode[],
  backend: Backend,
): void {
  const slots = loweredPlan.scalarSlots;
  if (!slots || slots.length === 0) return;
  if (ENV.TORCHLETTE_SCALAR_TABLE === "0") return; // debug kill switch
  const device = (backend as Backend & { device?: GPUDevice }).device;
  if (!device) return; // non-WebGPU backends keep the legacy full() path

  // [step-tape 1b] declare the table's positions as value-level coverage
  // (guard 3): a scalar that varies step→step at one of these positions is a
  let table = loweredPlan.scalarTable;
  if (!table || table.destroyed) {
    const buffers: GPUBuffer[] = [];
    const storages: StorageHandle[] = [];
    for (let i = 0; i < slots.length; i++) {
      const buf = device.createBuffer({
        size: 4,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      liveScalarTableBuffers.add(buf);
      buffers.push(buf);
      storages.push(wrapScalarBuffer(buf));
    }
    table = {
      buffers,
      storages,
      values: new Float32Array(slots.length).fill(Number.NaN),
      destroyed: false,
    };
    loweredPlan.scalarTable = table;
  }

  const map = new Map<LazyRef, StorageHandle>();
  const scratch = new Float32Array(1);
  let flushed = false;
  for (let i = 0; i < slots.length; i++) {
    const ref = planNodes[slots[i].nodeIndex]?.inputs[slots[i].inputIndex];
    if (!ref || ref.kind !== "scalar") {
      if (ENV.TORCHLETTE_DEBUG_SCALARS) {
        console.log(
          `[scalar-table] STRUCTURAL MISMATCH slot ${i}: planNodes[${slots[i].nodeIndex}].inputs[${slots[i].inputIndex}] is ${ref ? ref.kind : "missing"} (table value stays stale!)`,
        );
      }
      continue; // structural mismatch — skip
    }
    // Compare in f32 space: table.values is a Float32Array MIRROR of the
    // 4-byte GPU buffers, so a non-f32-exact scalar (e.g. 1/√128) stored as
    // f64 never Object.is-equals its own rounded mirror — every step issued
    // a byte-identical 4-byte writeBuffer per slot (28/token on Qwen3
    // decode). Round first; the GPU value is f32 either way. (§8.4 item 4.)
    const v = Math.fround(ref.value);
    if (!Object.is(table.values[i], v)) {
      // Same-template re-execution within one submission scope (e.g. shared
      // block templates) would otherwise make earlier-encoded passes read the
      // later value: queue.writeBuffer lands before the encoder submits. If
      // passes are pending and a value actually changes, submit them first.
      if (!flushed && sharedEncoderActive) {
        flushSharedEncoder();
        flushed = true;
      }
      scratch[0] = v;
      device.queue.writeBuffer(table.buffers[i], 0, scratch);
      table.values[i] = v;
    }
    map.set(ref, table.storages[i]);
  }
  activeScalarStorages = map;
}

/** Module-level registry of LIVE scalar-table buffers. The compiled plan's
 *  persistent-slot pinning must NOT adopt these: their lifetime belongs to
 *  destroyScalarTable, and a plan invalidation that destroyed them would
 *  leave the table writing to dead buffers (caught by the stage-4 stream
 *  determinism gate as "used in submit while destroyed" on table refresh). */
const liveScalarTableBuffers = new Set<unknown>();

export function isScalarTableBuffer(buf: unknown): boolean {
  return liveScalarTableBuffers.has(buf);
}

/** Destroy a plan's scalar-table buffers (template eviction/invalidation). */
export function destroyScalarTable(loweredPlan: {
  scalarTable?: PlanScalarTable;
}): void {
  const table = loweredPlan.scalarTable;
  if (!table || table.destroyed) return;
  table.destroyed = true;
  for (const buf of table.buffers) {
    liveScalarTableBuffers.delete(buf);
    // DEFERRED destruction (fence-gated), never immediate: table buffers can
    // be bound by encoded-but-unsubmitted passes when a template is evicted
    // mid-step (liveness-mode pool pressure evicts arenas + tables while the
    // step's encoder is open). An immediate destroy() poisons the pending
    // submit — Dawn rejects it wholesale and every downstream read sees
    // stale data (the forced-liveness late-LR failure: params silently
    // frozen from the eviction step onward).
    bufferPool.deferredDestroy(buf, 4);
  }
  loweredPlan.scalarTable = undefined;
}
