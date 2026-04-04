/**
 * GPU buffer lifecycle tracking for debugging aliasing/reuse corruption.
 *
 * Enable: TORCHLETTE_BUF_DEBUG=1       — track + warn on suspicious patterns
 *         TORCHLETTE_BUF_DEBUG=verbose — log every event
 *         TORCHLETTE_BUF_DEBUG=assert  — throw on aliasing
 *
 * Zero-cost when disabled (single boolean check at each hook).
 *
 * Usage pattern:
 *   - bufRegister(buffer, size, caller)   at GPU buffer creation
 *   - bufAcquire(buffer, caller)          when pulled from pool
 *   - bufRelease(buffer, caller)          when released to pool
 *   - bufLogBindings(label, buffers)      at bind group creation (BEFORE dispatch)
 *   - bufMarkWrite(buffer)                for each buffer used as output
 *   - bufBeginScope(name) / bufEndScope() around shared encoder scopes
 *
 * Inspecting when bug hits:
 *   - bufDumpHistory(bufferId)            shows lifecycle + dispatches using it
 *   - bufGetWarnings()                    list of detected aliasing patterns
 */

type GPUBuffer = unknown;

const _env =
  typeof process !== "undefined" ? (process.env?.TORCHLETTE_BUF_DEBUG ?? "") : "";
const enabled = !!_env;
const verbose = _env === "verbose" || _env === "trace";
const assertMode = _env === "assert";

export function bufDebugEnabled(): boolean {
  return enabled;
}

// ============================================================================
// Per-buffer state
// ============================================================================

interface BufferInfo {
  id: number;
  size: number;
  caller: string;
  createdAt: number; // dispatch index at creation
  acquireCount: number;
  releaseCount: number;
  destroyed: boolean;
  // Track release events by dispatch number for pool-reuse filtering
  releasedAtDispatch: number[];
}

const bufferToId = new WeakMap<object, number>();
const bufferInfo = new Map<number, BufferInfo>();
let nextBufferId = 1;

function getOrAssignId(buffer: GPUBuffer): number {
  let id = bufferToId.get(buffer as object);
  if (id === undefined) {
    id = nextBufferId++;
    bufferToId.set(buffer as object, id);
  }
  return id;
}

export function bufId(buffer: GPUBuffer): number {
  if (!enabled) return 0;
  return getOrAssignId(buffer);
}

export function bufRegister(
  buffer: GPUBuffer,
  size: number,
  caller: string,
): void {
  if (!enabled) return;
  const id = getOrAssignId(buffer);
  bufferInfo.set(id, {
    id,
    size,
    caller,
    createdAt: _scopeDispatchCount,
    acquireCount: 0,
    releaseCount: 0,
    destroyed: false,
    releasedAtDispatch: [],
  });
  if (verbose) {
    console.warn(`[buf#${id}] ALLOC size=${size} caller=${caller}`);
  }
}

export function bufAcquire(buffer: GPUBuffer, caller: string): void {
  if (!enabled) return;
  const id = getOrAssignId(buffer);
  const info = bufferInfo.get(id);
  if (info) info.acquireCount++;
  if (verbose) console.warn(`[buf#${id}] ACQUIRE caller=${caller}`);
}

export function bufRelease(buffer: GPUBuffer, caller: string): void {
  if (!enabled) return;
  const id = getOrAssignId(buffer);
  const info = bufferInfo.get(id);
  if (info) {
    info.releaseCount++;
    info.releasedAtDispatch.push(_scopeDispatchCount);
  }
  if (verbose) console.warn(`[buf#${id}] RELEASE caller=${caller}`);
}

export function bufDestroy(buffer: GPUBuffer, caller: string): void {
  if (!enabled) return;
  const id = getOrAssignId(buffer);
  const info = bufferInfo.get(id);
  if (info) info.destroyed = true;
  if (verbose) console.warn(`[buf#${id}] DESTROY caller=${caller}`);
}

// ============================================================================
// Dispatch binding log + aliasing detector
// ============================================================================

interface DispatchInfo {
  dispatchId: number;
  label: string;
  bufferIds: number[];
  writes: Set<number>; // subset marked via bufMarkWrite
}

let _scopeDispatchCount = 0;
let _currentScope = "";
const _dispatches: DispatchInfo[] = [];
// For each buffer ID used in scope: list of (dispatchId, wasWrite) events
const _bufferUsage = new Map<number, Array<{ d: number; w: boolean }>>();
const _warnings: string[] = [];

export function bufBeginScope(name: string): void {
  if (!enabled) return;
  _currentScope = name;
  _scopeDispatchCount = 0;
  _dispatches.length = 0;
  _bufferUsage.clear();
  if (verbose) console.warn(`[buf-scope] BEGIN ${name}`);
}

export function bufEndScope(name: string): void {
  if (!enabled) return;
  if (verbose) {
    console.warn(
      `[buf-scope] END ${name} (${_scopeDispatchCount} dispatches, ${_bufferUsage.size} distinct buffers)`,
    );
  }
  _currentScope = "";
  _scopeDispatchCount = 0;
  _dispatches.length = 0;
  _bufferUsage.clear();
}

/**
 * Log the buffers bound to a bind group (called at bind group creation).
 * This starts a new dispatch record. Call bufMarkWrite for each output.
 */
export function bufLogBindings(
  label: string,
  buffers: readonly GPUBuffer[],
): void {
  if (!enabled) return;
  const dispatchId = ++_scopeDispatchCount;
  const bufferIds = buffers.map(getOrAssignId);
  _dispatches.push({
    dispatchId,
    label,
    bufferIds,
    writes: new Set(),
  });
  if (verbose) {
    console.warn(
      `[dispatch#${dispatchId}] ${label} bindings=[${bufferIds.join(",")}]`,
    );
  }
}

/**
 * Mark a buffer as being written in the CURRENT (most recent) dispatch.
 * Should be called after bufLogBindings and before the next dispatch.
 */
export function bufMarkWrite(buffer: GPUBuffer): void {
  if (!enabled) return;
  const id = getOrAssignId(buffer);
  const current = _dispatches[_dispatches.length - 1];
  if (!current) return;
  current.writes.add(id);

  // Update usage history and check for cross-dispatch aliasing
  const usage = _bufferUsage.get(id) ?? [];
  // Look at previous usages of this buffer in current scope
  const prevWriter = [...usage].reverse().find((u) => u.w);
  const prevReader = [...usage].reverse().find((u) => !u.w);

  // Aliasing pattern: buffer was written in dispatch D1, read in dispatch D2 (D2>D1),
  // now written AGAIN in dispatch D3 (D3>D2). The D2 read expected D1's content;
  // D3 write will overwrite before next read. This is FINE (implicit barrier).
  //
  // REAL corruption pattern: buffer is read in dispatch D1 as input, then WRITTEN
  // in a LATER dispatch D2 that was supposed to write a DIFFERENT buffer. This
  // happens if the pool reuses a buffer still referenced by earlier bind groups.
  //
  // We can't detect pool-reuse aliasing directly here (bind groups are opaque by
  // the time we dispatch). But we CAN detect "same buffer written twice without
  // barrier explicitly flushed" — which is the shared-encoder reuse class.
  if (prevWriter && !prevReader) {
    // Writer → Writer without a read in between. Check if the buffer was
    // RELEASED to the pool between the two writes — if so, this is legitimate
    // pool reuse (different storages sharing the same buffer over time).
    const info = bufferInfo.get(id);
    const releasedBetween = info?.releasedAtDispatch.some(
      (d) => d > prevWriter.d && d <= current.dispatchId,
    );
    if (!releasedBetween) {
      const msg = `[aliasing] buf#${id} written twice without read or release: first by dispatch #${prevWriter.d}, now by dispatch #${current.dispatchId} "${current.label}"`;
      _warnings.push(msg);
      if (verbose || assertMode) console.warn(msg);
      if (assertMode) throw new Error(msg);
    }
  }
  usage.push({ d: current.dispatchId, w: true });
  _bufferUsage.set(id, usage);
}

/**
 * After bufLogBindings, call this to record reads (non-write bindings).
 * Call bufMarkWrite first for all writes; remaining bindings are assumed reads.
 */
export function bufFinalizeBindings(): void {
  if (!enabled) return;
  const current = _dispatches[_dispatches.length - 1];
  if (!current) return;
  for (const id of current.bufferIds) {
    if (current.writes.has(id)) continue;
    const usage = _bufferUsage.get(id) ?? [];
    usage.push({ d: current.dispatchId, w: false });
    _bufferUsage.set(id, usage);
  }
}

// ============================================================================
// Reporting
// ============================================================================

export function bufGetWarnings(): readonly string[] {
  return _warnings;
}

export function bufResetWarnings(): void {
  _warnings.length = 0;
}

export function bufDumpHistory(bufferId: number): void {
  if (!enabled) return;
  const info = bufferInfo.get(bufferId);
  if (!info) {
    console.warn(`[buf#${bufferId}] NOT TRACKED`);
    return;
  }
  console.warn(
    `[buf#${bufferId}] size=${info.size} caller=${info.caller} acquire=${info.acquireCount} release=${info.releaseCount} destroyed=${info.destroyed}`,
  );
  const usage = _bufferUsage.get(bufferId) ?? [];
  for (const u of usage) {
    const d = _dispatches.find((x) => x.dispatchId === u.d);
    console.warn(
      `  dispatch #${u.d} "${d?.label ?? "?"}": ${u.w ? "WRITE" : "READ"}`,
    );
  }
}

export function bufDumpState(): void {
  if (!enabled) return;
  console.warn(
    `=== Buffer Debug: scope="${_currentScope}" ${_scopeDispatchCount} dispatches ${_bufferUsage.size} buffers ${_warnings.length} warnings ===`,
  );
  for (const w of _warnings.slice(0, 30)) console.warn(`  ${w}`);
}

/** Find buffers that were written in this scope. */
export function bufFindWriters(bufferId: number): number[] {
  if (!enabled) return [];
  const usage = _bufferUsage.get(bufferId) ?? [];
  return usage.filter((u) => u.w).map((u) => u.d);
}

/** Find buffers that were read in this scope. */
export function bufFindReaders(bufferId: number): number[] {
  if (!enabled) return [];
  const usage = _bufferUsage.get(bufferId) ?? [];
  return usage.filter((u) => !u.w).map((u) => u.d);
}
