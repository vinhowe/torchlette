/**
 * Reference counting for StorageHandle lifecycle management.
 *
 * Each StorageHandle starts at rc=0 (no owner). Owners retain/release:
 *   - Tensor._materialize / constructor  →  rcRetain
 *   - Tensor.dispose / _updateLazyRef / GC  →  rcRelease
 *   - View baseStorageId  →  rcRetain on base
 *   - View destruction  →  rcRelease on base
 *
 * A storage is alive when rc > 0, dead when rc <= 0.
 * Views keep their base alive through the base retain — no separate
 * "needed by views" walk is required.
 *
 * Debug tracing: TORCHLETTE_RC_TRACE=1 (warnings) or =verbose (full trail)
 */

const _traceEnv =
  typeof process !== "undefined" ? (process.env?.TORCHLETTE_RC_TRACE ?? "") : "";
const trace = !!_traceEnv;
const verbose = _traceEnv === "verbose";

// storageId → current reference count
const rc = new Map<number, number>();

/** Increment reference count. */
export function rcRetain(storageId: number, site: string): void {
  const cur = rc.get(storageId) ?? 0;
  rc.set(storageId, cur + 1);
  if (verbose) {
    console.warn(`[rc] ${storageId}: retain ${cur}→${cur + 1} @ ${site}`);
  }
}

/** Decrement reference count. Returns the new count. */
export function rcRelease(storageId: number, site: string): number {
  const cur = rc.get(storageId);
  if (cur === undefined) return -1; // unknown storage
  const next = cur - 1;
  rc.set(storageId, next);
  if (trace && next < 0) {
    console.warn(`[rc] DOUBLE-RELEASE: ${storageId} @ ${site} (${cur}→${next})`);
  }
  if (verbose) {
    console.warn(`[rc] ${storageId}: release ${cur}→${next} @ ${site}`);
  }
  return next;
}

/** Read current reference count (-1 if unknown). */
export function rcGet(storageId: number): number {
  return rc.get(storageId) ?? -1;
}

/** Remove entry (after storage is destroyed). */
export function rcDelete(storageId: number): void {
  rc.delete(storageId);
}

/** Reset (for tests). */
export function rcReset(): void {
  rc.clear();
}
