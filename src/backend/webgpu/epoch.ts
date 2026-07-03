/**
 * GPU-timeline epoch advance (scoped-memory stage 0, design doc §1 role 2).
 *
 * `advanceEpoch(reason)` is THE way a quiescent point on the GPU timeline
 * is expressed: it bumps the engine epoch counter (src/core/epoch.ts) and
 * flushes the buffer pool's pendingRelease queue into the available pool.
 *
 * Callers must guarantee the quiescence precondition the pool flush has
 * always required: the work that released those buffers has been SUBMITTED
 * (queue ordering makes reuse safe) — i.e. "encoder closed/flushed", and
 * for the fence-carrying sites, "fence completed". These are exactly the
 * former direct `bufferPool.flushPendingToAvailable()` call sites:
 *   - endSharedEncoder()        (encoder submitted; the per-step epoch)
 *   - endBatchExecution()       (after onSubmittedWorkDone)
 *   - syncWebGPU()/waitForGPU() (after onSubmittedWorkDone)
 *   - readback paths            (after onSubmittedWorkDone)
 * Never call this mid-encoder — flushing pendingRelease while a shared
 * encoder holds encoded-but-unsubmitted passes is the documented
 * deterministic-corruption class (§14.1).
 *
 * NOTE: the deferred-fence path (awaitDeferredFence / flushBufferPool in
 * buffer-pool.ts) bumps the epoch via core `bumpEpoch` directly to avoid
 * an import cycle; the flush there is the same pool primitive.
 */
import { bumpEpoch, currentEpoch } from "../../core/epoch";
import { bufferPool } from "./buffer-pool";

export { currentEpoch };

/** Advance the engine epoch at a GPU-quiescent point and flush the pool's
 *  pendingRelease queue. Returns the new epoch id. */
export function advanceEpoch(reason: string): number {
  const id = bumpEpoch(reason);
  bufferPool.flushPendingToAvailable();
  return id;
}
